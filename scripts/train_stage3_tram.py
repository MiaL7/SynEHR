from __future__ import annotations
import argparse
import csv
import json
import os
import random
import sys
from pathlib import Path
from typing import Optional
import numpy as np
import torch
from huggingface_hub import login as hf_login
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_from_disk
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from synehr.models.backbone import BACKBONE_SPECS, load_frozen_backbone, register_norm_hook, get_h_last, load_frozen_time_head
from synehr.models.tscm import TIME_EMB_DIM
from synehr.models.tram import RelationAdapter, N_PREFIX
from synehr.utils.adapter_utils import load_stage1_embeddings
from synehr.data.dataset import load_timecond_raw_splits, StepTimeCDataset, make_timecond_collate_fn

MAX_LEN = 6144
ABLATIONS = [{
    'name': 'static_only',
    'disable_static': False,
    'disable_dynamic': True,
    'random_zt': False,
    'film_gate': False,
    'note': 'Static branch only (no temporal conditioning)'
}, {
    'name': 'dynamic_only',
    'disable_static': True,
    'disable_dynamic': False,
    'random_zt': False,
    'film_gate': False,
    'note': 'Dynamic branch only (no static co-occurrence)'
}, {
    'name': 'full_tram',
    'disable_static': False,
    'disable_dynamic': False,
    'random_zt': False,
    'film_gate': False,
    'note': 'Full TRAM (static + dynamic FiLM)'
}, {
    'name': 'full_tram_random_zt',
    'disable_static': False,
    'disable_dynamic': False,
    'random_zt': True,
    'film_gate': False,
    'note': 'Full TRAM but z_t = random noise (ablates TSCM signal)'
}, {
    'name': 'full_tram_sigmoid',
    'disable_static': False,
    'disable_dynamic': False,
    'random_zt': False,
    'film_gate': True,
    'note': 'Full TRAM with sigmoid gate instead of FiLM'
}]


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)


def _adapter_scales(adapter: RelationAdapter) -> tuple[float, float]:
    return (float(adapter.alpha_s.detach()), float(adapter.alpha_d.detach()))


def _check_finite(name, tensor, *, run_name, split_name):
    if not torch.isfinite(tensor).all():
        raise FloatingPointError(
            f'Non-finite {name} in {run_name} [{split_name}]')


class _SigmoidGateDynamic(nn.Module):
    def __init__(self, hidden_size, time_emb_dim, rel_dim):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.h_proj = nn.Linear(hidden_size, rel_dim)
        self.gate = nn.Linear(time_emb_dim, rel_dim)
        self.q_proj = nn.Linear(3, rel_dim)
        self.out_proj = nn.Sequential(nn.Linear(2 * rel_dim, rel_dim),
                                      nn.GELU(), nn.LayerNorm(rel_dim))

    def forward(self, h, z_t, q3_input):
        z_h = self.h_proj(self.norm(h)) * torch.sigmoid(self.gate(z_t))
        z_q = self.q_proj(q3_input)

        return self.out_proj(torch.cat([z_h, z_q], dim=-1))


class _SigmoidGateAdapter(RelationAdapter):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.dynamic_branch = _SigmoidGateDynamic(
            hidden_size=kw.get('hidden_size', 4096),
            time_emb_dim=kw.get('time_emb_dim', TIME_EMB_DIM),
            rel_dim=kw.get('rel_dim', 128))


def run_step_epoch(model,
                   time_head,
                   adapter,
                   loader,
                   optimizer,
                   scheduler,
                   h_cache,
                   spec,
                   device,
                   n_prefix=N_PREFIX,
                   train=True,
                   grad_accum=1,
                   run_name='tram') -> dict:
    adapter.train(train)
    emb_layer = spec.get_embed_layer(model)
    total_loss = total_tokens = total_conf = total_samples = accum_step = 0

    if train:
        optimizer.zero_grad()

    with torch.set_grad_enabled(train):
        for batch in tqdm(loader,
                          desc='train' if train else 'eval ',
                          leave=False):
            hist_ids = batch['hist_ids'].to(device)
            hist_mask = batch['hist_mask'].to(device)
            full_ids = batch['full_ids'].to(device)
            full_mask = batch['full_mask'].to(device)
            labels = batch['labels'].to(device)
            dx_ids = batch['dx_ids'].to(device)
            proc_ids = batch['proc_ids'].to(device)
            med_ids = batch['med_ids'].to(device)
            lab_ids = batch['lab_ids'].to(device)
            vtype_ids = batch['vtype_ids'].to(device)
            gap_days = batch['gap_days'].to(device)
            gap_missing = batch['gap_missing'].to(device)
            visit_mask = batch['visit_mask'].to(device)
            demo_feats = batch['demo_feats'].to(device)
            B = full_ids.shape[0]
            with torch.no_grad():
                model(input_ids=hist_ids, attention_mask=hist_mask)
            h_last = get_h_last(h_cache, hist_mask, spec)
            with torch.no_grad():
                t_out = time_head(dx_ids=dx_ids,
                                  proc_ids=proc_ids,
                                  med_ids=med_ids,
                                  lab_ids=lab_ids,
                                  vtype_ids=vtype_ids,
                                  gap_days=gap_days,
                                  gap_missing=gap_missing,
                                  visit_mask=visit_mask,
                                  demo_feats=demo_feats)
                z_t = t_out['z_t']
                q3 = t_out['q3']
                conf_t = t_out['conf']
            total_conf += float(conf_t.sum())
            total_samples += B
            split_name = 'train' if train else 'eval'
            P_emb = adapter(h_last.detach(),
                            z_t.detach(),
                            conf_t.detach(),
                            q3_input=q3.detach(),
                            dx_ids=dx_ids,
                            proc_ids=proc_ids,
                            med_ids=med_ids,
                            lab_ids=lab_ids,
                            vtype_ids=vtype_ids,
                            visit_mask=visit_mask)
            _check_finite('P_emb',
                          P_emb,
                          run_name=run_name,
                          split_name=split_name)
            with torch.no_grad():
                input_embeds = emb_layer(full_ids)
            P_emb_cast = P_emb.to(input_embeds.dtype)
            extended_emb = torch.cat([P_emb_cast, input_embeds], dim=1)
            prefix_ones = torch.ones(B,
                                     n_prefix,
                                     dtype=full_mask.dtype,
                                     device=device)
            extended_mask = torch.cat([prefix_ones, full_mask], dim=1)
            outputs = model(inputs_embeds=extended_emb,
                            attention_mask=extended_mask,
                            use_cache=False)
            logits = outputs.logits
            _check_finite('logits',
                          logits,
                          run_name=run_name,
                          split_name=split_name)
            logit_slice = logits[:, n_prefix:-1, :]
            target_slice = full_ids[:, 1:]
            mask_slice = labels[:, 1:] != -100
            n_tokens = int(mask_slice.sum())
            if n_tokens == 0:
                continue
            loss = F.cross_entropy(logit_slice[mask_slice].float(),
                                   target_slice[mask_slice])
            _check_finite('loss',
                          loss,
                          run_name=run_name,
                          split_name=split_name)
            if train:
                (loss / grad_accum).backward()
                accum_step += 1
                if accum_step % grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()
            total_loss += loss.item() * n_tokens
            total_tokens += n_tokens

    if train and accum_step % grad_accum != 0:
        torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()

    if train and scheduler is not None:
        scheduler.step()

    alpha_s, alpha_d = _adapter_scales(adapter)

    return {
        'lm_loss': total_loss / max(total_tokens, 1),
        'lm_ppl':
        float(torch.exp(torch.tensor(total_loss / max(total_tokens, 1)))),
        'n_tokens': total_tokens,
        'alpha_s': alpha_s,
        'alpha_d': alpha_d,
        'mean_conf_t': total_conf / max(total_samples, 1)
    }


def train_condition(config,
                    model,
                    tokenizer,
                    time_head,
                    h_cache,
                    spec,
                    train_ds,
                    eval_ds,
                    output_dir,
                    args,
                    device,
                    stage1_emb,
                    selection_split_name='val') -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    run_name = output_dir.name
    print(
        f"\n{'#' * 60}\n# Condition: {config['name']}  |  backbone: {spec.backbone_key}"
    )
    print(f"# {config['note']}\n{'#' * 60}")
    AdapterClass = _SigmoidGateAdapter if config[
        'film_gate'] else RelationAdapter

    adapter = AdapterClass(stage1_emb=stage1_emb,
                           hidden_size=spec.hidden_size,
                           time_emb_dim=TIME_EMB_DIM,
                           rel_dim=args.rel_dim,
                           n_prefix=args.n_prefix,
                           disable_static=config['disable_static'],
                           disable_dynamic=config['disable_dynamic'],
                           alpha_s_init=args.alpha_s_init,
                           alpha_d_init=args.alpha_d_init).to(device).to(
                               torch.float32)

    print(f'  Trainable adapter params: {adapter.n_trainable_params:,}')

    if config['random_zt']:
        _orig_fwd = adapter.forward

        def _noisy_fwd(h, z_t, conf_t=None, **kw):
            return _orig_fwd(h, torch.randn_like(z_t), conf_t, **kw)

        adapter.forward = _noisy_fwd

    optimizer = AdamW(adapter.parameters(), lr=args.lr, weight_decay=0.01)
    cosine_epochs = max(args.epochs - args.warmup_epochs, 1)
    scheduler = SequentialLR(optimizer,
                             schedulers=[
                                 LinearLR(optimizer,
                                          start_factor=0.1,
                                          end_factor=1.0,
                                          total_iters=args.warmup_epochs),
                                 CosineAnnealingLR(optimizer,
                                                   T_max=cosine_epochs,
                                                   eta_min=args.lr * 0.1)
                             ],
                             milestones=[args.warmup_epochs])

    collate_fn = make_timecond_collate_fn(tokenizer.pad_token_id)
    train_loader = DataLoader(train_ds,
                              batch_size=args.bs,
                              shuffle=True,
                              collate_fn=collate_fn,
                              num_workers=args.num_workers,
                              pin_memory=torch.cuda.is_available())

    eval_loader = DataLoader(eval_ds,
                             batch_size=args.bs,
                             shuffle=False,
                             collate_fn=collate_fn,
                             num_workers=args.num_workers,
                             pin_memory=torch.cuda.is_available())

    best_val_loss = float('inf')
    log_rows = []

    for epoch in range(1, args.epochs + 1):
        print(f"\n{'=' * 60}\nEpoch {epoch}/{args.epochs}  [{run_name}]")
        tr = run_step_epoch(model,
                            time_head,
                            adapter,
                            train_loader,
                            optimizer,
                            scheduler,
                            h_cache,
                            spec,
                            device,
                            n_prefix=args.n_prefix,
                            train=True,
                            grad_accum=args.grad_accum,
                            run_name=run_name)
        ev = run_step_epoch(model,
                            time_head,
                            adapter,
                            eval_loader,
                            None,
                            None,
                            h_cache,
                            spec,
                            device,
                            n_prefix=args.n_prefix,
                            train=False,
                            grad_accum=1,
                            run_name=run_name)
        print(
            f"  train | lm_loss={tr['lm_loss']:.4f}  α_s={tr['alpha_s']:.4f}  α_d={tr['alpha_d']:.4f}"
        )
        print(
            f"  {selection_split_name:<5}| lm_loss={ev['lm_loss']:.4f}  α_s={ev['alpha_s']:.4f}"
        )
        ckpt_path = output_dir / f'epoch_{epoch}.pt'
        torch.save(
            {
                'epoch': epoch,
                'state_dict': adapter.state_dict(),
                'train_metrics': tr,
                'eval_metrics': ev
            }, ckpt_path)
        if ev['lm_loss'] < best_val_loss:
            best_val_loss = ev['lm_loss']
            torch.save(adapter.state_dict(), output_dir / 'best.pt')
            print(
                f'  ★ New best {selection_split_name}_loss={best_val_loss:.4f}'
            )
        log_rows.append({'epoch': epoch, 'split': 'train', **tr})
        log_rows.append({'epoch': epoch, 'split': selection_split_name, **ev})

    with (output_dir / 'training_log.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(log_rows[0].keys()))
        writer.writeheader()
        writer.writerows(log_rows)

    meta = {
        'backbone': spec.backbone_key,
        'model_name': spec.model_name,
        'hidden_size': spec.hidden_size,
        'adapter_version': 'v2_struct_static',
        'lora_ckpt': str(args.lora),
        'time_head_ckpt': str(args.tscm_ckpt),
        'stage1_ckpt': str(args.stage1_ckpt),
        'output_dir': str(output_dir),
        'condition': config['name'],
        'note': config['note'],
        'disable_static': config['disable_static'],
        'disable_dynamic': config['disable_dynamic'],
        'random_zt': config['random_zt'],
        'film_gate': config['film_gate'],
        'epochs': args.epochs,
        'bs': args.bs,
        'grad_accum': args.grad_accum,
        'lr': args.lr,
        'warmup_epochs': args.warmup_epochs,
        'rel_dim': args.rel_dim,
        'n_prefix': args.n_prefix,
        'alpha_s_init': args.alpha_s_init,
        'alpha_d_init': args.alpha_d_init,
        'use_regime_confidence': adapter.use_regime_confidence,
        'branch_local_norm': adapter.branch_local_norm,
        'use_time_conditioning': adapter.use_time_conditioning,
        'selection_split': selection_split_name,
        'best_val_loss': best_val_loss
    }

    with (output_dir / 'run_metadata.json').open('w') as f:
        json.dump(meta, f, indent=2)

    print(
        f"  Best → {output_dir / 'best.pt'}  ({selection_split_name}_loss={best_val_loss:.4f})"
    )


def main():
    parser = argparse.ArgumentParser(
        description='Stage 3: TRAM training for SynEHR.')

    parser.add_argument('--backbone',
                        choices=list(BACKBONE_SPECS),
                        default='llama31')

    parser.add_argument('--lora',
                        type=Path,
                        required=True,
                        help='Stage-1 LoRA checkpoint directory')

    parser.add_argument('--tscm-ckpt',
                        type=Path,
                        required=True,
                        help='Stage-2 TSCM .pt checkpoint')

    parser.add_argument(
        '--stage1-ckpt',
        type=Path,
        required=True,
        help='Stage-1 .pt checkpoint for TRAM embedding initialization')

    parser.add_argument('--stage1-data-dir',
                        type=Path,
                        required=True,
                        help='Directory containing code_vocabs.json')

    parser.add_argument('--train-jsonl',
                        type=Path,
                        required=True,
                        help='MIMIC train JSONL path')

    parser.add_argument('--test-jsonl',
                        type=Path,
                        required=True,
                        help='MIMIC test JSONL path')

    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--bs', type=int, default=2)
    parser.add_argument('--grad-accum', type=int, default=8)
    parser.add_argument('--warmup-epochs', type=int, default=1)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--rel-dim', type=int, default=128)
    parser.add_argument('--n-prefix', type=int, default=1)
    parser.add_argument('--alpha-s-init', type=float, default=0.1)
    parser.add_argument('--alpha-d-init', type=float, default=0.03)
    parser.add_argument(
        '--only',
        default=None,
        help='Comma-separated condition names to run (e.g. full_tram)')

    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--hf-token',
                        default=None,
                        help='HuggingFace token for gated models (e.g. LLaMA)')

    args = parser.parse_args()

    if args.hf_token:
        hf_login(token=args.hf_token)

    set_seed(args.seed)
    spec = BACKBONE_SPECS[args.backbone]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}  |  backbone: {spec.backbone_key}')

    for path, label in [(args.lora, 'LoRA checkpoint'),
                        (args.tscm_ckpt, 'TSCM checkpoint'),
                        (args.stage1_ckpt, 'Stage-1 embedding checkpoint')]:
        if not path.exists():
            print(f'ERROR: {label} not found: {path}')
            sys.exit(1)

    model, tokenizer = load_frozen_backbone(spec, args.lora, device)
    h_cache, hook_handle = register_norm_hook(model, spec)
    print(f'Loading TSCM from {args.tscm_ckpt} …')
    time_head = load_frozen_time_head(args.tscm_ckpt, spec.hidden_size, device)
    print(f'Loading Stage-1 embeddings from {args.stage1_ckpt} …')
    stage1_emb = load_stage1_embeddings(args.stage1_ckpt, device)
    code_vocabs_path = args.stage1_data_dir / 'code_vocabs.json'

    if not code_vocabs_path.exists():
        print(f'ERROR: code_vocabs.json not found: {code_vocabs_path}')
        sys.exit(1)

    print('Loading dataset …')
    raw_splits = load_timecond_raw_splits(train_jsonl=args.train_jsonl,
                                          test_jsonl=args.test_jsonl)

    selection_split = 'val'
    train_ds = StepTimeCDataset(raw_splits['train'],
                                tokenizer,
                                'train',
                                MAX_LEN,
                                code_vocabs_path=code_vocabs_path)

    eval_ds = StepTimeCDataset(raw_splits['val'],
                               tokenizer,
                               'val',
                               MAX_LEN,
                               code_vocabs_path=code_vocabs_path)

    if len(train_ds) == 0 or len(eval_ds) == 0:
        print('ERROR: empty dataset.')
        sys.exit(1)

    only_set = set(args.only.split(',')) if args.only else None
    conditions = [
        a for a in ABLATIONS if only_set is None or a['name'] in only_set
    ]
    print(f"\nConditions to run: {[c['name'] for c in conditions]}")
    output_root = Path(args.output_root)

    for config in conditions:
        train_condition(config=config,
                        model=model,
                        tokenizer=tokenizer,
                        time_head=time_head,
                        h_cache=h_cache,
                        spec=spec,
                        train_ds=train_ds,
                        eval_ds=eval_ds,
                        output_dir=output_root / config['name'],
                        args=args,
                        device=device,
                        stage1_emb=stage1_emb,
                        selection_split_name=selection_split)

    hook_handle.remove()
    print('\nAll conditions done.')


if __name__ == '__main__':
    main()
