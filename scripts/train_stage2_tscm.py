from __future__ import annotations
import csv
import json
import os
import random
import sys
from pathlib import Path
import numpy as np
import torch
from huggingface_hub import login as hf_login
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from synehr.models.backbone import BACKBONE_SPECS, load_frozen_backbone
from synehr.models.tscm import TimeHead, discrete_time_survival_nll, gaussian_nll_log_days
from synehr.utils.adapter_utils import load_stage1_embeddings
from synehr.data.dataset import load_timecond_raw_splits, StepTimeCDataset, make_timecond_collate_fn

MAX_LEN = 6144
LAMBDA_DIST = 1.0
LAMBDA_REG = 1.0


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)


def compute_regime_weights(dataset) -> torch.Tensor:
    labels = torch.tensor([int(r['gt_regime3']) for r in dataset.records],
                          dtype=torch.long)

    counts = torch.bincount(labels, minlength=3).float().clamp(min=1.0)
    weights = 1.0 / counts

    return weights / weights.sum() * 3.0


def compute_demo_stats(dataset) -> tuple[torch.Tensor, torch.Tensor]:
    feats = torch.stack([r['demo_feats'] for r in dataset.records], dim=0)

    return (feats.mean(dim=0), feats.std(dim=0).clamp(min=1e-06))


def run_epoch(time_head,
              loader,
              optimizer,
              scheduler,
              device,
              regime_weight=None,
              train=True) -> dict:
    time_head.train(train)
    total_haz = total_dist = total_reg = total_loss = 0.0
    total_conf = total_scale_conf = 0.0
    all_pred, all_true = ([], [])
    n = 0
    weight = regime_weight.to(device) if regime_weight is not None else None

    with torch.set_grad_enabled(train):
        for batch in tqdm(loader,
                          desc='train' if train else 'eval ',
                          leave=False):
            dx_ids = batch['dx_ids'].to(device)
            proc_ids = batch['proc_ids'].to(device)
            med_ids = batch['med_ids'].to(device)
            lab_ids = batch['lab_ids'].to(device)
            vtype_ids = batch['vtype_ids'].to(device)
            gap_days = batch['gap_days'].to(device)
            gap_missing = batch['gap_missing'].to(device)
            visit_mask = batch['visit_mask'].to(device)
            demo_feats = batch['demo_feats'].to(device)
            target_days = torch.tensor(batch['target_days'],
                                       dtype=torch.float32,
                                       device=device)
            target_fine_bin = torch.tensor(batch['target_fine_bin'],
                                           dtype=torch.long,
                                           device=device)
            target_regime3 = torch.tensor(batch['gt_regime3'],
                                          dtype=torch.long,
                                          device=device)
            out = time_head(dx_ids=dx_ids,
                            proc_ids=proc_ids,
                            med_ids=med_ids,
                            lab_ids=lab_ids,
                            vtype_ids=vtype_ids,
                            gap_days=gap_days,
                            gap_missing=gap_missing,
                            visit_mask=visit_mask,
                            demo_feats=demo_feats)
            loss_haz = discrete_time_survival_nll(out['hazard'],
                                                  target_fine_bin)
            loss_dist = gaussian_nll_log_days(target_days, out['mu'],
                                              out['log_s'])
            loss_reg = F.nll_loss(torch.log(out['q3'].clamp(min=1e-08)),
                                  target_regime3,
                                  weight=weight)
            loss = loss_haz + LAMBDA_DIST * loss_dist + LAMBDA_REG * loss_reg
            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(time_head.parameters(), 1.0)
                optimizer.step()
            bsz = dx_ids.size(0)
            n += bsz
            total_haz += loss_haz.item() * bsz
            total_dist += loss_dist.item() * bsz
            total_reg += loss_reg.item() * bsz
            total_loss += loss.item() * bsz
            total_conf += float(out['conf'].mean()) * bsz
            total_scale_conf += float(
                (1.0 - torch.sigmoid(out['log_s'])).mean()) * bsz
            all_pred.extend(out['q3'].argmax(dim=-1).detach().cpu().tolist())
            all_true.extend(target_regime3.detach().cpu().tolist())

    if train and scheduler is not None:
        scheduler.step()

    from sklearn.metrics import f1_score as sk_f1
    macro_f1 = float(
        sk_f1(all_true, all_pred, average='macro', zero_division=0))

    acc = float(sum((p == t for p, t in zip(all_pred, all_true))) / max(n, 1))

    return {
        'loss': total_loss / max(n, 1),
        'haz_loss': total_haz / max(n, 1),
        'dist_loss': total_dist / max(n, 1),
        'reg_loss': total_reg / max(n, 1),
        'regime_acc': acc,
        'regime_macro_f1': macro_f1,
        'mean_conf': total_conf / max(n, 1),
        'mean_scale_conf': total_scale_conf / max(n, 1)
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description='Stage 2: TSCM training for SynEHR.')

    parser.add_argument('--backbone',
                        choices=list(BACKBONE_SPECS),
                        default='llama31')

    parser.add_argument('--ckpt',
                        type=Path,
                        required=True,
                        help='Stage-1 LoRA checkpoint directory')

    parser.add_argument('--stage1-ckpt',
                        type=Path,
                        required=True,
                        help='Frozen code embedding checkpoint')

    parser.add_argument('--stage1-data-dir',
                        type=Path,
                        required=True,
                        help='Directory containing code_vocabs.json')

    parser.add_argument('--train-jsonl', type=Path, required=True)
    parser.add_argument('--test-jsonl', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--bs', type=int, default=8)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--lr', type=float, default=0.001)
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
    output_dir = Path(args.output_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not args.ckpt.exists():
        print(f'ERROR: Stage-1 LoRA checkpoint not found: {args.ckpt}')
        sys.exit(1)

    if not args.stage1_ckpt.exists():
        print(
            f'ERROR: stage1 embedding checkpoint not found: {args.stage1_ckpt}'
        )
        sys.exit(1)

    code_vocabs_path = args.stage1_data_dir / 'code_vocabs.json'

    if not code_vocabs_path.exists():
        print(f'ERROR: code_vocabs.json not found: {code_vocabs_path}')
        sys.exit(1)

    stage1_emb = load_stage1_embeddings(args.stage1_ckpt, device='cpu')
    _, tokenizer = load_frozen_backbone(spec, args.ckpt, device)
    print('Loading dataset …')
    raw_splits = load_timecond_raw_splits(train_jsonl=args.train_jsonl,
                                          test_jsonl=args.test_jsonl)

    selection_split = 'val'
    train_ds = StepTimeCDataset(raw_splits['train'],
                                tokenizer,
                                'train',
                                MAX_LEN,
                                code_vocabs_path=code_vocabs_path)

    eval_ds = StepTimeCDataset(raw_splits[selection_split],
                               tokenizer,
                               selection_split,
                               MAX_LEN,
                               code_vocabs_path=code_vocabs_path)

    if len(train_ds) == 0 or len(eval_ds) == 0:
        print('ERROR: empty dataset.')
        sys.exit(1)

    collate_fn = make_timecond_collate_fn(tokenizer.pad_token_id)
    loader_kw = dict(collate_fn=collate_fn,
                     num_workers=args.num_workers,
                     pin_memory=torch.cuda.is_available())

    train_loader = DataLoader(train_ds,
                              batch_size=args.bs,
                              shuffle=True,
                              **loader_kw)

    eval_loader = DataLoader(eval_ds,
                             batch_size=args.bs,
                             shuffle=False,
                             **loader_kw)

    time_head = TimeHead(stage1_emb=stage1_emb).to(device).to(torch.float32)
    demo_mean, demo_std = compute_demo_stats(train_ds)
    time_head.set_demo_stats(demo_mean, demo_std)
    regime_weight = compute_regime_weights(train_ds)
    optimizer = AdamW(time_head.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    best_eval_f1 = -1.0
    log_rows = []

    for epoch in range(1, args.epochs + 1):
        print(f"\n{'=' * 60}\nEpoch {epoch}/{args.epochs}")
        tr = run_epoch(time_head,
                       train_loader,
                       optimizer,
                       scheduler,
                       device,
                       regime_weight=regime_weight,
                       train=True)
        ev = run_epoch(time_head,
                       eval_loader,
                       None,
                       None,
                       device,
                       regime_weight=regime_weight,
                       train=False)
        print(
            f"  train | loss={tr['loss']:.4f} haz={tr['haz_loss']:.4f} dist={tr['dist_loss']:.4f} reg={tr['reg_loss']:.4f} f1={tr['regime_macro_f1']:.3f}"
        )
        print(
            f"  {selection_split:<5}| loss={ev['loss']:.4f} haz={ev['haz_loss']:.4f} dist={ev['dist_loss']:.4f} reg={ev['reg_loss']:.4f} f1={ev['regime_macro_f1']:.3f}"
        )
        ckpt_ep = output_dir / f'epoch_{epoch}.pt'
        torch.save(
            {
                'epoch': epoch,
                'state_dict': time_head.state_dict(),
                'train_metrics': tr,
                'eval_metrics': ev
            }, ckpt_ep)
        if ev['regime_macro_f1'] > best_eval_f1:
            best_eval_f1 = ev['regime_macro_f1']
            torch.save(time_head.state_dict(), output_dir / 'best.pt')
            print(
                f'  ★ New best ({selection_split} macro_f1={best_eval_f1:.4f})'
            )
        log_rows.append({'epoch': epoch, 'split': 'train', **tr})
        log_rows.append({'epoch': epoch, 'split': selection_split, **ev})

    with (output_dir / 'training_log.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(log_rows[0].keys()))
        writer.writeheader()
        writer.writerows(log_rows)

    meta = {
        'backbone': spec.backbone_key,
        'model_name': spec.model_name,
        'ckpt': str(args.ckpt),
        'stage1_ckpt': str(args.stage1_ckpt),
        'epochs': args.epochs,
        'bs': args.bs,
        'lr': args.lr,
        'lambda_dist': LAMBDA_DIST,
        'lambda_reg': LAMBDA_REG,
        'selection_split': selection_split,
        'best_val_macro_f1': best_eval_f1,
        'best_checkpoint': str(output_dir / 'best.pt')
    }

    with (output_dir / 'run_metadata.json').open('w') as f:
        json.dump(meta, f, indent=2)

    print(
        f"\nBest → {output_dir / 'best.pt'}  ({selection_split} macro_f1={best_eval_f1:.4f})"
    )


if __name__ == '__main__':
    main()
