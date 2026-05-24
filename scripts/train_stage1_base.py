from __future__ import annotations
import csv
import hashlib
import json
import multiprocessing as mp
import os
import random
import shutil
import sys
from pathlib import Path
from typing import Any
import numpy as np
import torch
from datasets import load_from_disk
from peft import LoraConfig, get_peft_model
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset as TorchDataset
from tqdm.auto import tqdm
from huggingface_hub import login as hf_login
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
from synehr.models.backbone import BACKBONE_SPECS
from synehr.data.serialization import VISIT_EOS_TOKEN, fit_step_messages_by_visit

MAX_LEN = 6144
MP_CHUNKSIZE = 8
ADDED_TOKEN_STATE_NAME = 'added_token_rows.pt'


def _default_preprocess_workers() -> int:
    slurm_cpus = os.environ.get('SLURM_CPUS_PER_TASK')

    if slurm_cpus:
        try:
            return max(1, int(slurm_cpus))
        except ValueError:
            pass

    return max(1, min(os.cpu_count() or 1, 4))


def _cache_suffix_for_dataset(hf_dataset, *, backbone_key: str, mode: str,
                              max_len: int) -> str:
    fingerprint = str(getattr(hf_dataset, '_fingerprint', 'no_fingerprint'))
    fingerprint = hashlib.sha1(fingerprint.encode('utf-8')).hexdigest()[:12]

    return f'{mode}-{backbone_key}-ml{max_len}-{fingerprint}'


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)


def _asst_indices(messages: list) -> list[int]:
    return [i for i, m in enumerate(messages) if m['role'] == 'assistant']


def _ensure_visit_eos_token(tokenizer) -> None:
    specials = tokenizer.special_tokens_map.get('additional_special_tokens',
                                                [])

    if VISIT_EOS_TOKEN not in specials:
        tokenizer.add_special_tokens(
            {'additional_special_tokens': specials + [VISIT_EOS_TOKEN]})

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = 'right'


def _added_token_ids(tokenizer, base_vocab_size: int) -> list[int]:
    added_vocab = tokenizer.get_added_vocab()

    return sorted(
        (token_id for token_id in added_vocab.values()
         if isinstance(token_id, int) and token_id >= base_vocab_size))


def _build_added_token_state(model, tokenizer,
                             base_vocab_size: int) -> dict[str, Any] | None:
    token_ids = _added_token_ids(tokenizer, base_vocab_size)

    if not token_ids:
        return None

    input_emb = model.get_input_embeddings()
    output_emb = model.get_output_embeddings()
    max_token_id = max(token_ids)

    if input_emb is None or input_emb.weight.shape[0] <= max_token_id:
        raise RuntimeError(
            'Input embedding shape does not cover the added token ids.')

    if output_emb is not None and output_emb.weight.shape[0] <= max_token_id:
        raise RuntimeError(
            'Output embedding shape does not cover the added token ids.')

    state: dict[str, Any] = {
        'base_vocab_size': base_vocab_size,
        'tokenizer_len': len(tokenizer),
        'token_ids': token_ids,
        'tokens':
        {str(t): tokenizer.convert_ids_to_tokens(t)
         for t in token_ids},
        'input_embeddings': input_emb.weight.detach()[token_ids].cpu().clone()
    }

    if output_emb is not None:
        state['output_embeddings'] = output_emb.weight.detach()[token_ids].cpu(
        ).clone()

    return state


def _save_cache_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + '.tmp')
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)


def _load_cache(path: Path) -> dict[str, Any]:
    return torch.load(path, map_location='cpu')


def _save_adapter_checkpoint(model, tokenizer, ckpt_dir: Path,
                             base_vocab_size: int) -> None:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    added_token_state = _build_added_token_state(model, tokenizer,
                                                 base_vocab_size)

    if added_token_state is not None:
        _save_cache_atomic(ckpt_dir / ADDED_TOKEN_STATE_NAME,
                           added_token_state)

    try:
        model.save_pretrained(str(ckpt_dir), save_embedding_layers=False)

    except TypeError:
        model.save_pretrained(str(ckpt_dir))


def _replace_with_symlink(link_path: Path, target_dir: Path) -> None:
    if link_path.is_symlink() or link_path.is_file():
        link_path.unlink()

    elif link_path.is_dir():
        shutil.rmtree(link_path)

    rel_target = os.path.relpath(target_dir, start=link_path.parent)
    link_path.symlink_to(rel_target, target_is_directory=True)


def _build_step_sample(messages, asst_pos, tokenizer, max_len):
    fitted = fit_step_messages_by_visit(messages, asst_pos, tokenizer, max_len)

    if fitted is None:
        return None

    _, _, prompt_enc, full_enc = fitted
    prompt_len = len(prompt_enc['input_ids'])
    full_ids = full_enc['input_ids']
    attn_mask = full_enc.get('attention_mask', [1] * len(full_ids))
    labels = [-100] * prompt_len + full_ids[prompt_len:]

    return {
        'input_ids': torch.tensor(full_ids, dtype=torch.long),
        'attention_mask': torch.tensor(attn_mask, dtype=torch.long),
        'labels': torch.tensor(labels, dtype=torch.long)
    }


def _build_trajectory_labels(messages, tokenizer, max_len):
    try:
        full_text = tokenizer.apply_chat_template(messages,
                                                  tokenize=False,
                                                  add_generation_prompt=False)
        full_enc = tokenizer(full_text,
                             truncation=True,
                             max_length=max_len,
                             padding=False,
                             return_tensors=None,
                             add_special_tokens=False)

    except Exception:
        return None

    full_ids = full_enc['input_ids']
    attn_mask = full_enc.get('attention_mask', [1] * len(full_ids))
    labels = [-100] * len(full_ids)
    N = len(full_ids)

    for k in _asst_indices(messages):
        try:
            prefix_text = tokenizer.apply_chat_template(
                messages[:k], tokenize=False, add_generation_prompt=True)
            prefix_enc = tokenizer(prefix_text,
                                   truncation=False,
                                   padding=False,
                                   return_tensors=None,
                                   add_special_tokens=False)
            prefix_len = len(prefix_enc['input_ids'])
            end_text = tokenizer.apply_chat_template(
                messages[:k + 1], tokenize=False, add_generation_prompt=False)
            end_enc = tokenizer(end_text,
                                truncation=False,
                                padding=False,
                                return_tensors=None,
                                add_special_tokens=False)
            end_len = len(end_enc['input_ids'])
        except Exception:
            continue
        for pos in range(prefix_len, min(end_len, N)):
            labels[pos] = full_ids[pos]

    if all((l == -100 for l in labels)):
        return None

    return (full_ids, attn_mask, labels)


def _make_tensor_record(input_ids, attention_mask, labels):
    return {
        'input_ids': torch.tensor(input_ids, dtype=torch.long),
        'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
        'labels': torch.tensor(labels, dtype=torch.long)
    }


_WORKER_TOKENIZER = None
_WORKER_MAX_LEN = None


def _init_preprocess_worker(model_name: str, trust_remote_code: bool,
                            max_len: int) -> None:
    global _WORKER_TOKENIZER, _WORKER_MAX_LEN
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=trust_remote_code)

    _ensure_visit_eos_token(tokenizer)
    _WORKER_TOKENIZER = tokenizer
    _WORKER_MAX_LEN = max_len


def _expand_step_messages(messages, tokenizer, max_len):
    positions = _asst_indices(messages)

    if not positions:
        return {
            'records': [],
            'n_eos': 0,
            'skipped_steps': 0,
            'skipped_traj': 1
        }

    records = []
    n_eos = skipped_steps = 0
    got_any = False

    for k in positions:
        rec = _build_step_sample(messages, k, tokenizer, max_len)
        if rec is None:
            skipped_steps += 1
            continue
        records.append({
            'input_ids': rec['input_ids'].tolist(),
            'attention_mask': rec['attention_mask'].tolist(),
            'labels': rec['labels'].tolist()
        })
        got_any = True
        if VISIT_EOS_TOKEN in messages[k]['content']:
            n_eos += 1

    return {
        'records': records,
        'n_eos': n_eos,
        'skipped_steps': skipped_steps,
        'skipped_traj': 0 if got_any else 1
    }


def _expand_step_messages_worker(messages):
    return _expand_step_messages(messages, _WORKER_TOKENIZER, _WORKER_MAX_LEN)


def _expand_trajectory_messages(messages, tokenizer, max_len):
    result = _build_trajectory_labels(messages, tokenizer, max_len)

    if result is None:
        return {'record': None, 'skipped': 1}

    full_ids, attn_mask, labels = result

    return {
        'record': {
            'input_ids': full_ids,
            'attention_mask': attn_mask,
            'labels': labels
        },
        'skipped': 0
    }


def _expand_trajectory_messages_worker(messages):
    return _expand_trajectory_messages(messages, _WORKER_TOKENIZER,
                                       _WORKER_MAX_LEN)


def _parallel_preprocess(hf_dataset, *, desc, num_workers, worker_fn, local_fn,
                         tokenizer, max_len, model_name, trust_remote_code):
    if num_workers <= 1:
        for ex in tqdm(hf_dataset, desc=desc, leave=False):
            yield local_fn(ex['messages'], tokenizer, max_len)
        return

    start_method = 'fork' if 'fork' in mp.get_all_start_methods() else 'spawn'
    ctx = mp.get_context(start_method)

    with ctx.Pool(processes=num_workers,
                  initializer=_init_preprocess_worker,
                  initargs=(model_name, trust_remote_code, max_len)) as pool:
        yield from tqdm(pool.imap(worker_fn,
                                  (ex['messages'] for ex in hf_dataset),
                                  chunksize=MP_CHUNKSIZE),
                        total=len(hf_dataset),
                        desc=desc,
                        leave=False)


class PreprocessedTensorDataset(TorchDataset):
    def __init__(self, records, *, stats=None):
        self.records = records
        self.stats = stats or {}
        self.n_eos = int(self.stats.get('n_eos', 0))

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        return self.records[idx]


def _build_dataset(hf_dataset, *, mode, tokenizer, max_len, model_name,
                   trust_remote_code, split_name, backbone_key,
                   preprocess_workers, cache_dir, use_cache,
                   refresh_cache) -> PreprocessedTensorDataset:
    cache_path = None

    if cache_dir is not None:
        suffix = _cache_suffix_for_dataset(hf_dataset,
                                           backbone_key=backbone_key,
                                           mode=mode,
                                           max_len=max_len)
        cache_path = cache_dir / f'{split_name}-{suffix}.pt'

    if use_cache and cache_path is not None and cache_path.exists() and (
            not refresh_cache):
        print(
            f'Loading cached {split_name} {mode} dataset from {cache_path} …')
        payload = _load_cache(cache_path)
        print(
            f"  {split_name}: {len(payload['records']):,} samples (cache hit)")
        return PreprocessedTensorDataset(payload['records'], stats=payload)

    if mode == 'step':
        worker_fn = _expand_step_messages_worker
        local_fn = _expand_step_messages

    else:
        worker_fn = _expand_trajectory_messages_worker
        local_fn = _expand_trajectory_messages

    print(
        f'Building {mode} dataset [{split_name}] (workers={preprocess_workers}) …'
    )
    records = []
    n_eos = skipped_steps = skipped_traj = skipped = 0

    for result in _parallel_preprocess(hf_dataset,
                                       desc=f'[{split_name}]',
                                       num_workers=preprocess_workers,
                                       worker_fn=worker_fn,
                                       local_fn=local_fn,
                                       tokenizer=tokenizer,
                                       max_len=max_len,
                                       model_name=model_name,
                                       trust_remote_code=trust_remote_code):
        if mode == 'step':
            for rec in result['records']:
                records.append(
                    _make_tensor_record(rec['input_ids'],
                                        rec['attention_mask'], rec['labels']))
            n_eos += int(result['n_eos'])
            skipped_steps += int(result['skipped_steps'])
            skipped_traj += int(result['skipped_traj'])
        else:
            rec = result.get('record')
            if rec is None:
                skipped += int(result['skipped'])
                continue
            records.append(
                _make_tensor_record(rec['input_ids'], rec['attention_mask'],
                                    rec['labels']))

    payload = {
        'records': records,
        'n_eos': n_eos,
        'split_name': split_name,
        'backbone': backbone_key,
        'max_len': max_len
    }
    print(f'  {split_name}: {len(records):,} samples')

    if use_cache and cache_path is not None:
        _save_cache_atomic(cache_path, payload)

    return PreprocessedTensorDataset(records, stats=payload)


def make_collate_fn(pad_id: int):
    def collate_fn(batch):
        max_len = max((r['input_ids'].size(0) for r in batch))
        B = len(batch)
        input_ids = torch.full((B, max_len), pad_id, dtype=torch.long)
        attention_mask = torch.zeros(B, max_len, dtype=torch.long)
        labels = torch.full((B, max_len), -100, dtype=torch.long)

        for i, r in enumerate(batch):
            L = r['input_ids'].size(0)
            input_ids[i, :L] = r['input_ids']
            attention_mask[i, :L] = r['attention_mask']
            labels[i, :L] = r['labels']

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

    return collate_fn


def run_epoch(model,
              loader,
              optimizer,
              scheduler,
              device,
              train=True,
              grad_accum=1) -> dict:
    model.train(train)
    total_loss = total_tokens = accum_step = 0

    if train:
        optimizer.zero_grad()

    for batch in tqdm(loader,
                      desc='train' if train else 'eval ',
                      leave=False,
                      dynamic_ncols=True):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        with torch.set_grad_enabled(train):
            out = model(input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels)
            loss = out.loss
        n_tokens = (labels != -100).sum().item()
        if n_tokens == 0:
            continue
        total_loss += loss.item() * n_tokens
        total_tokens += n_tokens
        if train:
            (loss / grad_accum).backward()
            accum_step += 1
            if accum_step >= grad_accum:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                accum_step = 0

    if train and accum_step > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    return {
        'loss': total_loss / max(total_tokens, 1),
        'n_tokens': total_tokens
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description='Stage 1: LoRA base generator training for SynEHR.')

    parser.add_argument('--backbone',
                        choices=list(BACKBONE_SPECS),
                        default='llama31')

    parser.add_argument(
        '--dataset-dir',
        type=Path,
        required=True,
        help='Path to HuggingFace trajectory dataset (train/val/test splits)')

    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--mode',
                        choices=['step', 'trajectory'],
                        default='step')

    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--bs', type=int, default=2)
    parser.add_argument('--grad-accum', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--warmup-ratio', type=float, default=0.05)
    parser.add_argument('--max-len', type=int, default=MAX_LEN)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--preprocess-workers',
                        type=int,
                        default=_default_preprocess_workers())

    parser.add_argument('--cache-dir', type=Path, default=None)
    parser.add_argument('--no-preprocess-cache', action='store_true')
    parser.add_argument('--refresh-preprocess-cache', action='store_true')
    parser.add_argument('--hf-token',
                        default=None,
                        help='HuggingFace token for gated models (e.g. LLaMA)')

    args = parser.parse_args()

    if args.hf_token:
        hf_login(token=args.hf_token)

    set_seed(args.seed)
    spec = BACKBONE_SPECS[args.backbone]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(
        f'Device: {device}  |  backbone: {spec.backbone_key}  |  mode: {args.mode}'
    )
    output_dir = Path(args.output_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(
        spec.model_name, trust_remote_code=spec.trust_remote_code)

    _ensure_visit_eos_token(tokenizer)
    print(f'Loading trajectory dataset from {args.dataset_dir} …')
    ds = load_from_disk(str(args.dataset_dir))
    selection_split = 'val'
    final_eval_split = 'test'
    print(
        f"  train={len(ds['train']):,}  val={len(ds[selection_split]):,}  test={len(ds[final_eval_split]):,}"
    )
    cache_dir = args.cache_dir or output_dir / 'preprocessed_cache'
    use_cache = not args.no_preprocess_cache

    if use_cache:
        cache_dir.mkdir(parents=True, exist_ok=True)

    build_kw = dict(mode=args.mode,
                    tokenizer=tokenizer,
                    max_len=args.max_len,
                    model_name=spec.model_name,
                    trust_remote_code=spec.trust_remote_code,
                    backbone_key=spec.backbone_key,
                    preprocess_workers=args.preprocess_workers,
                    cache_dir=cache_dir if use_cache else None,
                    use_cache=use_cache,
                    refresh_cache=args.refresh_preprocess_cache)

    train_ds = _build_dataset(ds['train'], split_name='train', **build_kw)
    eval_ds = _build_dataset(ds[selection_split],
                             split_name=selection_split,
                             **build_kw)

    if len(train_ds) == 0 or len(eval_ds) == 0:
        print('ERROR: empty dataset after preprocessing.')
        sys.exit(1)

    _collate = make_collate_fn(tokenizer.pad_token_id)
    train_loader = DataLoader(train_ds,
                              batch_size=args.bs,
                              shuffle=True,
                              collate_fn=_collate,
                              num_workers=args.num_workers)

    eval_loader = DataLoader(eval_ds,
                             batch_size=args.bs,
                             shuffle=False,
                             collate_fn=_collate,
                             num_workers=args.num_workers)

    model = AutoModelForCausalLM.from_pretrained(
        spec.model_name,
        device_map='auto',
        torch_dtype=torch.bfloat16
        if torch.cuda.is_available() else torch.float32,
        trust_remote_code=spec.trust_remote_code)

    base_vocab_size = model.get_input_embeddings().num_embeddings
    model.resize_token_embeddings(len(tokenizer))
    tokenizer.save_pretrained(str(output_dir))
    lora_config = LoraConfig(r=64,
                             lora_alpha=128,
                             lora_dropout=0.05,
                             bias='none',
                             task_type='CAUSAL_LM',
                             target_modules=spec.lora_target_modules)

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=args.lr, weight_decay=0.01)
    total_steps = len(train_loader) // args.grad_accum * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps,
                                                total_steps)

    log_rows = []
    best_val_loss = float('inf')

    for epoch in range(1, args.epochs + 1):
        print(f"\n{'=' * 60}\nEpoch {epoch}/{args.epochs}")
        train_stats = run_epoch(model,
                                train_loader,
                                optimizer,
                                scheduler,
                                device,
                                train=True,
                                grad_accum=args.grad_accum)
        eval_stats = run_epoch(model,
                               eval_loader,
                               optimizer,
                               scheduler,
                               device,
                               train=False)
        print(
            f"  train_loss={train_stats['loss']:.4f}  val_loss={eval_stats['loss']:.4f}"
        )
        log_rows.append({
            'epoch': epoch,
            'train_loss': train_stats['loss'],
            'val_loss': eval_stats['loss']
        })
        ckpt_dir = output_dir / f'checkpoint-epoch{epoch}'
        _save_adapter_checkpoint(model, tokenizer, ckpt_dir, base_vocab_size)
        if eval_stats['loss'] < best_val_loss:
            best_val_loss = eval_stats['loss']
            _replace_with_symlink(output_dir / 'best', ckpt_dir)
            print(f'  ★ New best val_loss={best_val_loss:.4f}')

    with (output_dir / 'training_log.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['epoch', 'train_loss', 'val_loss'])
        w.writeheader()
        w.writerows(log_rows)

    meta = {
        'backbone': spec.backbone_key,
        'model_name': spec.model_name,
        'hidden_size': spec.hidden_size,
        'dataset_mode': args.mode,
        'selection_split': selection_split,
        'best_val_loss': best_val_loss,
        'epochs': args.epochs,
        'bs': args.bs,
        'grad_accum': args.grad_accum,
        'lr': args.lr,
        'max_len': args.max_len,
        'lora_r': 64,
        'lora_alpha': 128
    }

    with (output_dir / 'run_metadata.json').open('w') as f:
        json.dump(meta, f, indent=2)

    best_dir = output_dir / 'best'

    if best_dir.exists():
        with (best_dir / 'run_metadata.json').open('w') as f:
            json.dump(meta, f, indent=2)

    print(f'\nDone. Best val_loss = {best_val_loss:.4f}')


if __name__ == '__main__':
    main()
