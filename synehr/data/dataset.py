from __future__ import annotations
import json
import random
from collections import Counter
from pathlib import Path
from typing import Optional
import torch
from torch.utils.data import Dataset
from synehr.data.serialization import VISIT_EOS_TOKEN, SYSTEM_PROMPT, fit_step_messages_by_visit, make_user_message
from synehr.data.time_bins import days_to_fine_bin, days_to_fine_label, fine_bin_to_regime3
from synehr.data.code_encoding import encode_patient_demographics, encode_visit_for_inference

VAL_RATIO = 0.1
SPLIT_SEED = 42
LOOKUP_KEY_FIELDS = ('split', 'raw_sample_id', 'occurrence_idx')


def make_lookup_key(split_name: str, raw_sample_id: str,
                    occurrence_idx: int) -> dict:
    return {
        'split': split_name,
        'raw_sample_id': raw_sample_id,
        'occurrence_idx': int(occurrence_idx),
        'key_str': f'{split_name}::{raw_sample_id}::{int(occurrence_idx)}'
    }


def make_lookup_manifest(split_name: str,
                         raw_sample_ids: list[str],
                         occurrence_indices: list[int],
                         *,
                         max_examples: int = 5) -> dict:
    examples = [
        make_lookup_key(split_name, raw_sample_id, occurrence_idx)
        for raw_sample_id, occurrence_idx in list(
            zip(raw_sample_ids, occurrence_indices))[:max_examples]
    ]

    return {
        'lookup_key_fields': list(LOOKUP_KEY_FIELDS),
        'lookup_key_definition': '(split, raw_sample_id, occurrence_idx)',
        'split': split_name,
        'n_keys': len(raw_sample_ids),
        'examples': examples
    }


def format_patient(record: dict) -> dict | None:
    patient = record['patient']
    visits = record['visits']

    if len(visits) < 2:
        return None

    visits_binned = []

    for visit in visits:
        visit_binned = dict(visit)
        delta_days = visit.get('delta_days')
        visit_binned[
            'delta_days'] = None if delta_days is None else days_to_fine_label(
                float(delta_days))
        visits_binned.append(visit_binned)

    messages = [{'role': 'system', 'content': SYSTEM_PROMPT}]
    current_visits: list[dict] = []

    for visit_index, visit in enumerate(visits_binned):
        if visit_index == 0:
            current_visits.append(visit)
            messages.append({
                'role':
                'user',
                'content':
                make_user_message(patient, list(current_visits))
            })
            continue
        messages.append({
            'role': 'assistant',
            'content': json.dumps(visit, indent=2)
        })
        current_visits.append(visit)
        if visit_index < len(visits_binned) - 1:
            messages.append({
                'role':
                'user',
                'content':
                make_user_message(patient, list(current_visits))
            })

    messages.append({'role': 'assistant', 'content': VISIT_EOS_TOKEN})

    return {'messages': messages}


def patient_id_from_record(record: dict) -> str:
    patient = record['patient']

    return f"{patient['age']}_{patient['sex']}_{patient['race']}"


def load_jsonl_records(path: Path) -> list[dict]:
    records: list[dict] = []

    with path.open('r', encoding='utf-8') as f:
        for line_idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            rec['_jsonl_line_idx'] = line_idx
            records.append(rec)

    return records


def split_train_val_records(
        records: list[dict],
        *,
        seed: int = SPLIT_SEED,
        val_ratio: float = VAL_RATIO) -> tuple[list[dict], list[dict]]:
    rng = random.Random(seed)
    indices = list(range(len(records)))
    rng.shuffle(indices)
    n_val = max(1, int(len(indices) * val_ratio))
    val_idx = set(indices[:n_val])
    train_idx = set(indices[n_val:])
    train_records = [records[i] for i in sorted(train_idx)]
    val_records = [records[i] for i in sorted(val_idx)]

    return (train_records, val_records)


def load_timecond_raw_splits(
        train_jsonl: Path | None = None,
        test_jsonl: Path | None = None) -> dict[str, list[dict]]:
    if train_jsonl is None or test_jsonl is None:
        raise ValueError(
            'load_timecond_raw_splits requires explicit train_jsonl and test_jsonl paths. Pass --train-jsonl and --test-jsonl to the training script.'
        )

    train_records_all = load_jsonl_records(Path(train_jsonl))
    test_records = load_jsonl_records(Path(test_jsonl))
    train_records, val_records = split_train_val_records(train_records_all)

    return {'train': train_records, 'val': val_records, 'test': test_records}


class StepTimeCDataset(Dataset):
    def __init__(self,
                 records: list[dict],
                 tokenizer,
                 split_name: str,
                 max_len: int,
                 code_vocabs_path: Path | str | None = None) -> None:
        self.split_name = split_name
        self.records: list[dict] = []
        sample_counter: Counter[str] = Counter()
        skipped_single_visit = 0
        skipped_fit = 0
        skipped_missing_gap = 0

        if code_vocabs_path is None:
            raise ValueError(
                'StepTimeCDataset requires code_vocabs_path pointing to code_vocabs.json. Pass --stage1-data-dir to the training script.'
            )

        with Path(code_vocabs_path).open() as _f:
            _code_vocabs = json.load(_f)

        for trajectory_index, raw_record in enumerate(records):
            if len(raw_record.get('visits', [])) < 2:
                skipped_single_visit += 1
                continue
            formatted = format_patient(raw_record)
            if formatted is None:
                skipped_single_visit += 1
                continue
            messages = formatted['messages']
            patient_id = patient_id_from_record(raw_record)
            asst_positions = [
                idx for idx, msg in enumerate(messages)
                if msg['role'] == 'assistant'
                and msg['content'] != VISIT_EOS_TOKEN
            ]
            target_visits = raw_record['visits'][1:]
            for step_idx, (asst_pos, target_visit) in enumerate(zip(
                asst_positions, target_visits),
                    start=1):
                target_days = target_visit.get('delta_days')
                if target_days is None:
                    skipped_missing_gap += 1
                    continue
                raw_sample_id = f'{patient_id}_step{step_idx}'
                occurrence_idx = sample_counter[raw_sample_id]
                sample_counter[raw_sample_id] += 1
                lookup_key = make_lookup_key(split_name, raw_sample_id,
                                             occurrence_idx)
                fitted = fit_step_messages_by_visit(messages, asst_pos,
                                                    tokenizer, max_len)
                if fitted is None:
                    skipped_fit += 1
                    continue
                _, _, hist_enc, full_enc = fitted
                hist_ids = hist_enc['input_ids']
                hist_mask = hist_enc.get('attention_mask', [1] * len(hist_ids))
                full_ids = full_enc['input_ids']
                full_mask = full_enc.get('attention_mask', [1] * len(full_ids))
                prompt_len = len(hist_ids)
                labels = [-100] * prompt_len + full_ids[prompt_len:]
                target_days_f = float(target_days)
                gt_regime3 = fine_bin_to_regime3(
                    days_to_fine_bin(target_days_f))
                hist_codes = [
                    encode_visit_for_inference(v, _code_vocabs)
                    for v in raw_record['visits'][:step_idx]
                ]
                target_fine_bin = days_to_fine_bin(target_days_f)
                demo_feats = encode_patient_demographics(raw_record['patient'])
                self.records.append({
                    'hist_ids':
                    torch.tensor(hist_ids, dtype=torch.long),
                    'hist_mask':
                    torch.tensor(hist_mask, dtype=torch.long),
                    'full_ids':
                    torch.tensor(full_ids, dtype=torch.long),
                    'full_mask':
                    torch.tensor(full_mask, dtype=torch.long),
                    'labels':
                    torch.tensor(labels, dtype=torch.long),
                    'split_name':
                    split_name,
                    'raw_sample_id':
                    raw_sample_id,
                    'sample_id':
                    raw_sample_id,
                    'patient_id':
                    patient_id,
                    'prefix_idx':
                    step_idx,
                    'target_days':
                    target_days_f,
                    'target_fine_bin':
                    target_fine_bin,
                    'gt_regime3':
                    gt_regime3,
                    'occurrence_idx':
                    occurrence_idx,
                    'lookup_key':
                    lookup_key,
                    'trajectory_index':
                    trajectory_index,
                    'demo_feats':
                    torch.tensor(demo_feats, dtype=torch.float32),
                    'hist_codes':
                    hist_codes
                })

        print(
            f'  StepTimeCDataset[{split_name}] samples={len(self.records):,} skipped_single={skipped_single_visit} skipped_fit={skipped_fit} skipped_missing_gap={skipped_missing_gap}'
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        return self.records[idx]

    def teacher_keys(self) -> list[tuple[str, str, int]]:
        return [(r['split_name'], r['raw_sample_id'], r['occurrence_idx'])
                for r in self.records]

    def lookup_manifest(self, *, max_examples: int = 5) -> dict:
        return make_lookup_manifest(
            self.split_name, [r['raw_sample_id'] for r in self.records],
            [r['occurrence_idx'] for r in self.records],
            max_examples=max_examples)


def make_timecond_collate_fn(pad_id: int):
    def collate_fn(batch: list[dict]) -> dict:
        max_hist = max((r['hist_ids'].size(0) for r in batch))
        max_full = max((r['full_ids'].size(0) for r in batch))
        batch_size = len(batch)
        hist_ids = torch.full((batch_size, max_hist), pad_id, dtype=torch.long)
        hist_mask = torch.zeros((batch_size, max_hist), dtype=torch.long)
        full_ids = torch.full((batch_size, max_full), pad_id, dtype=torch.long)
        full_mask = torch.zeros((batch_size, max_full), dtype=torch.long)
        labels = torch.full((batch_size, max_full), -100, dtype=torch.long)
        raw_sample_ids: list[str] = []
        patient_ids: list[str] = []
        split_names: list[str] = []
        prefix_idx: list[int] = []
        target_days: list[float] = []
        target_fine_bin: list[int] = []
        gt_regime3: list[int] = []
        occurrence_idx: list[int] = []
        trajectory_index: list[int] = []
        lookup_keys: list[dict] = []
        demo_feats_list = [r['demo_feats'] for r in batch]
        demo_feats = torch.stack(demo_feats_list, dim=0)
        hist_codes_list = [r['hist_codes'] for r in batch]
        V_max = max((len(hc) for hc in hist_codes_list))
        Ldx_max = max((max((len(v['dx']) for v in hc), default=1)
                       for hc in hist_codes_list),
                      default=1)

        Lproc_max = max((max((len(v['proc']) for v in hc), default=1)
                         for hc in hist_codes_list),
                        default=1)

        Lmed_max = max((max((len(v['med']) for v in hc), default=1)
                        for hc in hist_codes_list),
                       default=1)

        Llab_max = max((max((len(v['lab']) for v in hc), default=1)
                        for hc in hist_codes_list),
                       default=1)

        dx_ids_out = torch.zeros(batch_size, V_max, Ldx_max, dtype=torch.long)
        proc_ids_out = torch.zeros(batch_size,
                                   V_max,
                                   Lproc_max,
                                   dtype=torch.long)

        med_ids_out = torch.zeros(batch_size,
                                  V_max,
                                  Lmed_max,
                                  dtype=torch.long)

        lab_ids_out = torch.zeros(batch_size,
                                  V_max,
                                  Llab_max,
                                  dtype=torch.long)

        vtype_ids_out = torch.zeros(batch_size, V_max, dtype=torch.long)
        gap_days_out = torch.zeros(batch_size, V_max, dtype=torch.float32)
        gap_missing_out = torch.ones(batch_size, V_max, dtype=torch.float32)
        visit_mask_out = torch.zeros(batch_size, V_max, dtype=torch.bool)

        for i, record in enumerate(batch):
            hist_len = record['hist_ids'].size(0)
            full_len = record['full_ids'].size(0)
            hist_ids[i, :hist_len] = record['hist_ids']
            hist_mask[i, :hist_len] = record['hist_mask']
            full_ids[i, :full_len] = record['full_ids']
            full_mask[i, :full_len] = record['full_mask']
            labels[i, :full_len] = record['labels']
            raw_sample_ids.append(record['raw_sample_id'])
            patient_ids.append(record['patient_id'])
            split_names.append(record['split_name'])
            prefix_idx.append(record['prefix_idx'])
            target_days.append(record['target_days'])
            target_fine_bin.append(record['target_fine_bin'])
            gt_regime3.append(record['gt_regime3'])
            occurrence_idx.append(record['occurrence_idx'])
            trajectory_index.append(record['trajectory_index'])
            lookup_keys.append(record['lookup_key'])
            for t, vis in enumerate(hist_codes_list[i]):
                visit_mask_out[i, t] = True
                dx = vis['dx']
                dx_ids_out[i, t, :len(dx)] = torch.tensor(dx, dtype=torch.long)
                proc = vis['proc']
                proc_ids_out[i, t, :len(proc)] = torch.tensor(proc,
                                                              dtype=torch.long)
                med = vis['med']
                med_ids_out[i, t, :len(med)] = torch.tensor(med,
                                                            dtype=torch.long)
                lab = vis['lab']
                lab_ids_out[i, t, :len(lab)] = torch.tensor(lab,
                                                            dtype=torch.long)
                vtype_ids_out[i, t] = vis['vtype']
                gap_days_out[i, t] = float(vis['gap_prev_days'])
                gap_missing_out[i, t] = float(vis['gap_missing'])

        return {
            'hist_ids': hist_ids,
            'hist_mask': hist_mask,
            'full_ids': full_ids,
            'full_mask': full_mask,
            'labels': labels,
            'raw_sample_ids': raw_sample_ids,
            'sample_ids': raw_sample_ids,
            'patient_ids': patient_ids,
            'split_names': split_names,
            'prefix_idx': prefix_idx,
            'target_days': target_days,
            'target_fine_bin': target_fine_bin,
            'gt_regime3': gt_regime3,
            'occurrence_idx': occurrence_idx,
            'trajectory_index': trajectory_index,
            'lookup_keys': lookup_keys,
            'demo_feats': demo_feats,
            'dx_ids': dx_ids_out,
            'proc_ids': proc_ids_out,
            'med_ids': med_ids_out,
            'lab_ids': lab_ids_out,
            'vtype_ids': vtype_ids_out,
            'gap_days': gap_days_out,
            'gap_missing': gap_missing_out,
            'visit_mask': visit_mask_out
        }

    return collate_fn
