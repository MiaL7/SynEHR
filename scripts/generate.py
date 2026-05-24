from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import torch
from huggingface_hub import login as hf_login
from datasets import load_from_disk
from tqdm.auto import tqdm
from synehr.models.backbone import BACKBONE_SPECS, load_frozen_backbone, register_norm_hook, get_h_last, load_frozen_time_head, load_adapter_from_meta
from synehr.models import N_PREFIX
from synehr.data.serialization import VISIT_EOS_TOKEN, append_visit_to_history, count_visits_in_user_message, extract_history_payload
from synehr.data.code_encoding import encode_patient_demographics, encode_visit_for_inference

MAX_NEW_TOKENS = 2048


def _encode_history_to_tensors(patient: dict, visits: list[dict],
                               code_vocabs: dict, device) -> dict:
    encoded = [encode_visit_for_inference(v, code_vocabs) for v in visits]
    V = len(encoded)

    def _pad_field(field: str) -> torch.Tensor:
        seqs = [e[field] for e in encoded]
        max_len = max((len(s) for s in seqs))
        t = torch.zeros(1, V, max_len, dtype=torch.long, device=device)

        for i, seq in enumerate(seqs):
            t[0, i, :len(seq)] = torch.tensor(seq,
                                              dtype=torch.long,
                                              device=device)

        return t

    return dict(
        dx_ids=_pad_field('dx'),
        proc_ids=_pad_field('proc'),
        med_ids=_pad_field('med'),
        lab_ids=_pad_field('lab'),
        vtype_ids=torch.tensor([[e['vtype'] for e in encoded]],
                               dtype=torch.long,
                               device=device),
        gap_days=torch.tensor([[float(e['gap_prev_days']) for e in encoded]],
                              dtype=torch.float32,
                              device=device),
        gap_missing=torch.tensor([[float(e['gap_missing']) for e in encoded]],
                                 dtype=torch.float32,
                                 device=device),
        visit_mask=torch.ones(1, V, dtype=torch.bool, device=device),
        demo_feats=torch.tensor([encode_patient_demographics(patient)],
                                dtype=torch.float32,
                                device=device))


def _resolve_objective_mode(ckpt_dir, cli_mode: str) -> tuple[str, str]:
    meta_path = Path(ckpt_dir) / 'run_metadata.json'

    if meta_path.exists():
        try:
            with meta_path.open() as f:
                meta = json.load(f)
            obj = meta.get('training_objective', '')
            if 'step' in obj:
                return ('step', 'metadata')
            if 'trajectory' in obj:
                return ('trajectory', 'metadata')
        except Exception:
            pass

    if cli_mode != 'auto':
        return (cli_mode, 'cli')

    return ('step', 'default')


def count_visits_in_prompt(example) -> int:
    try:
        return count_visits_in_user_message(example['messages'][1]['content'])

    except Exception:
        return 0


def update_history_content(user_content: str, new_visit_str: str):
    try:
        clean = new_visit_str.replace('```json', '').replace('```', '').strip()
        new_v = json.loads(clean)
        return append_visit_to_history(user_content, new_v)

    except Exception as e:
        return (None, str(e))


def generate_ar(model,
                tokenizer,
                time_head,
                adapter,
                eval_dataset,
                output_path: Path,
                h_cache: dict,
                spec,
                device,
                objective_mode: str = 'step',
                emit_debug: bool = False,
                max_new_tokens: int = MAX_NEW_TOKENS,
                n_prefix: int = N_PREFIX,
                max_visits: int | None = None,
                do_sample: bool = True,
                temperature: float = 0.7,
                top_p: float = 0.9,
                rep_penalty: float = 1.2,
                n_eval: int = None,
                code_vocabs: dict | None = None,
                source_split_name: str = 'test'):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    use_adapter = adapter is not None and time_head is not None
    emb_layer = None

    if use_adapter:
        emb_layer = spec.get_embed_layer(model)

    visit_eos_id = None

    try:
        eid = tokenizer.convert_tokens_to_ids(VISIT_EOS_TOKEN)
        if eid != tokenizer.unk_token_id:
            visit_eos_id = eid

    except Exception:
        pass

    if visit_eos_id is None:
        print(
            f'  WARNING: could not resolve {VISIT_EOS_TOKEN} — will only stop on eos_token_id'
        )

    eos_ids = [tokenizer.eos_token_id, visit_eos_id
               ] if visit_eos_id else [tokenizer.eos_token_id]

    gen_kwargs = dict(max_new_tokens=max_new_tokens,
                      do_sample=do_sample,
                      use_cache=True,
                      eos_token_id=eos_ids,
                      pad_token_id=tokenizer.pad_token_id,
                      repetition_penalty=rep_penalty)

    if do_sample:
        gen_kwargs.update(temperature=temperature, top_p=top_p)

    eval_indices = list(range(len(eval_dataset)))

    if n_eval is not None:
        eval_indices = eval_indices[:n_eval]
        print(
            f'  Smoke-test mode: {len(eval_indices)} patients (of {len(eval_dataset):,} total)'
        )

    else:
        print(f'  {len(eval_indices):,} patients')

    json_errors = 0

    with open(output_path, 'w', encoding='utf-8') as f_out:
        for ar_idx, eval_idx in enumerate(
                tqdm(eval_indices, desc='AR rollout', dynamic_ncols=True)):
            example = eval_dataset[eval_idx]
            current_messages = copy.deepcopy(example['messages'][:2])
            generated_visits: list[str] = []
            stop_reason = 'visit_eos'
            step_prompt_tokens: list[int] = []
            step_generated_tokens: list[int] = []
            json_parse_error: str | None = None
            hit_visit_eos = False
            if emit_debug:
                try:
                    init_text = tokenizer.apply_chat_template(
                        current_messages,
                        tokenize=False,
                        add_generation_prompt=True)
                    initial_prompt_tokens = len(
                        tokenizer(init_text,
                                  return_tensors=None,
                                  add_special_tokens=False)['input_ids'])
                except Exception:
                    initial_prompt_tokens = -1
            step_idx = 0
            while True:
                if max_visits is not None and step_idx >= max_visits:
                    stop_reason = 'max_visits'
                    break
                step_idx += 1
                text = tokenizer.apply_chat_template(
                    current_messages,
                    tokenize=False,
                    add_generation_prompt=True)
                enc = tokenizer([text],
                                return_tensors='pt',
                                add_special_tokens=False).to(device)
                input_ids = enc['input_ids']
                attention_mask = enc['attention_mask']
                if emit_debug:
                    step_prompt_tokens.append(input_ids.shape[1])
                if use_adapter:
                    with torch.no_grad():
                        model(input_ids=input_ids,
                              attention_mask=attention_mask)
                    h_last = get_h_last(h_cache, attention_mask, spec)
                    _payload = extract_history_payload(
                        current_messages[1]['content'])
                    _patient = _payload.get('patient', {}) if _payload else {}
                    _visits = _payload.get('visits', []) if _payload else []
                    if not _visits:
                        _visits = [{}]
                    _code_fields = _encode_history_to_tensors(
                        _patient, _visits, code_vocabs, device)
                    with torch.no_grad():
                        t_out = time_head(**_code_fields)
                        z_t = t_out['z_t']
                        q3 = t_out['q3']
                        conf_t = t_out['conf']
                    with torch.no_grad():
                        P_emb = adapter(h_last,
                                        z_t,
                                        conf_t,
                                        q3_input=q3,
                                        **_code_fields)
                    input_embeds = emb_layer(input_ids)
                    ext_emb = torch.cat(
                        [P_emb.to(input_embeds.dtype), input_embeds], dim=1)
                    ext_mask = torch.cat([
                        torch.ones(1,
                                   n_prefix,
                                   dtype=attention_mask.dtype,
                                   device=device), attention_mask
                    ],
                        dim=1)
                    with torch.no_grad():
                        gen_ids = model.generate(inputs_embeds=ext_emb,
                                                 attention_mask=ext_mask,
                                                 **gen_kwargs)
                else:
                    with torch.no_grad():
                        gen_ids = model.generate(input_ids=input_ids,
                                                 attention_mask=attention_mask,
                                                 **gen_kwargs)
                        gen_ids = gen_ids[:, input_ids.shape[1]:]
                if emit_debug:
                    step_generated_tokens.append(gen_ids.shape[1])
                decoded = tokenizer.decode(gen_ids[0],
                                           skip_special_tokens=False)
                clean = decoded.replace(tokenizer.eos_token or '', '').strip()
                if VISIT_EOS_TOKEN in clean:
                    clean = clean.replace(VISIT_EOS_TOKEN, '').strip()
                    if clean:
                        generated_visits.append(clean)
                    stop_reason = 'visit_eos'
                    hit_visit_eos = True
                    break
                updated, err = update_history_content(
                    current_messages[1]['content'], clean)
                if updated is None:
                    stop_reason = 'json_error'
                    json_parse_error = err
                    json_errors += 1
                    break
                try:
                    raw = clean.replace('```json', '').replace('```',
                                                               '').strip()
                    generated_visits.append(
                        json.dumps(json.loads(raw), separators=(',', ':')))
                except json.JSONDecodeError as e:
                    stop_reason = 'json_error'
                    json_parse_error = str(e)
                    json_errors += 1
                    break
                current_messages[1]['content'] = updated
            record = {
                'ar_index': ar_idx,
                'source_split': source_split_name,
                'original_source_idx': eval_idx,
                'original_test_idx': eval_idx,
                'generated_count': len(generated_visits),
                'stop_reason': stop_reason,
                'generated_visits_json_raw': generated_visits
            }
            if emit_debug:
                record['objective_mode'] = objective_mode
                record['initial_prompt_tokens'] = initial_prompt_tokens
                record['step_prompt_tokens'] = step_prompt_tokens
                record['step_generated_tokens'] = step_generated_tokens
                record['hit_visit_eos'] = hit_visit_eos
                record['json_parse_error'] = json_parse_error
            f_out.write(json.dumps(record, ensure_ascii=False) + '\n')
            f_out.flush()

    print(f'  Done: {len(eval_indices)} patients → {output_path}')

    if json_errors:
        print(f'  WARNING: {json_errors} JSON errors during rollout')


def main():
    parser = argparse.ArgumentParser(
        description='SynEHR inference: autoregressive EHR trajectory rollout.')

    parser.add_argument('--backbone',
                        choices=list(BACKBONE_SPECS),
                        default='llama31')

    parser.add_argument('--ckpt',
                        type=Path,
                        required=True,
                        help='Stage 1 LoRA checkpoint directory')

    parser.add_argument(
        '--dataset-dir',
        type=Path,
        required=True,
        help='Arrow dataset directory produced by train_stage1_base.py')

    parser.add_argument('--no-adapter',
                        action='store_true',
                        help='Baseline rollout without TSCM/TRAM')

    parser.add_argument('--adapter-ckpt',
                        default=None,
                        help='Stage 3 TRAM checkpoint (.pt)')

    parser.add_argument('--adapter-meta',
                        default=None,
                        help='run_metadata.json for the TRAM checkpoint')

    parser.add_argument('--tscm-ckpt',
                        default=None,
                        help='Stage 2 TSCM checkpoint (.pt)')

    parser.add_argument(
        '--stage1-data-dir',
        type=Path,
        default=None,
        help='Dir containing code_vocabs.json (required with adapter)')

    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--rollout-split',
                        choices=['train', 'val', 'test'],
                        default='test',
                        help='Dataset split to roll out from.')

    parser.add_argument('--objective-mode',
                        choices=['auto', 'step', 'trajectory'],
                        default='auto')

    parser.add_argument('--emit-debug-fields', action='store_true')
    parser.add_argument('--max-new-tokens', type=int, default=MAX_NEW_TOKENS)
    parser.add_argument('--max-visits', type=int, default=None)
    parser.add_argument('--do-sample', action='store_true')
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--top-p', type=float, default=0.9)
    parser.add_argument('--rep-penalty', type=float, default=1.2)
    parser.add_argument('--hf-token',
                        default=None,
                        help='HuggingFace token for gated models (e.g. LLaMA)')

    parser.add_argument('--n-eval',
                        type=int,
                        default=None,
                        help='Limit to first N V1 patients (smoke test)')

    args = parser.parse_args()

    if args.hf_token:
        hf_login(token=args.hf_token)

    spec = BACKBONE_SPECS[args.backbone]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}  |  backbone: {spec.backbone_key}')
    objective_mode, obj_source = _resolve_objective_mode(
        args.ckpt, args.objective_mode)

    print(f'Objective mode: {objective_mode}  (source: {obj_source})')

    if obj_source == 'metadata' and args.objective_mode not in (
            'auto', objective_mode):
        print(
            f'  WARNING: CLI --objective-mode={args.objective_mode} differs from checkpoint metadata ({objective_mode}). Using metadata value.'
        )

    print(f'Loading backbone from {args.ckpt} …')
    model, tokenizer = load_frozen_backbone(spec, args.ckpt, device)
    model.eval()
    model.config.use_cache = True
    h_cache, hook_handle = register_norm_hook(model, spec)

    try:
        eid = tokenizer.convert_tokens_to_ids(VISIT_EOS_TOKEN)
        if eid == tokenizer.unk_token_id:
            print(f'  WARNING: {VISIT_EOS_TOKEN} resolves to unk_token_id.')
        else:
            print(f'  {VISIT_EOS_TOKEN} token id: {eid}')

    except Exception as e:
        print(f'  WARNING: could not resolve {VISIT_EOS_TOKEN}: {e}')

    time_head = None
    adapter = None

    if not args.no_adapter:
        if args.adapter_ckpt is None:
            parser.error('--adapter-ckpt required unless --no-adapter')
        if args.tscm_ckpt is None:
            parser.error('--tscm-ckpt required unless --no-adapter')
        if args.stage1_data_dir is None:
            parser.error('--stage1-data-dir required unless --no-adapter')
        th_path = Path(args.tscm_ckpt)
        if not th_path.exists():
            raise FileNotFoundError(f'TSCM checkpoint not found: {th_path}')
        print(f'Loading TSCM from {th_path} …')
        time_head = load_frozen_time_head(th_path, spec.hidden_size, device)
        adapter_ckpt = Path(args.adapter_ckpt)
        meta_path = Path(
            args.adapter_meta
        ) if args.adapter_meta else adapter_ckpt.parent / 'run_metadata.json'
        print(f'Loading TRAM adapter from {adapter_ckpt} …')
        adapter = load_adapter_from_meta(adapter_ckpt,
                                         meta_path,
                                         device,
                                         spec=spec)

    else:
        print('No-adapter mode (baseline).')

    code_vocabs = None

    if not args.no_adapter:
        vocabs_path = Path(args.stage1_data_dir) / 'code_vocabs.json'
        if not vocabs_path.exists():
            raise FileNotFoundError(
                f'code_vocabs.json not found: {vocabs_path}')
        with vocabs_path.open() as _f:
            code_vocabs = json.load(_f)
        print(f'Loaded code_vocabs from {vocabs_path}')

    print(f'Loading dataset from {args.dataset_dir} …')
    ds = load_from_disk(str(args.dataset_dir))
    rollout_split_name = str(args.rollout_split)
    eval_ds = ds[rollout_split_name]
    print(f'  {rollout_split_name} split: {len(eval_ds):,} examples')
    generate_ar(model=model,
                tokenizer=tokenizer,
                time_head=time_head,
                adapter=adapter,
                eval_dataset=eval_ds,
                output_path=args.output,
                h_cache=h_cache,
                spec=spec,
                device=device,
                objective_mode=objective_mode,
                emit_debug=args.emit_debug_fields,
                max_new_tokens=args.max_new_tokens,
                max_visits=args.max_visits,
                do_sample=args.do_sample,
                temperature=args.temperature,
                top_p=args.top_p,
                rep_penalty=args.rep_penalty,
                n_eval=args.n_eval,
                code_vocabs=code_vocabs,
                source_split_name=rollout_split_name)

    hook_handle.remove()
    print('Done.')


if __name__ == '__main__':
    main()
