from __future__ import annotations
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ADDED_TOKEN_STATE_NAME = 'added_token_rows.pt'


@dataclass
class BackboneSpec:
    backbone_key: str
    model_name: str
    hidden_size: int
    num_layers: int
    hook_layers: List[int]
    lora_target_modules: List[str]
    trust_remote_code: bool = False
    disable_thinking: bool = False

    def _inner(self, model):
        return model.base_model.model if isinstance(model,
                                                    PeftModel) else model

    def get_embed_layer(self, model):
        return self._inner(model).model.embed_tokens

    def get_layer_container(self, model):
        return self._inner(model).model.layers

    def get_final_norm(self, model):
        return self._inner(model).model.norm


BACKBONE_SPECS: dict[str, BackboneSpec] = {
    'llama31':
    BackboneSpec(backbone_key='llama31',
                 model_name='meta-llama/Meta-Llama-3.1-8B-Instruct',
                 hidden_size=4096,
                 num_layers=32,
                 hook_layers=[29, 30, 31],
                 lora_target_modules=[
                     'q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj',
                     'up_proj', 'down_proj'
                 ]),
    'qwen25':
    BackboneSpec(backbone_key='qwen25',
                 model_name='Qwen/Qwen2.5-7B-Instruct',
                 hidden_size=3584,
                 num_layers=28,
                 hook_layers=[25, 26, 27],
                 lora_target_modules=[
                     'q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj',
                     'up_proj', 'down_proj'
                 ],
                 trust_remote_code=True),
    'qwen3':
    BackboneSpec(backbone_key='qwen3',
                 model_name='Qwen/Qwen3-4B-Instruct-2507',
                 hidden_size=2560,
                 num_layers=36,
                 hook_layers=[33, 34, 35],
                 lora_target_modules=[
                     'q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj',
                     'up_proj', 'down_proj'
                 ],
                 trust_remote_code=True,
                 disable_thinking=True)
}


def _restore_added_token_rows(model, lora_ckpt: Path) -> None:
    state_path = lora_ckpt / ADDED_TOKEN_STATE_NAME

    if not state_path.exists():
        return

    state = torch.load(state_path, map_location='cpu')
    token_ids = state.get('token_ids', [])

    if not token_ids:
        return

    token_index = torch.tensor(token_ids, dtype=torch.long)

    with torch.no_grad():
        input_emb = model.get_input_embeddings()
        input_rows = state.get('input_embeddings')
        if input_rows is not None:
            input_emb.weight.index_copy_(
                0, token_index.to(device=input_emb.weight.device),
                input_rows.to(device=input_emb.weight.device,
                              dtype=input_emb.weight.dtype))
        output_emb = model.get_output_embeddings()
        output_rows = state.get('output_embeddings')
        if output_emb is not None and output_rows is not None:
            output_emb.weight.index_copy_(
                0, token_index.to(device=output_emb.weight.device),
                output_rows.to(device=output_emb.weight.device,
                               dtype=output_emb.weight.dtype))

    print(f'Restored {len(token_ids)} added token row(s) from {state_path}')


def load_frozen_backbone(spec: BackboneSpec, lora_ckpt: Path, device):
    print(f'Loading tokenizer: {spec.model_name}')
    tokenizer = AutoTokenizer.from_pretrained(
        spec.model_name, trust_remote_code=spec.trust_remote_code)

    specials = tokenizer.special_tokens_map.get('additional_special_tokens',
                                                [])

    if '<VISIT_EOS>' not in specials:
        tokenizer.add_special_tokens(
            {'additional_special_tokens': specials + ['<VISIT_EOS>']})

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = 'right'
    print(f'Loading base model: {spec.model_name}')
    base = AutoModelForCausalLM.from_pretrained(
        spec.model_name,
        torch_dtype=torch.bfloat16,
        device_map='auto',
        trust_remote_code=spec.trust_remote_code)

    base.resize_token_embeddings(len(tokenizer))
    print(f'Loading LoRA checkpoint: {lora_ckpt}')
    model = PeftModel.from_pretrained(base, str(lora_ckpt))
    _restore_added_token_rows(model, lora_ckpt)
    model.eval()

    for p in model.parameters():
        p.requires_grad_(False)

    if spec.disable_thinking and hasattr(model.generation_config,
                                         'enable_thinking'):
        model.generation_config.enable_thinking = False

    n_trainable = sum(
        (p.numel() for p in model.parameters() if p.requires_grad))

    print(f'Backbone frozen. Trainable params: {n_trainable:,}')

    return (model, tokenizer)


def register_norm_hook(model, spec: BackboneSpec) -> tuple[dict, object]:
    _cache: dict = {}

    def _extract_hidden(output, key: str) -> torch.Tensor:
        hidden = output[0] if isinstance(output, (tuple, list)) else output

        if not isinstance(hidden, torch.Tensor) or hidden.ndim != 3:
            raise RuntimeError(
                f"Unexpected hidden-state shape for {key}: {type(hidden).__name__} shape={getattr(hidden, 'shape', None)}"
            )

        return hidden

    def _make_hook(key):
        def _hook(module, inp, output):
            _cache[key] = _extract_hidden(output, key)

        return _hook

    layer_container = spec.get_layer_container(model)
    handles = []

    for idx in spec.hook_layers:
        handles.append(layer_container[idx].register_forward_hook(
            _make_hook(f'h{idx}')))

    final_norm = spec.get_final_norm(model)
    handles.append(final_norm.register_forward_hook(_make_hook('h_norm')))

    class _MultiHandle:
        def remove(self):
            for h in handles:
                h.remove()

    return (_cache, _MultiHandle())


def get_h_last(h_cache: dict, attention_mask: torch.Tensor,
               spec: BackboneSpec) -> torch.Tensor:
    device = attention_mask.device
    last_pos = attention_mask.sum(dim=1) - 1
    batch_idx = torch.arange(attention_mask.size(0), device=device)
    expected = (attention_mask.size(0), attention_mask.size(1))
    keys = [f'h{i}' for i in spec.hook_layers] + ['h_norm']

    for key in keys:
        if key not in h_cache:
            raise RuntimeError(f'Missing hidden-state cache entry: {key}')
        if h_cache[key].shape[:2] != expected:
            raise RuntimeError(
                f'Cache shape mismatch for {key}: got {tuple(h_cache[key].shape)}, expected {expected}'
            )

    stacked = torch.stack([h_cache[k][batch_idx, last_pos] for k in keys],
                          dim=0)

    return stacked.mean(dim=0).detach().to(torch.float32)


def load_frozen_time_head(ckpt_path: Path, hidden_size: int, device):
    from synehr.models.tscm import TimeHead
    state = torch.load(ckpt_path, map_location=device)

    if isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']

    time_head = TimeHead.from_state_dict(state).to(device).to(torch.float32)
    time_head.load_state_dict(state)
    time_head.eval()

    for p in time_head.parameters():
        p.requires_grad_(False)

    return time_head


def load_adapter_from_meta(adapter_ckpt: Path,
                           meta_path: Path,
                           device,
                           spec: BackboneSpec | None = None):
    from synehr.models.tram import RelationAdapter, N_PREFIX, TIME_EMB_DIM
    from synehr.utils.adapter_utils import load_stage1_embeddings

    with meta_path.open() as f:
        meta = json.load(f)

    version = meta.get('adapter_version')

    if version != 'v2_struct_static':
        raise ValueError(
            f"load_adapter_from_meta only supports adapter_version='v2_struct_static', got {version!r}."
        )

    hidden_size = int(meta['hidden_size'])

    if spec is not None:
        if meta.get('backbone') != spec.backbone_key:
            raise ValueError(
                f"Checkpoint backbone {meta.get('backbone')!r} != spec {spec.backbone_key!r}"
            )
        if hidden_size != spec.hidden_size:
            raise ValueError(
                f'Checkpoint hidden_size {hidden_size} != spec {spec.hidden_size}'
            )

    stage1_ckpt = meta.get('stage1_ckpt') or meta.get('phase_a_ckpt')

    if not stage1_ckpt:
        raise ValueError("run_metadata.json missing 'stage1_ckpt' field.")

    stage1_emb = load_stage1_embeddings(Path(stage1_ckpt), device)
    rel_dim = int(meta.get('rel_dim', 128))
    n_prefix = int(meta.get('n_prefix', N_PREFIX))
    disable_static = bool(meta.get('disable_static', False))
    disable_dynamic = bool(meta.get('disable_dynamic', False))
    alpha_s_init = float(meta.get('alpha_s_init', 0.1))
    alpha_d_init = float(meta.get('alpha_d_init', 0.03))
    use_regime_confidence = bool(meta.get('use_regime_confidence', True))
    branch_local_norm = bool(meta.get('branch_local_norm', True))
    use_time_conditioning = bool(meta.get('use_time_conditioning', False))
    adapter = RelationAdapter(
        stage1_emb=stage1_emb,
        hidden_size=hidden_size,
        time_emb_dim=TIME_EMB_DIM,
        rel_dim=rel_dim,
        n_prefix=n_prefix,
        disable_static=disable_static,
        disable_dynamic=disable_dynamic,
        alpha_s_init=alpha_s_init,
        alpha_d_init=alpha_d_init,
        use_regime_confidence=use_regime_confidence,
        branch_local_norm=branch_local_norm,
        use_time_conditioning=use_time_conditioning).to(device).to(
            torch.float32)

    state = torch.load(adapter_ckpt, map_location=device)
    adapter.load_state_dict(state, strict=True)
    adapter.eval()

    for p in adapter.parameters():
        p.requires_grad_(False)

    adapter._adapter_version = version
    print(
        f'  RelationAdapter loaded: rel_dim={rel_dim}, n_prefix={n_prefix}, disable_static={disable_static}, disable_dynamic={disable_dynamic}, hidden_size={hidden_size}'
    )

    return adapter
