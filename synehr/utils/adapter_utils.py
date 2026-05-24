from __future__ import annotations
from pathlib import Path
from typing import Sequence
import torch


def load_stage1_embeddings(ckpt_path: Path | str,
                           device) -> dict[str, torch.Tensor]:
    sd = torch.load(Path(ckpt_path), map_location='cpu', weights_only=False)
    sd = sd['model_state_dict']
    keys = [
        'E_dx.weight', 'E_proc.weight', 'E_med.weight', 'E_lab.weight',
        'E_vtype.weight'
    ]
    missing = [k for k in keys if k not in sd]

    if missing:
        raise KeyError(
            f'Stage-1 checkpoint missing expected embedding keys: {missing}')

    return {k: sd[k].to(device) for k in keys}


load_phase_a_embeddings = load_stage1_embeddings


def build_code_tensors(
    history_visits: Sequence[dict], device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
           torch.Tensor, torch.Tensor]:
    V = len(history_visits)

    if V == 0:
        raise ValueError('history_visits is empty — cannot build code tensors')

    Ldx = max((len(v['dx']) for v in history_visits))
    Lproc = max((len(v['proc']) for v in history_visits))
    Lmed = max((len(v['med']) for v in history_visits))
    Llab = max((len(v['lab']) for v in history_visits))
    dx_ids = torch.zeros(1, V, Ldx, dtype=torch.long, device=device)
    proc_ids = torch.zeros(1, V, Lproc, dtype=torch.long, device=device)
    med_ids = torch.zeros(1, V, Lmed, dtype=torch.long, device=device)
    lab_ids = torch.zeros(1, V, Llab, dtype=torch.long, device=device)
    vtype_ids = torch.zeros(1, V, dtype=torch.long, device=device)
    visit_mask = torch.ones(1, V, dtype=torch.bool, device=device)

    for t, vis in enumerate(history_visits):
        dx = vis['dx']
        dx_ids[0, t, :len(dx)] = torch.tensor(dx, dtype=torch.long)
        proc = vis['proc']
        proc_ids[0, t, :len(proc)] = torch.tensor(proc, dtype=torch.long)
        med = vis['med']
        med_ids[0, t, :len(med)] = torch.tensor(med, dtype=torch.long)
        lab = vis['lab']
        lab_ids[0, t, :len(lab)] = torch.tensor(lab, dtype=torch.long)
        vtype_ids[0, t] = vis['vtype']

    return (dx_ids, proc_ids, med_ids, lab_ids, vtype_ids, visit_mask)
