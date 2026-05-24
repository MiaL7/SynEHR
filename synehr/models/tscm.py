from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from synehr.data.time_bins import FINE_BINS, aggregate_fine_to_q3, compute_conf_from_q3

TEMPORAL_FEAT_NAMES = [
    'age', 'age_sq', 'sex_m', 'sex_f', 'sex_o', 'race_white', 'race_black',
    'race_asian', 'race_hispanic', 'race_other'
]
TEMPORAL_FEAT_SIZE = len(TEMPORAL_FEAT_NAMES)
TIME_EMB_DIM = 64
FIELD_EMB_DIM = 128
VTYPE_DIM = 32
GAP_DIM = 32
VISIT_DIM = 256
TRAJ_DIM = 256
DEMO_DIM = 64
LEN_DIM = 32
PREFIX_DIM = 128
STATE_DIM = 256


class TimeHead(nn.Module):
    def __init__(self,
                 stage1_emb: dict[str, torch.Tensor],
                 demo_feat_size: int = TEMPORAL_FEAT_SIZE,
                 time_emb_dim: int = TIME_EMB_DIM,
                 gap_dim: int = GAP_DIM,
                 visit_dim: int = VISIT_DIM,
                 traj_dim: int = TRAJ_DIM,
                 demo_dim: int = DEMO_DIM,
                 len_dim: int = LEN_DIM,
                 prefix_dim: int = PREFIX_DIM,
                 state_dim: int = STATE_DIM,
                 fine_bins: int = FINE_BINS) -> None:
        super().__init__()
        self.demo_feat_size = demo_feat_size
        self.time_emb_dim = time_emb_dim
        self.fine_bins = fine_bins
        self.temporal_feat_size = demo_feat_size
        self.register_buffer('E_dx',
                             stage1_emb['E_dx.weight'].to(torch.float32))

        self.register_buffer('E_proc',
                             stage1_emb['E_proc.weight'].to(torch.float32))

        self.register_buffer('E_med',
                             stage1_emb['E_med.weight'].to(torch.float32))

        self.register_buffer('E_lab',
                             stage1_emb['E_lab.weight'].to(torch.float32))

        self.register_buffer('E_vtype',
                             stage1_emb['E_vtype.weight'].to(torch.float32))

        self.register_buffer('demo_mean',
                             torch.zeros(demo_feat_size, dtype=torch.float32))

        self.register_buffer('demo_std',
                             torch.ones(demo_feat_size, dtype=torch.float32))

        visit_input_dim = FIELD_EMB_DIM * 4 + VTYPE_DIM + gap_dim
        self.gap_encoder = nn.Sequential(nn.Linear(2, gap_dim), nn.GELU(),
                                         nn.Linear(gap_dim, gap_dim))

        self.visit_encoder = nn.Sequential(
            nn.Linear(visit_input_dim, visit_dim), nn.GELU(),
            nn.LayerNorm(visit_dim))

        self.traj_encoder = nn.GRU(visit_dim, traj_dim, batch_first=True)
        self.demo_encoder = nn.Sequential(nn.Linear(demo_feat_size, demo_dim),
                                          nn.GELU(), nn.LayerNorm(demo_dim))

        self.len_encoder = nn.Sequential(nn.Linear(1, len_dim), nn.GELU())
        self.prefix_encoder = nn.Sequential(
            nn.Linear(demo_dim + len_dim + 2, prefix_dim), nn.GELU(),
            nn.LayerNorm(prefix_dim))

        self.state_encoder = nn.Sequential(
            nn.Linear(traj_dim + prefix_dim, state_dim), nn.GELU(),
            nn.LayerNorm(state_dim))

        self.temporal_head = nn.Sequential(nn.Linear(state_dim, state_dim),
                                           nn.GELU(),
                                           nn.Linear(state_dim, time_emb_dim))

        self.hazard_head = nn.Linear(state_dim, fine_bins)
        self.dist_head = nn.Linear(state_dim, 2)

    @classmethod
    def from_state_dict(cls, state: dict[str, torch.Tensor]) -> 'TimeHead':
        stage1_emb = {
            'E_dx.weight': state['E_dx'],
            'E_proc.weight': state['E_proc'],
            'E_med.weight': state['E_med'],
            'E_lab.weight': state['E_lab'],
            'E_vtype.weight': state['E_vtype']
        }

        return cls(stage1_emb=stage1_emb,
                   demo_feat_size=state['demo_mean'].numel(),
                   time_emb_dim=state['temporal_head.2.weight'].shape[0],
                   gap_dim=state['gap_encoder.0.weight'].shape[0],
                   visit_dim=state['visit_encoder.0.weight'].shape[0],
                   traj_dim=state['traj_encoder.weight_hh_l0'].shape[1],
                   demo_dim=state['demo_encoder.0.weight'].shape[0],
                   len_dim=state['len_encoder.0.weight'].shape[0],
                   prefix_dim=state['prefix_encoder.0.weight'].shape[0],
                   state_dim=state['state_encoder.0.weight'].shape[0],
                   fine_bins=state['hazard_head.weight'].shape[0])

    @staticmethod
    def _masked_mean(embedding_table: torch.Tensor,
                     ids: torch.Tensor) -> torch.Tensor:
        emb = embedding_table[ids]
        mask = (ids != 0).to(emb.dtype).unsqueeze(-1)
        denom = mask.sum(dim=2).clamp(min=1.0)

        return (emb * mask).sum(dim=2) / denom

    def set_demo_stats(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        self.demo_mean.copy_(
            mean.to(dtype=torch.float32, device=self.demo_mean.device))

        self.demo_std.copy_(
            std.to(dtype=torch.float32, device=self.demo_std.device))

    def encode_temporal_state(
            self, dx_ids: torch.Tensor, proc_ids: torch.Tensor,
            med_ids: torch.Tensor, lab_ids: torch.Tensor,
            vtype_ids: torch.Tensor, gap_days: torch.Tensor,
            gap_missing: torch.Tensor, visit_mask: torch.Tensor,
            demo_feats: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        e_dx = self._masked_mean(self.E_dx, dx_ids)
        e_proc = self._masked_mean(self.E_proc, proc_ids)
        e_med = self._masked_mean(self.E_med, med_ids)
        e_lab = self._masked_mean(self.E_lab, lab_ids)
        e_type = self.E_vtype[vtype_ids]
        gap_input = torch.stack(
            [torch.log1p(gap_days.clamp(min=0.0)), gap_missing], dim=-1)

        g = self.gap_encoder(gap_input)
        v = self.visit_encoder(
            torch.cat([e_dx, e_proc, e_med, e_lab, e_type, g], dim=-1))

        lengths = visit_mask.long().sum(dim=1).clamp(min=1)
        packed = pack_padded_sequence(v,
                                      lengths.cpu(),
                                      batch_first=True,
                                      enforce_sorted=False)

        _, h_n = self.traj_encoder(packed)
        r = h_n[-1]
        demo_norm = (demo_feats - self.demo_mean.unsqueeze(0)
                     ) / self.demo_std.unsqueeze(0).clamp(min=1e-06)

        d = self.demo_encoder(demo_norm)
        n_log = torch.log(lengths.to(v.dtype).unsqueeze(-1) + 1.0)
        len_emb = self.len_encoder(n_log)
        last_idx = lengths - 1
        batch_idx = torch.arange(v.size(0), device=v.device)
        last_gap_days = gap_days[batch_idx, last_idx].unsqueeze(-1)
        last_gap_missing = gap_missing[batch_idx, last_idx].unsqueeze(-1)
        prefix_in = torch.cat([
            d, len_emb,
            torch.log1p(last_gap_days.clamp(min=0.0)), last_gap_missing
        ],
            dim=-1)

        p = self.prefix_encoder(prefix_in)
        l = self.state_encoder(torch.cat([r, p], dim=-1))

        return (l, r)

    def forward(self, dx_ids: torch.Tensor, proc_ids: torch.Tensor,
                med_ids: torch.Tensor, lab_ids: torch.Tensor,
                vtype_ids: torch.Tensor, gap_days: torch.Tensor,
                gap_missing: torch.Tensor, visit_mask: torch.Tensor,
                demo_feats: torch.Tensor) -> dict[str, torch.Tensor]:
        latent_state, traj_state = self.encode_temporal_state(
            dx_ids=dx_ids,
            proc_ids=proc_ids,
            med_ids=med_ids,
            lab_ids=lab_ids,
            vtype_ids=vtype_ids,
            gap_days=gap_days,
            gap_missing=gap_missing,
            visit_mask=visit_mask,
            demo_feats=demo_feats)

        z_t = self.temporal_head(latent_state)
        hazard_logits = self.hazard_head(latent_state)
        hazard = torch.sigmoid(hazard_logits).clamp(min=1e-06, max=1.0 - 1e-06)
        log_survival_prefix = torch.cumsum(torch.log1p(-hazard),
                                           dim=-1) - torch.log1p(-hazard)

        log_mass = torch.log(hazard) + log_survival_prefix
        q_fine = torch.softmax(log_mass, dim=-1)
        q3 = aggregate_fine_to_q3(q_fine)
        mu, log_s = self.dist_head(latent_state).chunk(2, dim=-1)
        conf_ent = compute_conf_from_q3(q3).unsqueeze(-1)
        conf_scale = 1.0 - torch.sigmoid(log_s)
        conf = 0.5 * conf_ent + 0.5 * conf_scale

        return {
            'latent_state': latent_state,
            'traj_state': traj_state,
            'z_t': z_t,
            'hazard_logits': hazard_logits,
            'hazard': hazard,
            'log_mass': log_mass,
            'q_fine': q_fine,
            'q3': q3,
            'mu': mu,
            'log_s': log_s,
            'conf': conf.clamp(0.0, 1.0)
        }


def discrete_time_survival_nll(hazard: torch.Tensor,
                               target_bin: torch.Tensor) -> torch.Tensor:
    mask_before = torch.arange(
        hazard.size(1),
        device=hazard.device).unsqueeze(0) < target_bin.unsqueeze(1)

    mask_at = torch.arange(
        hazard.size(1),
        device=hazard.device).unsqueeze(0) == target_bin.unsqueeze(1)

    loss_before = -(torch.log1p(-hazard) * mask_before).sum(dim=-1)
    loss_at = -(torch.log(hazard) * mask_at).sum(dim=-1)

    return (loss_before + loss_at).mean()


def gaussian_nll_log_days(target_days: torch.Tensor, mu: torch.Tensor,
                          log_s: torch.Tensor) -> torch.Tensor:
    y = torch.log1p(target_days).unsqueeze(-1)
    inv_var = torch.exp(-2.0 * log_s)

    return (log_s + 0.5 * (y - mu).pow(2) * inv_var).mean()
