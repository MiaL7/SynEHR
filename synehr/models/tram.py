from __future__ import annotations
import math
import torch
import torch.nn as nn

HIDDEN_SIZE = 4096
TIME_EMB_DIM = 64
REL_DIM = 128
N_PREFIX = 1
_FIELD_EMB_DIM = 128
_VTYPE_DIM = 32


def _softplus_inverse(value: float) -> float:
    if value <= 0.0:
        raise ValueError(f'Expected a positive scale init, got {value}.')

    return math.log(math.expm1(value))


class StaticRelationAdapter(nn.Module):
    def __init__(self,
                 stage1_emb: dict[str, torch.Tensor],
                 rel_dim: int = REL_DIM) -> None:
        super().__init__()
        self.E_dx = nn.Parameter(stage1_emb['E_dx.weight'].to(
            torch.float32).clone())

        self.E_proc = nn.Parameter(stage1_emb['E_proc.weight'].to(
            torch.float32).clone())

        self.E_med = nn.Parameter(stage1_emb['E_med.weight'].to(
            torch.float32).clone())

        self.E_lab = nn.Parameter(stage1_emb['E_lab.weight'].to(
            torch.float32).clone())

        self.E_vtype = nn.Parameter(stage1_emb['E_vtype.weight'].to(
            torch.float32).clone())

        visit_in_dim = _FIELD_EMB_DIM * 4 + _VTYPE_DIM
        self.visit_mlp = nn.Sequential(nn.Linear(visit_in_dim, rel_dim),
                                       nn.GELU(), nn.LayerNorm(rel_dim))

        self.global_score = nn.Linear(rel_dim, 1)
        self.field_proj = nn.ModuleList(
            [nn.Linear(_FIELD_EMB_DIM, rel_dim) for _ in range(4)])

        self.field_score = nn.ModuleList(
            [nn.Linear(rel_dim, 1) for _ in range(4)])

        self.pair_mlp = nn.Sequential(nn.Linear(4 * rel_dim, rel_dim),
                                      nn.GELU())

        self.pair_fuse = nn.Linear(6 * rel_dim, rel_dim)
        self.fuse_mlp = nn.Sequential(nn.Linear(3 * rel_dim, rel_dim),
                                      nn.GELU(), nn.LayerNorm(rel_dim))

    @staticmethod
    def _masked_mean(embeddings: torch.Tensor,
                     ids: torch.Tensor) -> torch.Tensor:
        mask = (ids != 0).float().unsqueeze(-1)
        n = mask.sum(dim=1).clamp(min=1.0)

        return (embeddings * mask).sum(dim=1) / n

    def _field_attn_pool(self, emb_seq: torch.Tensor, visit_mask: torch.Tensor,
                         proj: nn.Linear,
                         score_head: nn.Linear) -> torch.Tensor:
        h_f = proj(emb_seq)
        scores = score_head(h_f)
        scores = scores.masked_fill(~visit_mask.unsqueeze(-1), float('-inf'))
        a = torch.softmax(scores, dim=1)

        return (a * h_f).sum(dim=1)

    def forward(self, dx_ids: torch.Tensor, proc_ids: torch.Tensor,
                med_ids: torch.Tensor, lab_ids: torch.Tensor,
                vtype_ids: torch.Tensor,
                visit_mask: torch.Tensor) -> torch.Tensor:
        V = dx_ids.size(1)
        e_dx_list, e_proc_list, e_med_list, e_lab_list, u_list = ([], [], [],
                                                                  [], [])

        for t in range(V):
            e_dx_t = self.E_dx[dx_ids[:, t, :]]
            e_proc_t = self.E_proc[proc_ids[:, t, :]]
            e_med_t = self.E_med[med_ids[:, t, :]]
            e_lab_t = self.E_lab[lab_ids[:, t, :]]
            e_vt_t = self.E_vtype[vtype_ids[:, t]]
            e_dx_m = self._masked_mean(e_dx_t, dx_ids[:, t, :])
            e_proc_m = self._masked_mean(e_proc_t, proc_ids[:, t, :])
            e_med_m = self._masked_mean(e_med_t, med_ids[:, t, :])
            e_lab_m = self._masked_mean(e_lab_t, lab_ids[:, t, :])
            e_dx_list.append(e_dx_m)
            e_proc_list.append(e_proc_m)
            e_med_list.append(e_med_m)
            e_lab_list.append(e_lab_m)
            v_in = torch.cat([e_dx_m, e_proc_m, e_med_m, e_lab_m, e_vt_t],
                             dim=-1)
            u_list.append(self.visit_mlp(v_in))

        U = torch.stack(u_list, dim=1)
        E_dx_seq = torch.stack(e_dx_list, dim=1)
        E_proc_seq = torch.stack(e_proc_list, dim=1)
        E_med_seq = torch.stack(e_med_list, dim=1)
        E_lab_seq = torch.stack(e_lab_list, dim=1)
        g_scores = self.global_score(U)
        g_scores = g_scores.masked_fill(~visit_mask.unsqueeze(-1),
                                        float('-inf'))

        g_attn = torch.softmax(g_scores, dim=1)
        z_global = (g_attn * U).sum(dim=1)
        field_seqs = [E_dx_seq, E_proc_seq, E_med_seq, E_lab_seq]
        anchors = [
            self._field_attn_pool(seq, visit_mask, self.field_proj[f],
                                  self.field_score[f])
            for f, seq in enumerate(field_seqs)
        ]
        a_dx, a_proc, a_med, a_lab = anchors
        z_hist = (a_dx + a_proc + a_med + a_lab) / 4.0
        pair_outs = []

        for i in range(4):
            for j in range(i + 1, 4):
                ai, aj = (anchors[i], anchors[j])
                p_ij = torch.cat([ai, aj, (ai - aj).abs(), ai * aj], dim=-1)
                pair_outs.append(self.pair_mlp(p_ij))

        z_pair = self.pair_fuse(torch.cat(pair_outs, dim=-1))
        z_s = self.fuse_mlp(torch.cat([z_global, z_hist, z_pair], dim=-1))

        return z_s


class DynamicRelationAdapter(nn.Module):
    def __init__(self,
                 hidden_size: int = HIDDEN_SIZE,
                 time_emb_dim: int = TIME_EMB_DIM,
                 rel_dim: int = REL_DIM) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.h_proj = nn.Linear(hidden_size, rel_dim)
        self.film = nn.Linear(time_emb_dim, 2 * rel_dim)
        self.q_proj = nn.Linear(3, rel_dim)
        self.out_proj = nn.Sequential(nn.Linear(2 * rel_dim, rel_dim),
                                      nn.GELU(), nn.LayerNorm(rel_dim))

        nn.init.normal_(self.film.weight, std=0.01)
        nn.init.zeros_(self.film.bias)
        nn.init.normal_(self.q_proj.weight, std=0.01)
        nn.init.zeros_(self.q_proj.bias)

    def forward(self, h: torch.Tensor, z_t: torch.Tensor,
                q3_input: torch.Tensor) -> torch.Tensor:
        z_h = self.h_proj(self.norm(h))
        gamma, beta = self.film(z_t).chunk(2, dim=-1)
        z_mod = gamma * z_h + beta
        z_q = self.q_proj(q3_input)

        return self.out_proj(torch.cat([z_mod, z_q], dim=-1))


class TimeConditioner(nn.Module):
    def __init__(self, q3_dim: int = 3, rel_dim: int = REL_DIM) -> None:
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(q3_dim, rel_dim), nn.GELU(),
                                 nn.Linear(rel_dim, 2 * rel_dim))

        for layer in (self.mlp[0], self.mlp[2]):
            nn.init.normal_(layer.weight, std=0.01)
            nn.init.zeros_(layer.bias)

    def forward(self, q3_input: torch.Tensor) -> torch.Tensor:
        return self.mlp(q3_input)


class RelationAdapter(nn.Module):
    def __init__(self,
                 stage1_emb: dict[str, torch.Tensor],
                 hidden_size: int = HIDDEN_SIZE,
                 time_emb_dim: int = TIME_EMB_DIM,
                 rel_dim: int = REL_DIM,
                 n_prefix: int = N_PREFIX,
                 disable_static: bool = False,
                 disable_dynamic: bool = False,
                 alpha_s_init: float = 0.1,
                 alpha_d_init: float = 0.03,
                 use_regime_confidence: bool = True,
                 branch_local_norm: bool = True,
                 use_time_conditioning: bool = False) -> None:
        super().__init__()
        self.n_prefix = n_prefix
        self.disable_static = disable_static
        self.disable_dynamic = disable_dynamic
        self.use_regime_confidence = use_regime_confidence
        self.branch_local_norm = branch_local_norm
        self.use_time_conditioning = use_time_conditioning
        self.static_branch = StaticRelationAdapter(stage1_emb, rel_dim)
        self.dynamic_branch = DynamicRelationAdapter(hidden_size, time_emb_dim,
                                                     rel_dim)

        self.alpha_s_raw = nn.Parameter(
            torch.full((1, ), _softplus_inverse(alpha_s_init)))

        self.alpha_d_raw = nn.Parameter(
            torch.full((1, ), _softplus_inverse(alpha_d_init)))

        norm_cls = nn.LayerNorm if branch_local_norm else nn.Identity
        self.static_rel_norm = norm_cls(rel_dim)
        self.dynamic_rel_norm = norm_cls(rel_dim)
        self.emb_proj = nn.Linear(rel_dim, n_prefix * hidden_size)
        nn.init.normal_(self.emb_proj.weight, std=0.01)
        nn.init.zeros_(self.emb_proj.bias)

        if use_time_conditioning:
            self.time_conditioner = TimeConditioner(q3_dim=3, rel_dim=rel_dim)

    def forward(self,
                h: torch.Tensor,
                z_t: torch.Tensor,
                conf_t: torch.Tensor | None = None,
                q3_input: torch.Tensor | None = None,
                gate: torch.Tensor | None = None,
                dx_ids: torch.Tensor | None = None,
                proc_ids: torch.Tensor | None = None,
                med_ids: torch.Tensor | None = None,
                lab_ids: torch.Tensor | None = None,
                vtype_ids: torch.Tensor | None = None,
                visit_mask: torch.Tensor | None = None) -> torch.Tensor:
        if any((x is None for x in
                [dx_ids, proc_ids, med_ids, lab_ids, vtype_ids, visit_mask])):
            raise ValueError(
                'RelationAdapter requires structured code inputs. dx_ids, proc_ids, med_ids, lab_ids, vtype_ids, and visit_mask must all be provided.'
            )

        if q3_input is None:
            raise ValueError('RelationAdapter requires q3_input from TSCM.')

        B = h.size(0)
        z_s = self.static_branch(dx_ids, proc_ids, med_ids, lab_ids, vtype_ids,
                                 visit_mask)

        z_d = self.dynamic_branch(h, z_t, q3_input)
        _, _, z_rel = self._merge_branches(z_s,
                                           z_d,
                                           conf_t=conf_t,
                                           apply_dropout=self.training)

        if self.use_time_conditioning and q3_input is not None:
            film_params = self.time_conditioner(q3_input)
            gamma, beta = film_params.chunk(2, dim=-1)
            if gate is not None:
                z_rel = (1.0 + gate * gamma) * z_rel + gate * beta
            else:
                z_rel = (1.0 + gamma) * z_rel + beta

        P_emb = self.emb_proj(z_rel)

        return P_emb.view(B, self.n_prefix, -1)

    def _merge_branches(
        self,
        z_s: torch.Tensor,
        z_d: torch.Tensor,
        conf_t: torch.Tensor | None = None,
        apply_dropout: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z_s_proc = self.static_rel_norm(z_s)
        z_d_proc = self.dynamic_rel_norm(z_d)

        if self.disable_static:
            z_s_proc = torch.zeros_like(z_s_proc)

        if self.disable_dynamic:
            z_d_proc = torch.zeros_like(z_d_proc)

        if conf_t is None or not self.use_regime_confidence:
            conf_scale = torch.ones(z_d_proc.size(0),
                                    1,
                                    device=z_d_proc.device,
                                    dtype=z_d_proc.dtype)

        else:
            conf_scale = conf_t.to(device=z_d_proc.device,
                                   dtype=z_d_proc.dtype)

        z_s_scaled = self.alpha_s * z_s_proc
        z_d_scaled = self.alpha_d * conf_scale * z_d_proc

        return (z_s_scaled, z_d_scaled, z_s_scaled + z_d_scaled)

    def get_z_embeddings(
        self,
        h: torch.Tensor,
        z_t: torch.Tensor,
        conf_t: torch.Tensor | None = None,
        q3_input: torch.Tensor | None = None,
        dx_ids: torch.Tensor | None = None,
        proc_ids: torch.Tensor | None = None,
        med_ids: torch.Tensor | None = None,
        lab_ids: torch.Tensor | None = None,
        vtype_ids: torch.Tensor | None = None,
        visit_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if any((x is None for x in
                [dx_ids, proc_ids, med_ids, lab_ids, vtype_ids, visit_mask])):
            raise ValueError(
                'RelationAdapter requires structured code inputs in get_z_embeddings.'
            )

        if q3_input is None:
            raise ValueError(
                'RelationAdapter requires q3_input in get_z_embeddings.')

        z_s = self.static_branch(dx_ids, proc_ids, med_ids, lab_ids, vtype_ids,
                                 visit_mask)

        z_d = self.dynamic_branch(h, z_t, q3_input)

        return self._merge_branches(z_s,
                                    z_d,
                                    conf_t=conf_t,
                                    apply_dropout=False)

    @property
    def alpha_s(self) -> torch.Tensor:
        return torch.nn.functional.softplus(self.alpha_s_raw)

    @property
    def alpha_d(self) -> torch.Tensor:
        return torch.nn.functional.softplus(self.alpha_d_raw)

    @property
    def n_trainable_params(self) -> int:
        return sum((p.numel() for p in self.parameters() if p.requires_grad))
