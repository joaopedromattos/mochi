"""Mochi / Mochi++ model.

Architecture (both variants):

    projectors (frozen multi-hop SVD features)
        → GAMLP (trainable)                  ← node-adaptive hop attention
        → optional task-specific pooling     ← NC: gather, LP: Hadamard, GC: mean-pool
        → differentiable ridge readout       ← R2-D2 closed-form classifier

The only difference between the two variants is:

  * The residual + depth of GAMLP's output MLP.
  * How training batches episodes (handled in ``training.py``).

Everything else is shared.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ══════════════════════════════════════════════════════════════════════════
# GAMLP encoder (node-adaptive hop attention)
# ══════════════════════════════════════════════════════════════════════════


class GAMLPEncoder(nn.Module):
    """GAMLP-style encoder operating on multi-hop SVD projectors.

    Input: ``[N, (K+1) * latdim]`` where each ``latdim`` block is hop
    ``k = 0, 1, ..., K``. Output: ``[N, latdim]``.

    Args:
        num_hops:   K — number of propagation hops used when building projectors.
        latdim:     feature dimensionality per hop.
        dropout:    dropout rate inside the MLP branches.
        deep_head:  Mochi++ uses a deeper 3-layer out_mlp with a residual skip.
                    Mochi uses the shallower 2-layer variant without residual.
    """

    def __init__(self, num_hops: int, latdim: int, dropout: float = 0.0,
                 deep_head: bool = True):
        super().__init__()
        self.num_hops = num_hops
        self.latdim = latdim
        self.dropout = dropout
        self.deep_head = deep_head

        self.input_proj = nn.ModuleList([
            nn.Linear(latdim, latdim) for _ in range(num_hops + 1)
        ])

        # Node-adaptive hop weighting (GAMLP-style)
        self.att_mlp = nn.Sequential(
            nn.Linear(2 * latdim, latdim),
            nn.ReLU(),
            nn.Linear(latdim, 1),
        )

        if deep_head:
            self.out_mlp = nn.Sequential(
                nn.Linear(latdim, latdim), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(latdim, latdim), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(latdim, latdim),
            )
        else:
            self.out_mlp = nn.Sequential(
                nn.Linear(latdim, latdim), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(latdim, latdim),
            )

    def forward(self, embeds: torch.Tensor) -> torch.Tensor:
        xs = embeds.split(self.latdim, dim=-1)

        hs = []
        for k, x in enumerate(xs):
            h = torch.relu(self.input_proj[k](x))
            h = F.dropout(h, p=self.dropout, training=self.training)
            hs.append(h)

        # Use hop-0 as the query for hop attention
        h0 = hs[0]
        att_scores = torch.cat([self.att_mlp(torch.cat([h0, h], dim=-1)) for h in hs], dim=-1)
        att = torch.softmax(att_scores, dim=-1)           # [N, K+1]

        h_stack = torch.stack(hs, dim=1)                  # [N, K+1, latdim]
        h = (h_stack * att.unsqueeze(-1)).sum(dim=1)      # [N, latdim]

        if self.deep_head:
            return h + self.out_mlp(h)
        return self.out_mlp(h)


# ══════════════════════════════════════════════════════════════════════════
# Differentiable Ridge Readout (R2-D2, Bertinetto et al. 2019)
# ══════════════════════════════════════════════════════════════════════════


class DifferentiableRidge(nn.Module):
    """Closed-form ridge classifier in dual (Woodbury) form.

    Efficient when N_support << d_features — the N×N solve replaces a d×d solve,
    and gradients flow through ``torch.linalg.solve`` back to the support/query.
    """

    def __init__(self, lambda_reg: float = 10.0):
        super().__init__()
        self.lambda_reg = lambda_reg

    def forward(self, support_feat, support_labels, query_feat, n_classes):
        N_s = support_feat.size(0)
        device = support_feat.device

        X = torch.cat([support_feat, torch.ones(N_s, 1, device=device)], dim=1)  # [N_s, d+1]
        Y = F.one_hot(support_labels.long(), n_classes).float()                   # [N_s, n_classes]

        XXt = X @ X.T + self.lambda_reg * torch.eye(N_s, device=device)
        alpha = torch.linalg.solve(XXt, Y)
        W_star = X.T @ alpha                                                       # [d+1, n_classes]

        Q = torch.cat([query_feat, torch.ones(query_feat.size(0), 1, device=device)], dim=1)
        return Q @ W_star                                                          # [N_q, n_classes]


# ══════════════════════════════════════════════════════════════════════════
# Full Mochi model
# ══════════════════════════════════════════════════════════════════════════


class MochiModel(nn.Module):
    """GAMLP encoder + differentiable ridge readout.

    Episode structure (supplied by the data pipeline):
      - ``nc``: ``{support_idx, support_y, query_idx, query_y, n_way}`` — gather on Z
      - ``lp``: ``{support_src, support_dst, query_src, query_dst, support_y, query_y}``
                — Hadamard product of endpoint embeddings
      - ``gc``: ``{support_idx, support_y, query_idx, query_y, n_way}`` plus per-graph
                projectors — scatter_mean pools node embeddings per graph
    """

    def __init__(self, num_hops: int, latdim: int, *,
                 dropout: float = 0.0, deep_head: bool = True,
                 ridge_lambda: float = 10.0):
        super().__init__()
        self.encoder = GAMLPEncoder(num_hops, latdim, dropout=dropout,
                                    deep_head=deep_head)
        self.ridge = DifferentiableRidge(lambda_reg=ridge_lambda)
        self.log_temp = nn.Parameter(torch.tensor(1.0).log())

    # -- task-specific pooling ------------------------------------------------
    def _encode_episode(self, Z, episode, task_type):
        if task_type == "nc" or task_type == "gc":
            sup = Z[episode["support_idx"]]
            qry = Z[episode["query_idx"]]
            return sup, episode["support_y"], qry, episode["query_y"], episode["n_way"]
        if task_type == "lp":
            sup = Z[episode["support_src"]] * Z[episode["support_dst"]]
            qry = Z[episode["query_src"]] * Z[episode["query_dst"]]
            return sup, episode["support_y"], qry, episode["query_y"], episode["n_way"]
        raise ValueError(f"Unknown task type: {task_type}")

    def forward(self, projectors, episode, task_type, gc_graph_projectors=None):
        """
        Args:
            projectors: ``[M, (K+1)*latdim]`` episode-only node projectors (NC/LP).
                        Set to ``None`` for GC.
            episode: episode dict with (pre-remapped) indices local to ``projectors``.
            task_type: ``'nc'`` | ``'lp'`` | ``'gc'``.
            gc_graph_projectors: list of ``[n_i, (K+1)*latdim]`` per-graph projectors
                                  for GC episodes; otherwise ``None``.
        """
        if task_type == "gc":
            from torch_scatter import scatter_mean
            batch_ids = [torch.full((gp.size(0),), i, dtype=torch.long, device=gp.device)
                         for i, gp in enumerate(gc_graph_projectors)]
            concat = torch.cat(gc_graph_projectors, dim=0)
            batch_vec = torch.cat(batch_ids, dim=0)
            Z_nodes = self.encoder(concat)
            Z = scatter_mean(Z_nodes, batch_vec, dim=0)
        else:
            Z = self.encoder(projectors)

        sup, sup_y, qry, qry_y, n_classes = self._encode_episode(Z, episode, task_type)
        temp = self.log_temp.exp()
        sup = F.normalize(sup, dim=-1) * temp
        qry = F.normalize(qry, dim=-1) * temp
        logits = self.ridge(sup, sup_y, qry, n_classes)
        return logits, qry_y


# ══════════════════════════════════════════════════════════════════════════
# Public factory
# ══════════════════════════════════════════════════════════════════════════


def Mochi(**params) -> MochiModel:
    """Build a Mochi or Mochi++ model.

    Accepts any subset of :class:`MochiConfig` fields (unknown keys are ignored).
    ``model_variant`` controls which head is used:

      * ``"mochi"``   — shallow out_mlp, no residual.
      * ``"mochi++"`` — 3-layer out_mlp with residual (default, paper setup).

    Example::

        from mochi import Mochi, default_params
        model = Mochi(**default_params)          # Mochi++
        model = Mochi(model_variant="mochi")     # Mochi
    """
    from .config import MochiConfig

    valid_keys = {f.name for f in MochiConfig.__dataclass_fields__.values()}
    cfg_kwargs = {k: v for k, v in params.items() if k in valid_keys}
    cfg = MochiConfig(**cfg_kwargs)

    deep_head = cfg.model_variant == "mochi++"

    return MochiModel(
        num_hops=cfg.gnn_layer,
        latdim=cfg.latdim,
        dropout=cfg.drop_rate,
        deep_head=deep_head,
        ridge_lambda=cfg.ridge_lambda,
    )
