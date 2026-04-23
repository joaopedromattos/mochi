"""SVD-based feature/adjacency projectors + multi-hop GNN propagation.

These precompute the frozen input to Mochi's trainable GAMLP encoder. The
projection pipeline is:

    x = SVD(A)                 # adjacency structure
    x = x + SVD(X)             # add raw-feature SVD when available
    x = LayerNorm(x)
    hops = [x, A x, A^2 x, ..., A^K x]
    proj = concat(hops, dim=-1)   # multi-hop stack for GAMLP

The result is a ``[N, (K+1) * latdim]`` tensor per (sub)graph.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn


def _sym_normalize_scipy(adj_coo):
    """D^{-1/2} A D^{-1/2}."""
    degree = np.array(adj_coo.sum(axis=-1)).flatten()
    d_inv_sqrt = np.power(degree, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    D = sp.diags(d_inv_sqrt)
    return (D @ adj_coo @ D).tocoo()


def make_normalized_adj(edge_index, num_nodes):
    """Symmetric, deduplicated, D^{-1/2} A D^{-1/2} as a torch sparse COO tensor."""
    from torch_geometric.utils import to_undirected
    ei = to_undirected(edge_index, num_nodes=num_nodes)
    row, col = ei[0].numpy(), ei[1].numpy()
    vals = np.ones(len(row), dtype=np.float32)
    adj = sp.coo_matrix((vals, (row, col)), shape=(num_nodes, num_nodes))
    adj_norm = _sym_normalize_scipy(adj)
    indices = torch.tensor(np.stack([adj_norm.row, adj_norm.col]), dtype=torch.long)
    values = torch.tensor(adj_norm.data, dtype=torch.float32)
    return torch.sparse_coo_tensor(indices, values, (num_nodes, num_nodes))


def svd_adj(adj_sparse, latdim=512, niter=2, device="cpu"):
    """SVD of normalized adjacency → [N, latdim]."""
    adj = adj_sparse.to(device)
    N = adj.shape[0]
    q = min(latdim, N)
    u, s, v = torch.svd_lowrank(adj, q=q, niter=niter)
    sqrt_s = torch.sqrt(s)
    proj = u * sqrt_s.unsqueeze(0) + v * sqrt_s.unsqueeze(0)
    if q < latdim:
        proj = torch.cat([proj, torch.zeros(N, latdim - q, device=device)], dim=1)
    return proj


def svd_feat(feats, latdim=512, niter=2, device="cpu"):
    """SVD of feature matrix → [N, latdim] (flipped to keep large singular values last)."""
    feats = feats.float().to(device)
    N, D = feats.shape
    q = min(latdim, N, D)
    u, s, _ = torch.svd_lowrank(feats, q=q, niter=niter)
    proj = u * torch.sqrt(s).unsqueeze(0)
    if q < latdim:
        proj = torch.cat([proj, torch.zeros(N, latdim - q, device=device)], dim=1)
    return torch.flip(proj, dims=[-1])


def _gnn_propagate_multihop(adj_sparse, embeds, n_layers=3):
    """Return [x, A x, A^2 x, ..., A^K x] concatenated on the feature axis."""
    ln = nn.LayerNorm(embeds.shape[1], elementwise_affine=False).to(embeds.device)
    embeds = ln(embeds)
    hops = [embeds]
    x = embeds
    for _ in range(n_layers):
        x = torch.spmm(adj_sparse, x)
        hops.append(x)
    return torch.cat(hops, dim=-1)


def compute_projectors(edge_index, num_nodes, feats=None, *,
                       latdim=512, gnn_layers=3, niter=2, device="cpu"):
    """Full projection pipeline — returns ``[N, (gnn_layers + 1) * latdim]``."""
    adj_norm = make_normalized_adj(edge_index, num_nodes)
    with torch.no_grad():
        proj = svd_adj(adj_norm, latdim=latdim, niter=niter, device=device)
        if feats is not None and feats.numel() > 0 and feats.shape[1] > 1:
            feat_p = svd_feat(feats, latdim=latdim, niter=niter, device=device)
            proj = proj + feat_p.to(proj.device)
        proj = _gnn_propagate_multihop(adj_norm.to(device), proj, n_layers=gnn_layers)
    return proj.cpu()
