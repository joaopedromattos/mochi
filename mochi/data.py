"""Dataset loading for Mochi.

Three task families:

  * **Link prediction (LP)** — pre-processed AnyGraph-format pickles
    (``trn_mat.pkl``, ``feats.pkl``) under ``lp_data_root/<name>/``. We only
    need the training adjacency and node features; ``LPDataHandler`` wraps
    those and computes multi-hop SVD projectors.
  * **Node classification (NC)** — PyG ``Planetoid``, ``Coauthor``, ``Amazon``,
    OGB ``NodePropPred``, or CS-TAG CSVs.
  * **Graph classification (GC)** — PyG ``TUDataset`` or OGB ``GraphPropPred``.
"""

from __future__ import annotations

import os
import pickle
from typing import List, Optional, Tuple

import numpy as np
import scipy.sparse as sp
import torch

from .projectors import compute_projectors


# ══════════════════════════════════════════════════════════════════════════
# Node classification
# ══════════════════════════════════════════════════════════════════════════

NC_DATASETS = {
    "cora":      dict(cls="Planetoid", name="Cora"),
    "citeseer":  dict(cls="Planetoid", name="CiteSeer"),
    "pubmed":    dict(cls="Planetoid", name="PubMed"),
    "cs":        dict(cls="Coauthor",  name="CS"),
    "physics":   dict(cls="Coauthor",  name="Physics"),
    "computers": dict(cls="Amazon",    name="Computers"),
    "photo":     dict(cls="CSTAG",     name="Photo"),
    "arxiv":     dict(cls="OGB",       name="ogbn-arxiv"),
    "products":  dict(cls="OGB",       name="ogbn-products"),
    "Fitness":   dict(cls="CSTAG",     name="Fitness"),
}


def _load_cstag(dataset_name: str, root: str):
    """Load a CS-TAG CSV dataset. Returns (y, edge_index, num_nodes)."""
    import ast
    import pandas as pd

    csv_path = os.path.join(root, dataset_name, f"{dataset_name}.csv")
    df = pd.read_csv(csv_path)

    label_col = next((c for c in ("label", "label_id", "y") if c in df.columns), None)
    if label_col is None:
        raise ValueError(f"No label column in {csv_path} (found: {list(df.columns)})")

    if "node_id" in df.columns:
        df = df.sort_values("node_id").reset_index(drop=True)

    y = torch.tensor(df[label_col].to_numpy(), dtype=torch.long)
    node_ids = df["node_id"].to_numpy()
    neighbours = df["neighbour"].apply(ast.literal_eval).to_numpy()
    src = np.concatenate([np.full(len(nbs), nid, dtype=np.int64) for nid, nbs in zip(node_ids, neighbours)])
    dst = np.concatenate([np.array(nbs, dtype=np.int64) for nbs in neighbours])
    edge_index = torch.tensor(np.stack([src, dst]), dtype=torch.long)
    return y, edge_index, len(df)


def load_nc_dataset(key: str, root: str = "./data", cstag_root: Optional[str] = None):
    """Load a node-classification dataset. Returns (edge_index, N, y, feats|None)."""
    info = NC_DATASETS[key]
    pyg_root = os.path.join(root, "pyg")

    if info["cls"] == "Planetoid":
        from torch_geometric.datasets import Planetoid
        data = Planetoid(root=pyg_root, name=info["name"])[0]
    elif info["cls"] == "Coauthor":
        from torch_geometric.datasets import Coauthor
        data = Coauthor(root=pyg_root, name=info["name"])[0]
    elif info["cls"] == "Amazon":
        from torch_geometric.datasets import Amazon
        data = Amazon(root=pyg_root, name=info["name"])[0]
    elif info["cls"] == "OGB":
        from ogb.nodeproppred import PygNodePropPredDataset
        data = PygNodePropPredDataset(name=info["name"], root=os.path.join(root, "ogb"))[0]
    elif info["cls"] == "CSTAG":
        if cstag_root is None:
            raise ValueError(f"{key} requires cstag_root to be set")
        y, edge_index, num_nodes = _load_cstag(info["name"], root=cstag_root)
        return edge_index, num_nodes, y, None
    else:
        raise ValueError(info["cls"])

    feats = data.x if data.x is not None else None
    return data.edge_index, data.num_nodes, data.y.view(-1).long(), feats


# ══════════════════════════════════════════════════════════════════════════
# Graph classification
# ══════════════════════════════════════════════════════════════════════════

GC_DATASETS = {
    "MUTAG":            dict(cls="TUDataset",  name="MUTAG"),
    "PROTEINS":         dict(cls="TUDataset",  name="PROTEINS"),
    "NCI1":             dict(cls="TUDataset",  name="NCI1"),
    "IMDB-BINARY":      dict(cls="TUDataset",  name="IMDB-BINARY"),
    "COLLAB":           dict(cls="TUDataset",  name="COLLAB"),
    "DD":               dict(cls="TUDataset",  name="DD"),
    "ENZYMES":          dict(cls="TUDataset",  name="ENZYMES"),
    "REDDIT-MULTI-5K":  dict(cls="TUDataset",  name="REDDIT-MULTI-5K"),
    "REDDIT-MULTI-12K": dict(cls="TUDataset",  name="REDDIT-MULTI-12K"),
    "ogbg-ppa":         dict(cls="OGB-Graph",  name="ogbg-ppa"),
}


def load_gc_dataset(key: str, root: str = "./data"):
    """Load a graph-classification dataset as (list of PyG Data, labels)."""
    info = GC_DATASETS[key]
    if info["cls"] == "OGB-Graph":
        from ogb.graphproppred import PygGraphPropPredDataset
        dataset = PygGraphPropPredDataset(name=info["name"], root=os.path.join(root, "ogb"))
        graphs = list(dataset)
        y = torch.tensor([g.y.view(-1)[0].item() for g in graphs], dtype=torch.long)
    else:
        from torch_geometric.datasets import TUDataset
        dataset = TUDataset(root=os.path.join(root, "pyg"), name=info["name"])
        graphs = list(dataset)
        y = torch.tensor([g.y.item() for g in graphs], dtype=torch.long)
    return graphs, y


def compute_gc_projectors(graphs, *, latdim=512, gnn_layers=3, niter=2, device="cpu",
                          cache_dir: Optional[str] = None, dataset_name: Optional[str] = None):
    """Compute multi-hop SVD projectors independently for every graph."""
    if cache_dir and dataset_name:
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir,
                                  f"gc_{dataset_name}_d{latdim}_g{gnn_layers}_n{niter}_mh.pt")
        if os.path.exists(cache_path):
            print(f"    (cached) {cache_path}")
            return torch.load(cache_path, map_location="cpu", weights_only=True)
    else:
        cache_path = None

    per_graph = []
    for g in graphs:
        feats = g.x if g.x is not None else None
        proj = compute_projectors(
            g.edge_index, g.num_nodes, feats,
            latdim=latdim, gnn_layers=gnn_layers, niter=niter, device=device,
        ).cpu()
        per_graph.append(proj)

    if cache_path:
        torch.save(per_graph, cache_path)
        print(f"    cached → {cache_path}")
    return per_graph


# ══════════════════════════════════════════════════════════════════════════
# Link prediction — simplified DataHandler for AnyGraph-format pickles
# ══════════════════════════════════════════════════════════════════════════


def _load_pickle_sparse(path: str) -> sp.coo_matrix:
    with open(path, "rb") as fs:
        ret = (pickle.load(fs) != 0).astype(np.float32)
    if not isinstance(ret, sp.coo_matrix):
        ret = sp.coo_matrix(ret)
    return ret


def _normalize_adj(mat: sp.spmatrix) -> sp.coo_matrix:
    """Row-normalize (or sym-normalize) an adjacency for propagation."""
    if mat.shape[0] == mat.shape[1]:
        degree = np.array(mat.sum(axis=-1)).flatten()
        d_inv_sqrt = np.power(degree, -0.5)
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
        D = sp.diags(d_inv_sqrt)
        return mat.dot(D).transpose().dot(D).tocoo()
    # Bipartite: D^{-1/2} on each side
    row_deg = np.array(mat.sum(axis=-1)).flatten()
    col_deg = np.array(mat.sum(axis=0)).flatten()
    row_inv = np.power(row_deg, -0.5); row_inv[np.isinf(row_inv)] = 0.0
    col_inv = np.power(col_deg, -0.5); col_inv[np.isinf(col_inv)] = 0.0
    return sp.diags(row_inv).dot(mat).dot(sp.diags(col_inv)).tocoo()


def _sparse_coo_to_torch(mat: sp.coo_matrix) -> torch.Tensor:
    idxs = torch.from_numpy(np.vstack([mat.row, mat.col]).astype(np.int64))
    vals = torch.from_numpy(mat.data.astype(np.float32))
    return torch.sparse_coo_tensor(idxs, vals, torch.Size(mat.shape))


def _symmetrize_and_normalize(trn_mat: sp.coo_matrix) -> torch.Tensor:
    """Symmetrize homogeneous graphs, or build (U+V)x(U+V) from bipartite data."""
    if trn_mat.shape[0] == trn_mat.shape[1]:
        N = trn_mat.shape[0]
        row = np.concatenate([trn_mat.row, trn_mat.col]).astype(np.int64)
        col = np.concatenate([trn_mat.col, trn_mat.row]).astype(np.int64)
        # deduplicate
        hash_vals = row * N + col
        hash_vals = np.unique(hash_vals)
        col = hash_vals % N
        row = (hash_vals - col) // N
        vals = np.ones_like(row, dtype=np.float32)
        mat = sp.coo_matrix((vals, (row, col)), (N, N))
        return _sparse_coo_to_torch(_normalize_adj(mat))

    # Bipartite → build block matrix [[0, B], [B^T, 0]]
    U, V = trn_mat.shape
    zero_u = sp.csr_matrix((U, U))
    zero_v = sp.csr_matrix((V, V))
    stacked = sp.vstack([sp.hstack([zero_u, trn_mat]),
                         sp.hstack([trn_mat.transpose(), zero_v])])
    stacked = (stacked != 0).astype(np.float32)
    return _sparse_coo_to_torch(_normalize_adj(stacked))


class LPDataHandler:
    """Wraps an AnyGraph-format LP dataset (``trn_mat.pkl`` + optional ``feats.pkl``).

    The handler loads the raw training adjacency, symmetrizes it, and computes
    the multi-hop SVD projectors that Mochi consumes. Only the attributes that
    downstream code actually uses are exposed:

      * ``trn_mat`` — raw COO adjacency (used by :class:`LinkEpisodeSampler`).
      * ``projectors`` — ``[N, (K+1)*latdim]`` tensor.
      * ``data_name`` — dataset key.
    """

    def __init__(self, data_name: str, *, data_root: str,
                 latdim: int, gnn_layer: int, niter: int = 2,
                 device: str = "cpu", cache_dir: Optional[str] = None):
        self.data_name = data_name
        self.latdim = latdim
        self.gnn_layer = gnn_layer
        self.niter = niter

        predir = os.path.join(data_root, data_name)
        self.trn_mat = _load_pickle_sparse(os.path.join(predir, "trn_mat.pkl"))

        feat_path = os.path.join(predir, "feats.pkl")
        if os.path.exists(feat_path):
            with open(feat_path, "rb") as fs:
                feats = pickle.load(fs)
            self.feats = torch.from_numpy(feats).float()
        else:
            self.feats = None

        self.projectors = self._make_projectors(device=device, cache_dir=cache_dir)

    # -- projectors ---------------------------------------------------------
    def _cache_path(self, cache_dir: str) -> str:
        os.makedirs(cache_dir, exist_ok=True)
        tag = f"lp_{self.data_name}_d{self.latdim}_g{self.gnn_layer}_n{self.niter}_mh"
        return os.path.join(cache_dir, f"{tag}.pt")

    def _make_projectors(self, *, device: str, cache_dir: Optional[str]):
        if cache_dir:
            cp = self._cache_path(cache_dir)
            if os.path.exists(cp):
                print(f"    (cached) {cp}")
                return torch.load(cp, map_location="cpu", weights_only=True)

        # Build SVD over adj (+ feats), then propagate K hops through the normalized adj.
        # Use our own projector helpers to stay dataset-agnostic.
        from .projectors import svd_adj, svd_feat, _gnn_propagate_multihop
        adj_norm = _symmetrize_and_normalize(self.trn_mat)

        with torch.no_grad():
            proj = svd_adj(adj_norm, latdim=self.latdim, niter=self.niter, device=device)
            if self.feats is not None and self.feats.numel() > 0 and self.feats.shape[1] > 1:
                # If feats is shorter than total nodes (bipartite case), pad with zeros at the item side.
                if self.feats.shape[0] != adj_norm.shape[0]:
                    pad = torch.zeros(adj_norm.shape[0] - self.feats.shape[0], self.feats.shape[1])
                    feats = torch.cat([self.feats, pad], dim=0)
                else:
                    feats = self.feats
                feat_p = svd_feat(feats, latdim=self.latdim, niter=self.niter, device=device)
                proj = proj + feat_p.to(proj.device)
            proj = _gnn_propagate_multihop(adj_norm.to(device), proj, n_layers=self.gnn_layer)

        proj = proj.cpu()
        if cache_dir:
            torch.save(proj, self._cache_path(cache_dir))
            print(f"    cached → {self._cache_path(cache_dir)}")
        return proj


def load_lp_datasets(dataset_names: List[str], *, data_root: str,
                     latdim: int, gnn_layer: int, niter: int = 2,
                     device: str = "cpu", cache_dir: Optional[str] = None) -> List[LPDataHandler]:
    """Load + project every LP dataset in ``dataset_names``."""
    handlers = []
    for name in dataset_names:
        try:
            h = LPDataHandler(name, data_root=data_root,
                              latdim=latdim, gnn_layer=gnn_layer,
                              niter=niter, device=device, cache_dir=cache_dir)
            print(f"  [LP] {name}: {h.trn_mat.shape[0]} nodes, {h.trn_mat.nnz} edges")
            handlers.append(h)
        except Exception as e:
            print(f"  [LP] Failed to load {name}: {e}")
    return handlers


def load_nc_datasets(dataset_names: List[str], *, data_root: str,
                     latdim: int, gnn_layer: int, niter: int = 2,
                     device: str = "cpu", cache_dir: Optional[str] = None,
                     cstag_root: Optional[str] = None) -> List[Tuple[torch.Tensor, torch.Tensor, str]]:
    """Load NC datasets and compute projectors. Returns (projectors, y, name) tuples."""
    out = []
    for name in dataset_names:
        if name not in NC_DATASETS:
            print(f"  [NC] Unknown dataset {name}, skipping")
            continue
        try:
            edge_index, N, y, feats = load_nc_dataset(name, root=data_root, cstag_root=cstag_root)
            proj = _load_or_compute_nc_projectors(
                name, edge_index, N, feats,
                latdim=latdim, gnn_layer=gnn_layer, niter=niter,
                device=device, cache_dir=cache_dir,
            )
            out.append((proj, y, name))
            print(f"  [NC] {name}: {N} nodes, {len(torch.unique(y))} classes")
        except Exception as e:
            print(f"  [NC] Failed to load {name}: {e}")
    return out


def _load_or_compute_nc_projectors(name, edge_index, N, feats, *,
                                   latdim, gnn_layer, niter, device, cache_dir):
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        cp = os.path.join(cache_dir, f"nc_{name}_d{latdim}_g{gnn_layer}_n{niter}_mh.pt")
        if os.path.exists(cp):
            print(f"    (cached) {cp}")
            return torch.load(cp, map_location="cpu", weights_only=True)

    try:
        proj = compute_projectors(edge_index, N, feats,
                                  latdim=latdim, gnn_layers=gnn_layer, niter=niter,
                                  device=device)
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        if "out of memory" in str(e).lower() or isinstance(e, torch.cuda.OutOfMemoryError):
            torch.cuda.empty_cache()
            print("    GPU OOM during projection, falling back to CPU…")
            proj = compute_projectors(edge_index, N, feats,
                                      latdim=latdim, gnn_layers=gnn_layer, niter=niter,
                                      device="cpu")
        else:
            raise
    proj = proj.cpu()
    if cache_dir:
        torch.save(proj, cp)
        print(f"    cached → {cp}")
    return proj


def load_gc_datasets(dataset_names: List[str], *, data_root: str,
                     latdim: int, gnn_layer: int, niter: int = 2,
                     device: str = "cpu", cache_dir: Optional[str] = None):
    """Load GC datasets and compute per-graph projectors. Returns (per_graph, y, name) tuples."""
    out = []
    for name in dataset_names:
        if name not in GC_DATASETS:
            print(f"  [GC] Unknown dataset {name}, skipping")
            continue
        try:
            graphs, y = load_gc_dataset(name, root=data_root)
            per_graph = compute_gc_projectors(
                graphs, latdim=latdim, gnn_layers=gnn_layer, niter=niter,
                device=device, cache_dir=cache_dir, dataset_name=name,
            )
            out.append((per_graph, y, name))
            print(f"  [GC] {name}: {len(y)} graphs, {len(torch.unique(y))} classes")
        except Exception as e:
            print(f"  [GC] Failed to load {name}: {e}")
    return out
