"""High-level ``build_datasets`` helper that bundles the whole data pipeline.

Given a :class:`MochiConfig`, this function resolves default paths, loads every
LP/NC/GC dataset listed in the config, computes projectors (with caching), and
returns the tuple of objects the training loop consumes.
"""

from __future__ import annotations

import os
from typing import Tuple

import torch

from .config import MochiConfig, LP_DATASET_GROUPS
from .data import load_lp_datasets, load_nc_datasets, load_gc_datasets
from .samplers import MultiTaskEpisodeSampler


def _resolve_paths(cfg: MochiConfig, repo_root: str):
    data_root = cfg.data_root or os.path.join(repo_root, "data")
    lp_root = cfg.lp_data_root or os.path.join(repo_root, "data", "lp")
    cache_dir = cfg.cache_dir or os.path.join(repo_root, "cache", "projectors")
    return data_root, lp_root, cache_dir


def build_datasets(cfg: MochiConfig, repo_root: str = ".") -> Tuple:
    """Load every dataset in ``cfg`` and return (sampler, lp, nc, gc, device).

    Side effects: creates caches under ``cfg.cache_dir`` and picks the GPU
    device requested by ``cfg.gpu``.
    """
    device = torch.device(f"cuda:{cfg.gpu}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(device)

    data_root, lp_root, cache_dir = _resolve_paths(cfg, repo_root)

    print("Loading datasets...")

    lp_names = LP_DATASET_GROUPS.get(cfg.dataset_setting, [cfg.dataset_setting])
    lp_handlers = load_lp_datasets(
        lp_names, data_root=lp_root,
        latdim=cfg.latdim, gnn_layer=cfg.gnn_layer, niter=cfg.niter,
        device=str(device), cache_dir=cache_dir,
    )

    nc_data = (load_nc_datasets(
        cfg.nc_datasets, data_root=data_root,
        latdim=cfg.latdim, gnn_layer=cfg.gnn_layer, niter=cfg.niter,
        device=str(device), cache_dir=cache_dir,
        cstag_root=cfg.cstag_root,
    ) if cfg.nc_datasets else [])

    gc_data = (load_gc_datasets(
        cfg.gc_datasets, data_root=data_root,
        latdim=cfg.latdim, gnn_layer=cfg.gnn_layer, niter=cfg.niter,
        device=str(device), cache_dir=cache_dir,
    ) if cfg.gc_datasets else [])

    print(f"Loaded: {len(lp_handlers)} LP, {len(nc_data)} NC, {len(gc_data)} GC datasets")

    nc_labels = [(y, name) for _, y, name in nc_data]
    gc_labels = [(y, name) for _, y, name in gc_data]
    sampler = MultiTaskEpisodeSampler(
        lp_handlers=lp_handlers,
        nc_labels_list=nc_labels,
        gc_labels_list=gc_labels,
        task_weights=cfg.task_weights,
    )
    print(f"Task types: {sampler.task_types}, probs: {sampler.task_probs}")

    return sampler, lp_handlers, nc_data, gc_data, device
