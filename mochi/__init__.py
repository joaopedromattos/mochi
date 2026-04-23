"""Mochi & Mochi++ — meta-learned few-shot graph foundation model.

Quickstart
----------

    from mochi import Mochi, default_params
    model = Mochi(**default_params)          # Mochi++ (paper default)
    model = Mochi(model_variant="mochi")     # Mochi (ablation)

End-to-end training and evaluation:

    from mochi import Mochi, MochiConfig, build_datasets, train, evaluate

    cfg = MochiConfig(**default_params)
    model = Mochi(**cfg.as_dict())
    sampler, lp, nc, gc, device = build_datasets(cfg)
    train(model, sampler, lp, nc, gc, device, cfg)
    evaluate(model, sampler, lp, nc, gc, device, cfg)

Or just run ``python train.py`` for the CLI.
"""

from .config import MochiConfig, default_params, LP_DATASET_GROUPS
from .model import Mochi, MochiModel, GAMLPEncoder, DifferentiableRidge
from .samplers import (
    NodeEpisodeSampler, LinkEpisodeSampler, GraphEpisodeSampler,
    MultiTaskEpisodeSampler,
)
from .data import (
    load_lp_datasets, load_nc_datasets, load_gc_datasets,
    load_nc_dataset, load_gc_dataset, LPDataHandler,
    NC_DATASETS, GC_DATASETS,
)
from .projectors import compute_projectors
from .training import (
    train, train_mochi, train_mochi_plus,
    evaluate, save_embeddings, seed_everything,
)
from .entrypoint import build_datasets
from .pretrained import (
    load_pretrained, download_checkpoint, checkpoint_filename, HF_REPO_ID,
)

__all__ = [
    # Config
    "MochiConfig", "default_params", "LP_DATASET_GROUPS",
    # Model
    "Mochi", "MochiModel", "GAMLPEncoder", "DifferentiableRidge",
    # Samplers
    "NodeEpisodeSampler", "LinkEpisodeSampler", "GraphEpisodeSampler",
    "MultiTaskEpisodeSampler",
    # Data
    "LPDataHandler", "load_lp_datasets", "load_nc_datasets", "load_gc_datasets",
    "load_nc_dataset", "load_gc_dataset", "compute_projectors",
    "NC_DATASETS", "GC_DATASETS",
    # Training
    "train", "train_mochi", "train_mochi_plus",
    "evaluate", "save_embeddings", "seed_everything",
    "build_datasets",
    # Pretrained weights
    "load_pretrained", "download_checkpoint", "checkpoint_filename", "HF_REPO_ID",
]
