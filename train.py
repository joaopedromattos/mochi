#!/usr/bin/env python3
"""Mochi / Mochi++ single-GPU training entry point.

Runs the full pipeline: load datasets → compute projectors → episodic
meta-training → episodic evaluation → export encoder embeddings.

Examples
--------

Default Mochi++ config (matches the paper)::

    python train.py

Override any ``MochiConfig`` field from the command line::

    python train.py --model_variant mochi --seed 1 --gpu 0
    python train.py --dataset_setting smoke --nc_datasets cora --train_steps 100
    python train.py --eval_only --load_model checkpoints/mochi_s2.pt
"""

import argparse
import os
import sys

import torch

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _REPO_ROOT)

from mochi import (
    Mochi, MochiConfig, default_params, build_datasets,
    train, evaluate, save_embeddings, seed_everything,
)


def _build_parser() -> argparse.ArgumentParser:
    import dataclasses
    p = argparse.ArgumentParser(description="Mochi / Mochi++ training")

    # All MochiConfig fields, auto-generated so defaults stay in sync.
    for f in MochiConfig.__dataclass_fields__.values():
        arg = f"--{f.name}"
        has_factory = f.default_factory is not dataclasses.MISSING
        default_val = f.default_factory() if has_factory else f.default
        if isinstance(default_val, list):
            p.add_argument(arg, nargs="*", default=default_val)
        elif isinstance(default_val, bool):
            p.add_argument(arg, action="store_true", default=default_val)
        elif isinstance(default_val, int):
            p.add_argument(arg, type=int, default=default_val)
        elif isinstance(default_val, float):
            p.add_argument(arg, type=float, default=default_val)
        else:
            p.add_argument(arg, type=str, default=default_val)

    # Script-only options
    p.add_argument("--eval_only", action="store_true", help="Skip training")
    p.add_argument("--load_model", default=None, help="Checkpoint path to load")
    p.add_argument("--no_save_embeddings", action="store_true",
                   help="Skip saving encoder embeddings at the end")
    return p


def main():
    parser = _build_parser()
    args = parser.parse_args()

    # Apply paper defaults first, then override with any user CLI flags.
    cfg_kwargs = dict(default_params)
    for k, v in vars(args).items():
        if k in MochiConfig.__dataclass_fields__:
            cfg_kwargs[k] = v
    cfg = MochiConfig(**cfg_kwargs)

    seed_everything(cfg.seed)

    print(f"Variant: {cfg.model_variant}  |  seed={cfg.seed}  |  gpu={cfg.gpu}")
    sampler, lp_handlers, nc_data, gc_data, device = build_datasets(cfg, repo_root=_REPO_ROOT)

    model = Mochi(**cfg.as_dict()).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params/1e6:.2f}M params ({cfg.model_variant})")

    if args.load_model:
        model.load_state_dict(torch.load(args.load_model, map_location=device))
        print(f"Loaded checkpoint: {args.load_model}")

    if not args.eval_only:
        print(f"\nStarting episodic meta-training ({cfg.train_steps} steps)...")
        train(model, sampler, lp_handlers, nc_data, gc_data, device, cfg)

        ckpt_dir = os.path.join(_REPO_ROOT, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt_path = os.path.join(ckpt_dir, f"{cfg.model_variant}_s{cfg.seed}.pt")
        torch.save(model.state_dict(), ckpt_path)
        print(f"Checkpoint saved: {ckpt_path}")

    print("\nEpisodic evaluation...")
    evaluate(model, sampler, lp_handlers, nc_data, gc_data, device, cfg)

    if not args.no_save_embeddings:
        print("\nSaving embeddings...")
        save_dir = os.path.join(_REPO_ROOT, "outputs")
        save_embeddings(model, lp_handlers, nc_data, gc_data, device, save_dir, seed=cfg.seed)


if __name__ == "__main__":
    main()
