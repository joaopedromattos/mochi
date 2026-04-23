#!/usr/bin/env python3
"""Mochi / Mochi++ multi-GPU DDP training.

Each rank independently samples episodes; DDP averages gradients across ranks,
effectively scaling the number of episodes per optimiser step by ``world_size``.
Evaluation runs on rank 0 only.

Launch::

    torchrun --nproc_per_node=4 train_ddp.py --gpus 0,1,2,3
    torchrun --nproc_per_node=2 train_ddp.py --gpus 2,3
"""

import argparse
import os
import random
import sys

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _REPO_ROOT)

from mochi import (
    Mochi, MochiConfig, default_params, build_datasets,
    evaluate, save_embeddings,
)
from mochi.training import _fetch_episode, _pin_lp_projectors


# ══════════════════════════════════════════════════════════════════════════
# DDP helpers
# ══════════════════════════════════════════════════════════════════════════


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def init_model_with_seed(seed, model_fn):
    """Instantiate deterministically so every rank sees identical initial weights."""
    torch_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    np_state = np.random.get_state()
    py_state = random.getstate()

    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    model = model_fn()

    torch.set_rng_state(torch_state)
    if cuda_state is not None:
        torch.cuda.set_rng_state_all(cuda_state)
    np.random.set_state(np_state)
    random.setstate(py_state)
    return model


def setup_ddp(gpus):
    if "LOCAL_RANK" not in os.environ:
        device = torch.device(f"cuda:{gpus[0]}" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            torch.cuda.set_device(device)
        return 0, 1, device, 0

    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    device = torch.device(f"cuda:{gpus[local_rank]}")
    torch.cuda.set_device(device)
    dist.init_process_group(backend="nccl")
    return rank, world_size, device, local_rank


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


# ══════════════════════════════════════════════════════════════════════════
# DDP training loop
# ══════════════════════════════════════════════════════════════════════════


def train_loop_ddp(model, sampler, lp_handlers, nc_data, gc_data, device, rank,
                   cfg: MochiConfig):
    from contextlib import nullcontext
    accum = cfg.accum_steps

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.meta_lr, weight_decay=cfg.wd)
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
    opt.step(); opt.zero_grad(set_to_none=True)
    opt_steps = cfg.train_steps // accum
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=opt_steps)

    _pin_lp_projectors(lp_handlers)
    model.train()
    running_loss = running_acc = 0.0

    for step in tqdm(range(1, cfg.train_steps + 1)):
        projectors, gc_projs, ep_dev, task_type = _fetch_episode(
            None, sampler, lp_handlers, nc_data, gc_data, device, cfg,
        )

        is_sync = (step % accum == 0)
        sync_ctx = nullcontext() if is_sync else model.no_sync()

        with sync_ctx:
            with torch.cuda.amp.autocast():
                logits, qry_y = model(projectors, ep_dev, task_type, gc_projs)
                loss = F.cross_entropy(logits, qry_y.to(device)) / accum
            scaler.scale(loss).backward()

        if is_sync:
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(opt); scaler.update()
            opt.zero_grad(set_to_none=True)
            sched.step()

        running_loss += loss.item() * accum
        running_acc += (logits.argmax(dim=-1) == qry_y.to(device)).float().mean().item()

        if rank == 0 and step % cfg.log_interval == 0:
            n = cfg.log_interval
            print(f"Step {step}/{cfg.train_steps} | loss={running_loss/n:.4f} "
                  f"| acc={running_acc/n:.4f} | lr={sched.get_last_lr()[0]:.6f} "
                  f"| task={task_type}")
            running_loss = running_acc = 0.0


# ══════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════


def _build_parser():
    import dataclasses
    p = argparse.ArgumentParser(description="Mochi / Mochi++ DDP training")
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

    p.add_argument("--eval_only", action="store_true")
    p.add_argument("--load_model", default=None)
    return p


def main():
    args = _build_parser().parse_args()

    cfg_kwargs = dict(default_params)
    for k, v in vars(args).items():
        if k in MochiConfig.__dataclass_fields__:
            cfg_kwargs[k] = v
    cfg = MochiConfig(**cfg_kwargs)

    gpus = [int(g) for g in cfg.gpus.split(",")]
    rank, world_size, device, local_rank = setup_ddp(gpus)

    seed_everything(cfg.seed + rank)

    if rank == 0:
        print(f"DDP: {world_size} GPUs, gpus={gpus}  |  variant={cfg.model_variant}")

    sampler, lp_handlers, nc_data, gc_data, _ = build_datasets(cfg, repo_root=_REPO_ROOT)

    model = init_model_with_seed(cfg.seed, lambda: Mochi(**cfg.as_dict()).to(device))
    if world_size > 1:
        model = DDP(model, device_ids=[gpus[local_rank]])

    if rank == 0:
        total = sum(p.numel() for p in model.parameters())
        print(f"Model: {total/1e6:.2f}M params")

    if not args.eval_only:
        if rank == 0:
            print(f"\nStarting DDP episodic meta-training ({cfg.train_steps} steps)...")
        train_loop_ddp(model, sampler, lp_handlers, nc_data, gc_data, device, rank, cfg)

        if rank == 0:
            ckpt_dir = os.path.join(_REPO_ROOT, "checkpoints")
            os.makedirs(ckpt_dir, exist_ok=True)
            state = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
            ckpt_path = os.path.join(ckpt_dir, f"{cfg.model_variant}_ddp_s{cfg.seed}.pt")
            torch.save(state, ckpt_path)
            print(f"Checkpoint saved: {ckpt_path}")

    if rank == 0:
        raw = model.module if hasattr(model, "module") else model
        print("\nEpisodic evaluation...")
        evaluate(raw, sampler, lp_handlers, nc_data, gc_data, device, cfg)
        print("\nSaving embeddings...")
        save_embeddings(raw, lp_handlers, nc_data, gc_data, device,
                        os.path.join(_REPO_ROOT, "outputs"), seed=cfg.seed)

    cleanup_ddp()


if __name__ == "__main__":
    main()
