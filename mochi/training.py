"""Episodic meta-training and evaluation.

Two training loops are exposed:

  * :func:`train_mochi`       — one episode per optimizer step (Mochi).
  * :func:`train_mochi_plus`  — 3-episode meta-batching (1 LP + 1 NC + 1 GC)
                                per optimizer step (Mochi++).

:func:`train` dispatches to whichever loop matches ``config.model_variant``.
"""

from __future__ import annotations

import os
import random
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from .config import MochiConfig
from .samplers import MultiTaskEpisodeSampler


# ══════════════════════════════════════════════════════════════════════════
# Reproducibility
# ══════════════════════════════════════════════════════════════════════════


def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ══════════════════════════════════════════════════════════════════════════
# Episode → GPU transfer helpers
# ══════════════════════════════════════════════════════════════════════════


def _pin_lp_projectors(lp_handlers):
    if not torch.cuda.is_available():
        return
    for h in lp_handlers:
        if hasattr(h, "projectors") and not h.projectors.is_pinned():
            h.projectors = h.projectors.pin_memory()


def _sample_ks_q(cfg: MochiConfig):
    """Sample (k_shot, q_query) either as a fixed value or uniform over [min, max]."""
    if cfg.k_shot_min > 0 and cfg.k_shot_min < cfg.k_shot:
        k = np.random.randint(cfg.k_shot_min, cfg.k_shot + 1)
    else:
        k = cfg.k_shot
    if cfg.q_query_min > 0 and cfg.q_query_min < cfg.q_query:
        q = np.random.randint(cfg.q_query_min, cfg.q_query + 1)
    else:
        q = cfg.q_query
    return k, q


def _fetch_episode(task_type, sampler, lp_handlers, nc_data, gc_data,
                   device, cfg: MochiConfig):
    """Sample ``task_type`` (or any if ``task_type`` is ``None``) and move to GPU.

    For NC/LP only the nodes referenced by the episode are transferred, with
    indices remapped to local [0, M) positions.
    """
    k, q = _sample_ks_q(cfg)
    n_way = cfg.max_classes

    gc_graph_projs = None
    proj_cpu = None

    if task_type is None:
        # Sampler picks the task type itself (used by Mochi single-episode loop).
        ep, task_type, ds_idx = sampler.sample(k_shot=k, q_query=q, max_classes=n_way)
    elif task_type == "nc":
        ds_idx = np.random.randint(len(sampler.nc_samplers))
        ncs = sampler.nc_samplers[ds_idx]
        ep = ncs.sample(n_way=min(n_way, len(ncs.classes)), k_shot=k, q_query=q)
    elif task_type == "lp":
        ds_idx = np.random.randint(len(sampler.lp_samplers))
        ep = sampler.lp_samplers[ds_idx].sample(k_shot=k, q_query=q)
    elif task_type == "gc":
        ds_idx = np.random.randint(len(sampler.gc_samplers))
        gcs = sampler.gc_samplers[ds_idx]
        ep = gcs.sample(n_way=min(n_way, len(gcs.classes)), k_shot=k, q_query=q)
    else:
        raise ValueError(f"Unknown task type: {task_type}")

    if task_type == "nc":
        full_proj = nc_data[ds_idx][0]
        all_idx = torch.cat([ep["support_idx"], ep["query_idx"]])
        unique_idx, inverse = torch.unique(all_idx, return_inverse=True)
        proj_cpu = full_proj[unique_idx]
        n_sup = ep["support_idx"].size(0)
        ep["support_idx"] = inverse[:n_sup]
        ep["query_idx"] = inverse[n_sup:]

    elif task_type == "lp":
        full_proj = lp_handlers[ds_idx].projectors
        all_idx = torch.cat([ep["support_src"], ep["support_dst"],
                             ep["query_src"], ep["query_dst"]])
        unique_idx, inverse = torch.unique(all_idx, return_inverse=True)
        proj_cpu = full_proj[unique_idx]
        sizes = [ep["support_src"].size(0), ep["support_dst"].size(0),
                 ep["query_src"].size(0), ep["query_dst"].size(0)]
        ss, sd, qs, qd = inverse.split(sizes)
        ep["support_src"], ep["support_dst"] = ss, sd
        ep["query_src"], ep["query_dst"] = qs, qd

    elif task_type == "gc":
        all_projs = gc_data[ds_idx][0]
        episode_graph_ids = torch.cat([ep["support_idx"], ep["query_idx"]]).unique()
        id_map = {gid.item(): i for i, gid in enumerate(episode_graph_ids)}
        ep["support_idx"] = torch.tensor([id_map[i.item()] for i in ep["support_idx"]], dtype=torch.long)
        ep["query_idx"] = torch.tensor([id_map[i.item()] for i in ep["query_idx"]], dtype=torch.long)
        gc_graph_projs = [all_projs[gid.item()].to(device) for gid in episode_graph_ids]

    projectors = proj_cpu.to(device) if proj_cpu is not None else None
    ep_dev = {k_: (v.to(device) if isinstance(v, torch.Tensor) else v) for k_, v in ep.items()}

    return projectors, gc_graph_projs, ep_dev, task_type


def _run_episode(model, projectors, ep_device, task_type, gc_graph_projs, device):
    """One forward pass under autocast. Returns (loss, accuracy)."""
    with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
        logits, qry_y = model(projectors, ep_device, task_type, gc_graph_projs)
        loss = F.cross_entropy(logits, qry_y.to(device), label_smoothing=0.1)
    acc = (logits.argmax(dim=-1) == qry_y.to(device)).float().mean().item()
    return loss, acc


# ══════════════════════════════════════════════════════════════════════════
# Mochi (single-episode) training
# ══════════════════════════════════════════════════════════════════════════


def train_mochi(model, sampler, lp_handlers, nc_data, gc_data, device,
                cfg: MochiConfig):
    """One episode per optimizer step, sampled from the task distribution."""
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.meta_lr, weight_decay=cfg.wd)
    sched = (torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.train_steps, eta_min=cfg.lr_min)
             if cfg.lr_min > 0 else None)
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    _pin_lp_projectors(lp_handlers)
    model.train()

    running_loss = running_acc = 0.0

    for step in tqdm(range(1, cfg.train_steps + 1)):
        projectors, gc_projs, ep_dev, task_type = _fetch_episode(
            None, sampler, lp_handlers, nc_data, gc_data, device, cfg,
        )

        opt.zero_grad(set_to_none=True)
        try:
            loss, acc = _run_episode(model, projectors, ep_dev, task_type, gc_projs, device)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(opt)
            scaler.update()
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            opt.zero_grad(set_to_none=True)
            continue
        if sched is not None:
            sched.step()

        running_loss += loss.item()
        running_acc += acc

        if step % cfg.log_interval == 0:
            n = cfg.log_interval
            lr = sched.get_last_lr()[0] if sched is not None else cfg.meta_lr
            print(f"Step {step}/{cfg.train_steps} | loss={running_loss/n:.4f} "
                  f"| acc={running_acc/n:.4f} | lr={lr:.6f} | task={task_type}")
            running_loss = running_acc = 0.0
    return model


# ══════════════════════════════════════════════════════════════════════════
# Mochi++ (3-episode meta-batched) training
# ══════════════════════════════════════════════════════════════════════════


def train_mochi_plus(model, sampler, lp_handlers, nc_data, gc_data, device,
                     cfg: MochiConfig):
    """Each optimizer step processes exactly 1 LP + 1 NC + 1 GC episode.

    Per-task losses are averaged via gradient accumulation (each ``loss / 3``).
    Available task types are skipped if the corresponding dataset list is empty.
    """
    TASK_TYPES = [t for t in ("lp", "nc", "gc") if t in sampler.task_types]
    N_TASKS = len(TASK_TYPES)
    if N_TASKS == 0:
        raise ValueError("No task types available — check your dataset lists.")

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.meta_lr, weight_decay=cfg.wd)
    sched = (torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.train_steps, eta_min=cfg.lr_min)
             if cfg.lr_min > 0 else None)
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    _pin_lp_projectors(lp_handlers)
    model.train()

    running = {t: {"loss": 0.0, "acc": 0.0} for t in TASK_TYPES}
    running_meta = 0.0

    for step in tqdm(range(1, cfg.train_steps + 1)):
        opt.zero_grad(set_to_none=True)
        meta_loss = 0.0
        step_ok = True

        for task_type in TASK_TYPES:
            projectors, gc_projs, ep_dev, _ = _fetch_episode(
                task_type, sampler, lp_handlers, nc_data, gc_data, device, cfg,
            )
            try:
                loss, acc = _run_episode(model, projectors, ep_dev, task_type, gc_projs, device)
                scaler.scale(loss / N_TASKS).backward()
                meta_loss += loss.detach().item()
                running[task_type]["loss"] += loss.detach().item()
                running[task_type]["acc"] += acc
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                opt.zero_grad(set_to_none=True)
                step_ok = False
                break

        if not step_ok:
            continue

        scaler.unscale_(opt)
        nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(opt)
        scaler.update()
        if sched is not None:
            sched.step()

        running_meta += meta_loss / N_TASKS

        if step % cfg.log_interval == 0:
            n = cfg.log_interval
            lr = sched.get_last_lr()[0] if sched is not None else cfg.meta_lr
            parts = [f"{t.upper()}(l={running[t]['loss']/n:.4f} a={running[t]['acc']/n:.4f})"
                     for t in TASK_TYPES]
            print(f"Step {step}/{cfg.train_steps} | meta_loss={running_meta/n:.4f} "
                  f"| {' | '.join(parts)} | lr={lr:.6f}")
            running_meta = 0.0
            for t in TASK_TYPES:
                running[t]["loss"] = running[t]["acc"] = 0.0

    return model


# ══════════════════════════════════════════════════════════════════════════
# Public dispatcher
# ══════════════════════════════════════════════════════════════════════════


def train(model, sampler, lp_handlers, nc_data, gc_data, device, cfg: MochiConfig):
    """Dispatch to :func:`train_mochi` or :func:`train_mochi_plus` based on ``cfg.model_variant``."""
    if cfg.model_variant == "mochi++":
        return train_mochi_plus(model, sampler, lp_handlers, nc_data, gc_data, device, cfg)
    if cfg.model_variant == "mochi":
        return train_mochi(model, sampler, lp_handlers, nc_data, gc_data, device, cfg)
    raise ValueError(f"Unknown model_variant: {cfg.model_variant!r}")


# ══════════════════════════════════════════════════════════════════════════
# Evaluation
# ══════════════════════════════════════════════════════════════════════════


@torch.no_grad()
def evaluate(model, sampler, lp_handlers, nc_data, gc_data, device, cfg: MochiConfig,
             n_episodes: int = 100, k_shot: Optional[int] = None,
             q_query: Optional[int] = None):
    """Few-shot evaluation across NC / LP / GC. Prints per-dataset + task averages."""
    model.eval()
    k = k_shot or cfg.k_shot
    q = q_query or cfg.q_query
    results = {}

    # NC --------------------------------------------------------------------
    for ds_idx, (proj, y, name) in enumerate(nc_data):
        accs = []
        ncs = sampler.nc_samplers[ds_idx]
        for _ in range(n_episodes):
            ep = ncs.sample(n_way=min(cfg.max_classes, len(ncs.classes)), k_shot=k, q_query=q)
            all_idx = torch.cat([ep["support_idx"], ep["query_idx"]])
            unique_idx, inverse = torch.unique(all_idx, return_inverse=True)
            proj_sub = proj[unique_idx].to(device)
            n_sup = ep["support_idx"].size(0)
            ep["support_idx"] = inverse[:n_sup]
            ep["query_idx"] = inverse[n_sup:]
            ep_dev = {k_: (v.to(device) if isinstance(v, torch.Tensor) else v) for k_, v in ep.items()}
            logits, qry_y = model(proj_sub, ep_dev, "nc")
            accs.append((logits.argmax(dim=-1) == qry_y.to(device)).float().mean().item())
        results.setdefault("nc", {})[name] = accs

    # LP --------------------------------------------------------------------
    for ds_idx, h in enumerate(lp_handlers):
        accs = []
        lps = sampler.lp_samplers[ds_idx]
        full_proj = h.projectors
        for _ in range(n_episodes):
            ep = lps.sample(k_shot=k, q_query=q)
            all_idx = torch.cat([ep["support_src"], ep["support_dst"],
                                 ep["query_src"], ep["query_dst"]])
            unique_idx, inverse = torch.unique(all_idx, return_inverse=True)
            proj_sub = full_proj[unique_idx].to(device)
            sizes = [ep["support_src"].size(0), ep["support_dst"].size(0),
                     ep["query_src"].size(0), ep["query_dst"].size(0)]
            ss, sd, qs, qd = inverse.split(sizes)
            ep["support_src"], ep["support_dst"] = ss, sd
            ep["query_src"], ep["query_dst"] = qs, qd
            ep_dev = {k_: (v.to(device) if isinstance(v, torch.Tensor) else v) for k_, v in ep.items()}
            logits, qry_y = model(proj_sub, ep_dev, "lp")
            accs.append((logits.argmax(dim=-1) == qry_y.to(device)).float().mean().item())
        results.setdefault("lp", {})[h.data_name] = accs

    # GC --------------------------------------------------------------------
    for ds_idx, (all_projs, y, name) in enumerate(gc_data):
        accs = []
        gcs = sampler.gc_samplers[ds_idx]
        for _ in range(n_episodes):
            ep = gcs.sample(n_way=min(cfg.max_classes, len(gcs.classes)), k_shot=k, q_query=q)
            episode_graph_ids = torch.cat([ep["support_idx"], ep["query_idx"]]).unique()
            id_map = {gid.item(): i for i, gid in enumerate(episode_graph_ids)}
            ep["support_idx"] = torch.tensor([id_map[i.item()] for i in ep["support_idx"]], dtype=torch.long)
            ep["query_idx"] = torch.tensor([id_map[i.item()] for i in ep["query_idx"]], dtype=torch.long)
            gc_projs = [all_projs[gid.item()].to(device) for gid in episode_graph_ids]
            ep_dev = {k_: (v.to(device) if isinstance(v, torch.Tensor) else v) for k_, v in ep.items()}
            logits, qry_y = model(None, ep_dev, "gc", gc_projs)
            accs.append((logits.argmax(dim=-1) == qry_y.to(device)).float().mean().item())
        results.setdefault("gc", {})[name] = accs

    # Summary ---------------------------------------------------------------
    for task_type in ("nc", "lp", "gc"):
        if task_type not in results:
            continue
        print(f"\n  [{task_type.upper()}] {k}-shot eval ({n_episodes} episodes):")
        flat = []
        for name, accs in results[task_type].items():
            print(f"    {name:25s}  acc={np.mean(accs):.4f} ± {np.std(accs):.4f}")
            flat.extend(accs)
        print(f"    {'AVERAGE':25s}  acc={np.mean(flat):.4f} ± {np.std(flat):.4f}")
    return results


# ══════════════════════════════════════════════════════════════════════════
# Embedding export
# ══════════════════════════════════════════════════════════════════════════


def _encode_batched(model, proj, device, batch_size=8192):
    if proj.shape[0] <= batch_size:
        return model.encoder(proj.to(device))
    chunks = []
    for i in range(0, proj.shape[0], batch_size):
        chunks.append(model.encoder(proj[i:i+batch_size].to(device)).cpu())
    return torch.cat(chunks, dim=0).to(device)


@torch.no_grad()
def save_embeddings(model, lp_handlers, nc_data, gc_data, device, save_dir: str,
                    seed: int = 0):
    """Write encoder outputs for every loaded dataset to ``save_dir``."""
    model.eval()
    os.makedirs(save_dir, exist_ok=True)

    for h in lp_handlers:
        Z = _encode_batched(model, h.projectors, device)
        path = os.path.join(save_dir, f"lp_{h.data_name}_{seed}.pt")
        torch.save(Z.cpu(), path)
        print(f"  [LP] {path} ({Z.shape})")

    for proj, y, name in nc_data:
        Z = _encode_batched(model, proj, device)
        path = os.path.join(save_dir, f"nc_{name}_{seed}.pt")
        torch.save({"embeddings": Z.cpu(), "labels": y}, path)
        print(f"  [NC] {path} ({Z.shape}, {len(torch.unique(y))} classes)")

    for all_projs, y, name in gc_data:
        graph_embeds = []
        for gp in all_projs:
            Z_g = model.encoder(gp.to(device))
            graph_embeds.append(Z_g.mean(dim=0).cpu())
        graph_embeds = torch.stack(graph_embeds, dim=0)
        path = os.path.join(save_dir, f"gc_{name}_{seed}.pt")
        torch.save({"embeddings": graph_embeds, "labels": y}, path)
        print(f"  [GC] {path} ({graph_embeds.shape}, {len(torch.unique(y))} classes)")
