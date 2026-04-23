#!/usr/bin/env bash
# Example commands reproducing the Mochi++ paper setup.
#
# Training datasets: NC = {citeseer, pubmed, physics, computers},
#                    GC = {DD, ENZYMES, REDDIT-MULTI-5K},
#                    LP = LP_DATASET_GROUPS["link1"] (15 datasets).
#
# Evaluation datasets (held out): link2 group, plus the GC/NC lists below.

# ── Mochi++ (default, paper setup) — three seeds ────────────────────────────
python train.py --seed 0 --gpu 0
python train.py --seed 1 --gpu 1
python train.py --seed 2 --gpu 2

# ── Mochi (ablation, single-episode training) ───────────────────────────────
python train.py --model_variant mochi --seed 0 --gpu 0

# ── Evaluation on held-out datasets ─────────────────────────────────────────
python train.py --eval_only --load_model checkpoints/mochi++_s2.pt \
    --dataset_setting link2 \
    --nc_datasets cora cs photo arxiv Fitness \
    --gc_datasets MUTAG PROTEINS DD ENZYMES NCI1 IMDB-BINARY COLLAB REDDIT-MULTI-5K \
    --gpu 2 --seed 2

# ── Multi-GPU DDP training ──────────────────────────────────────────────────
# torchrun --nproc_per_node=4 train_ddp.py --gpus 0,1,2,3
