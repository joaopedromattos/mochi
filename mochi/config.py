"""Configuration for Mochi / Mochi++.

``MochiConfig`` holds all hyperparameters in a dataclass. ``default_params``
is a plain dict of the recommended values from the paper (the Mochi++ setup)
so that ``Mochi(**default_params)`` gives the published model out of the box.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Optional


@dataclass
class MochiConfig:
    # Variant selector: "mochi" (single-episode) or "mochi++" (meta-batched)
    model_variant: str = "mochi++"

    # Encoder (SVD projectors + GAMLP) -------------------------------------
    latdim: int = 512               # latent dimensionality
    gnn_layer: int = 3              # number of propagation hops (TopoEncoder)
    drop_rate: float = 0.0          # dropout inside GAMLP
    niter: int = 2                  # SVD iterations
    proj_method: str = "both"       # SVD over adj + feats

    # Ridge readout --------------------------------------------------------
    ridge_lambda: float = 10.0

    # Episode sampling -----------------------------------------------------
    k_shot: int = 96                # max support samples / class
    k_shot_min: int = 0             # 0 → fixed k_shot; else uniform [min, k_shot]
    q_query: int = 192              # max query samples / class
    q_query_min: int = 0
    max_classes: int = 64
    task_weights: str = "1:1:1"     # only used by the Mochi variant

    # Optimization ---------------------------------------------------------
    meta_lr: float = 3e-4
    lr_min: float = 1e-5            # cosine schedule eta_min (0 disables)
    wd: float = 1e-4
    train_steps: int = 12991
    log_interval: int = 10
    grad_clip: float = 1.0

    # Misc -----------------------------------------------------------------
    seed: int = 2
    gpu: str = "0"
    cache_dir: Optional[str] = None  # defaults to <repo>/cache/projectors
    data_root: Optional[str] = None  # PyG datasets root
    lp_data_root: Optional[str] = None  # AnyGraph-format LP pickles root
    cstag_root: Optional[str] = None    # CS-TAG CSV datasets root

    # Dataset lists --------------------------------------------------------
    dataset_setting: str = "link1"
    nc_datasets: List[str] = field(default_factory=lambda: [
        "citeseer", "pubmed", "physics", "computers",
    ])
    gc_datasets: List[str] = field(default_factory=lambda: [
        "DD", "ENZYMES", "REDDIT-MULTI-5K",
    ])

    # DDP ------------------------------------------------------------------
    gpus: str = "0"
    accum_steps: int = 4

    def as_dict(self):
        return asdict(self)


# Recommended hyperparameters from the paper (Mochi++ setup). ``Mochi(**default_params)``
# reproduces the main-table configuration.
default_params: dict = {
    "model_variant": "mochi++",
    "dataset_setting": "link1",
    "nc_datasets": ["citeseer", "pubmed", "physics", "computers"],
    "gc_datasets": ["DD", "ENZYMES", "REDDIT-MULTI-5K"],
    "ridge_lambda": 10.0,
    "train_steps": 12991,
    "k_shot": 96,
    "k_shot_min": 0,
    "q_query": 192,
    "q_query_min": 0,
    "seed": 2,
    "gpu": "2",
}


# LP dataset groups (AnyGraph's split convention) ---------------------------
# link1 = training datasets for the main paper run
# link2 = held-out datasets for evaluation
LP_DATASET_GROUPS = {
    "link1": [
        "products_tech", "yelp2018", "yelp_textfeat", "products_home",
        "steam_textfeat", "amazon_textfeat", "amazon-book", "citation-2019",
        "citation-classic", "pubmed", "citeseer", "ppa", "p2p-Gnutella06",
        "soc-Epinions1", "email-Enron",
    ],
    "link2": [
        "Photo", "Goodreads", "Fitness", "ml1m", "ml10m", "gowalla", "arxiv",
        "arxiv-ta", "cora", "CS", "collab", "proteins_spec0", "proteins_spec1",
        "proteins_spec2", "proteins_spec3", "ddi", "web-Stanford", "roadNet-PA",
    ],
    "smoke": ["cora"],
}
