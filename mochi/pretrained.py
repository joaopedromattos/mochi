"""Helpers for loading pretrained Mochi / Mochi++ weights from Hugging Face.

Default repo: ``jrm28/mochi`` — contains per-seed checkpoints at
``checkpoints/mochi++_s{0,1,2}.pt``.

Example::

    from mochi import Mochi, default_params, load_pretrained
    model = Mochi(**default_params)
    load_pretrained(model, seed=2)      # downloads + loads weights in place
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


HF_REPO_ID = "jrm28/mochi"


def checkpoint_filename(variant: str = "mochi++", seed: int = 2) -> str:
    """Return the in-repo path of a pretrained checkpoint."""
    return f"checkpoints/{variant}_s{seed}.pt"


def download_checkpoint(variant: str = "mochi++", seed: int = 2,
                        repo_id: str = HF_REPO_ID,
                        cache_dir: Optional[str] = None) -> str:
    """Fetch a pretrained checkpoint from Hugging Face and return its local path.

    Uses ``huggingface_hub.hf_hub_download`` under the hood, which caches in
    ``~/.cache/huggingface`` (or ``cache_dir`` if given).
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:
        raise ImportError(
            "huggingface_hub is required to download pretrained weights. "
            "Install with `pip install huggingface_hub`."
        ) from e

    return hf_hub_download(
        repo_id=repo_id,
        filename=checkpoint_filename(variant, seed),
        cache_dir=cache_dir,
    )


def load_pretrained(model: nn.Module, *, variant: str = "mochi++", seed: int = 2,
                    repo_id: str = HF_REPO_ID,
                    cache_dir: Optional[str] = None,
                    map_location: str = "cpu",
                    strict: bool = True) -> nn.Module:
    """Download pretrained weights from Hugging Face and load them into ``model``.

    Args:
        model:        a :class:`MochiModel` instance (e.g. from ``Mochi(**default_params)``).
        variant:      ``"mochi"`` or ``"mochi++"`` (default).
        seed:         which seed's checkpoint to use (0, 1, or 2).
        repo_id:      HF model-repo id. Defaults to :data:`HF_REPO_ID`.
        cache_dir:    override the HF cache location.
        map_location: passed to ``torch.load``.
        strict:       passed to ``load_state_dict``.

    Returns:
        the same ``model`` (for chaining).
    """
    path = download_checkpoint(variant=variant, seed=seed,
                               repo_id=repo_id, cache_dir=cache_dir)
    state_dict = torch.load(path, map_location=map_location, weights_only=True)
    model.load_state_dict(state_dict, strict=strict)
    return model
