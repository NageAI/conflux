"""
CONFLUX Offline Cache Module (Module 3 Extension)

Caches source model hidden states to disk, allowing the source model
to be fully unloaded during training. This reduces VRAM to standard
QLoRA levels while still benefiting from cross-architecture residuals.

Flow:
    1. Load source model M → extract hidden states → save to disk → unload M
    2. During training, load cached states per batch (disk I/O, no GPU memory)
    3. Compute guidance loss against cached targets

This is the key to making CONFLUX practical on single-GPU setups.
"""

import torch
import numpy as np
from pathlib import Path
from typing import Optional
import json
import logging

logger = logging.getLogger(__name__)


class OfflineCache:
    """Disk-backed cache for source model hidden states.

    Stores hidden states as memory-mapped numpy arrays for
    zero-copy loading during training.

    Args:
        cache_dir: Directory for cached files.
        source_alias: Name of the source model.
    """

    def __init__(self, cache_dir: str, source_alias: str):
        self.cache_dir = Path(cache_dir) / source_alias
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.source_alias = source_alias
        self._metadata = {}

    def save_hidden_states(
        self,
        hidden_states: list[torch.Tensor],
        layer_indices: Optional[list[int]] = None,
    ):
        """Save per-layer hidden states to disk as memory-mapped arrays.

        Args:
            hidden_states: List of tensors, one per layer.
            layer_indices: Which layer indices these correspond to.
                          If None, uses 0, 1, 2, ...
        """
        if layer_indices is None:
            layer_indices = list(range(len(hidden_states)))

        for idx, hs in zip(layer_indices, hidden_states):
            arr = hs.cpu().float().numpy()
            path = self.cache_dir / f"layer_{idx}.npy"
            np.save(str(path), arr)
            logger.debug(f"Cached layer {idx}: shape={arr.shape}, size={arr.nbytes / 1e6:.1f}MB")

        self._metadata = {
            "source_alias": self.source_alias,
            "num_layers": len(hidden_states),
            "layer_indices": layer_indices,
            "shapes": {str(idx): list(hs.shape) for idx, hs in zip(layer_indices, hidden_states)},
            "dtype": "float32",
        }

        meta_path = self.cache_dir / "metadata.json"
        with open(meta_path, "w") as f:
            json.dump(self._metadata, f, indent=2)

        total_size = sum(hs.numel() * 4 for hs in hidden_states)
        logger.info(
            f"Cached {len(hidden_states)} layers for [{self.source_alias}]: "
            f"{total_size / 1e9:.2f} GB total"
        )

    def load_layer(self, layer_idx: int, device: str = "cpu") -> torch.Tensor:
        """Load a single layer's hidden states from cache.

        Uses memory mapping for efficient partial loading.

        Args:
            layer_idx: Which layer to load.
            device: Target device for the tensor.

        Returns:
            Hidden state tensor for the requested layer.
        """
        path = self.cache_dir / f"layer_{layer_idx}.npy"
        if not path.exists():
            raise FileNotFoundError(f"No cached data for layer {layer_idx} at {path}")

        arr = np.load(str(path), mmap_mode="r")
        return torch.from_numpy(np.array(arr)).to(device)

    def load_batch(
        self,
        layer_idx: int,
        start: int,
        end: int,
        device: str = "cpu",
    ) -> torch.Tensor:
        """Load a slice of samples from a cached layer.

        Memory-efficient: only reads the requested rows.

        Args:
            layer_idx: Which layer to load from.
            start: Start sample index.
            end: End sample index (exclusive).
            device: Target device.

        Returns:
            Tensor of shape (end - start, hidden_dim).
        """
        path = self.cache_dir / f"layer_{layer_idx}.npy"
        arr = np.load(str(path), mmap_mode="r")
        batch = np.array(arr[start:end])
        return torch.from_numpy(batch).to(device)

    def get_metadata(self) -> dict:
        """Load cache metadata."""
        meta_path = self.cache_dir / "metadata.json"
        if meta_path.exists():
            with open(meta_path) as f:
                return json.load(f)
        return self._metadata

    def exists(self) -> bool:
        """Check if cache exists and is valid."""
        meta_path = self.cache_dir / "metadata.json"
        return meta_path.exists()

    def clear(self):
        """Delete all cached files."""
        import shutil
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
            logger.info(f"Cleared cache for [{self.source_alias}]")


class CachedGuidanceProvider:
    """Provides cached target hidden states during training.

    Replaces live source model inference with disk-backed lookups.
    Integrates with ConfluxLoss for the guidance component.

    Args:
        caches: Dict mapping source_alias to OfflineCache instances.
        layer_matches: Dict mapping source_alias to list of (w_idx, m_idx) pairs.
    """

    def __init__(
        self,
        caches: dict[str, OfflineCache],
        layer_matches: dict[str, list[tuple[int, int]]],
    ):
        self.caches = caches
        self.layer_matches = layer_matches

    def get_targets(
        self,
        sample_indices: list[int],
        device: str = "cuda",
    ) -> dict[str, dict[int, torch.Tensor]]:
        """Get target hidden states for a batch of samples.

        Args:
            sample_indices: Which calibration samples are in this batch.
            device: Target device.

        Returns:
            Nested dict: {source_alias: {w_layer_idx: target_tensor}}
        """
        targets = {}

        for alias, cache in self.caches.items():
            matches = self.layer_matches.get(alias, [])
            targets[alias] = {}

            for w_idx, m_idx in matches:
                start = min(sample_indices)
                end = max(sample_indices) + 1
                cached = cache.load_batch(m_idx, start, end, device=device)
                targets[alias][w_idx] = cached

        return targets
