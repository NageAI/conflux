"""
CONFLUX Residual Extraction Module

Extracts cross-architecture knowledge residuals (Δ) between the primary
model W and source model(s) M. Handles dimension mismatch via learned
projection layers.

Δ = P · h_M - h_W

where P is a learned linear projection aligning M's hidden dimension to W's.
"""

import torch
import torch.nn as nn
from typing import Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class ResidualInfo:
    """Information about extracted residuals for a layer pair."""
    layer_w_idx: int
    layer_m_idx: int
    source_name: str
    residual_matrix: torch.Tensor
    residual_magnitude: float
    cka_score: float
    transfer_value: float = 0.0
    assigned_rank: int = 0
    assigned_bits: int = 4


class ProjectionLayer(nn.Module):
    """
    Projection to align hidden dimensions between architectures.

    Two modes:
    - procrustes (default): Closed-form orthogonal alignment. No training needed.
      P = UV^T where H_M^T @ H_W = U Σ V^T. Minimizes ||P·H_M - H_W||_F.
    - learned: Trainable linear projection (original approach).
    """

    def __init__(self, d_source: int, d_target: int, bias: bool = False,
                 mode: str = "procrustes"):
        super().__init__()
        self.d_source = d_source
        self.d_target = d_target
        self.mode = mode

        if mode == "learned":
            self.proj = nn.Linear(d_source, d_target, bias=bias)
            nn.init.orthogonal_(self.proj.weight)
        else:
            # Procrustes: P is computed from data, stored as buffer (not parameter)
            self.register_buffer("P", torch.eye(min(d_source, d_target)))
            self._fitted = False

    @torch.no_grad()
    def fit_procrustes(self, h_w: torch.Tensor, h_m: torch.Tensor):
        """Compute optimal orthogonal alignment P via Procrustes.

        Solves: argmin_P ||P·H_M - H_W||_F  subject to P^T P = I

        Closed-form solution:
            M = H_M^T @ H_W
            U, Σ, V^T = SVD(M)
            P = V @ U^T

        Args:
            h_w: [n_samples, d_target] — primary model hidden states
            h_m: [n_samples, d_source] — source model hidden states
        """
        # Truncate to min dimension for cross-dim alignment
        d = min(self.d_source, self.d_target)
        h_w_trunc = h_w[:, :d].float()
        h_m_trunc = h_m[:, :d].float()

        # Center both sets
        h_w_c = h_w_trunc - h_w_trunc.mean(dim=0, keepdim=True)
        h_m_c = h_m_trunc - h_m_trunc.mean(dim=0, keepdim=True)

        # SVD of cross-covariance
        M = h_m_c.T @ h_w_c  # [d, d]
        U, S, Vh = torch.linalg.svd(M, full_matrices=False)

        # Optimal orthogonal P = V @ U^T
        self.P = (Vh.T @ U.T).to(self.P.device)
        self._fitted = True

        # Alignment quality: ||P·H_M - H_W|| / ||H_W||
        aligned = h_m_c @ self.P.T
        error = torch.norm(aligned - h_w_c).item()
        baseline = torch.norm(h_w_c).item()
        quality = 1.0 - error / (baseline + 1e-8)

        logger.info(f"Procrustes fit: alignment quality={quality:.4f} (1.0=perfect)")

    @property
    def param_count(self) -> int:
        if self.mode == "learned":
            return sum(p.numel() for p in self.parameters())
        return 0  # Procrustes has no trainable params

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "learned":
            return self.proj(x)
        # Procrustes mode: truncate + rotate
        d = self.P.shape[0]
        x_trunc = x[..., :d].float()
        return (x_trunc @ self.P.T).to(x.dtype)


class ResidualExtractor:
    """
    Extracts knowledge residuals between primary and source models.

    The residual Δ = P(h_M) - h_W represents what model M knows that
    model W doesn't. This is the core signal that drives CONFLUX adaptation.

    Example:
        extractor = ResidualExtractor(d_primary=4096, d_source=5120)
        residual = extractor.compute_residual(h_w, h_m)
    """

    def __init__(
        self,
        d_primary: int,
        d_source: int,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        """
        Args:
            d_primary: Hidden dimension of primary model W
            d_source: Hidden dimension of source model M
            device: Device for projection layer
            dtype: Dtype for computation
        """
        self.d_primary = d_primary
        self.d_source = d_source
        self.device = device
        self.dtype = dtype

        self.needs_projection = d_primary != d_source
        self.projection: Optional[ProjectionLayer] = None

        if self.needs_projection:
            self.projection = ProjectionLayer(d_source, d_primary).to(device)
            logger.info(
                f"Projection layer created: {d_source} → {d_primary} "
                f"({self.projection.param_count:,} params)"
            )
        else:
            logger.info(f"Same dimensions ({d_primary}), no projection needed")

    def align_dimensions(self, h_m: torch.Tensor) -> torch.Tensor:
        """
        Project source hidden states to primary model's dimension space.

        Args:
            h_m: [batch, d_source] source model hidden states

        Returns:
            [batch, d_primary] aligned hidden states
        """
        if not self.needs_projection:
            return h_m

        return self.projection(h_m.to(self.device).to(self.dtype))

    @torch.no_grad()
    def compute_residual(
        self,
        h_w: torch.Tensor,
        h_m: torch.Tensor,
        normalize: bool = True,
    ) -> torch.Tensor:
        """
        Compute knowledge residual: Δ = P(h_M) - h_W

        Args:
            h_w: [batch, d_primary] primary model hidden states
            h_m: [batch, d_source] source model hidden states
            normalize: Whether to L2-normalize before computing residual

        Returns:
            [batch, d_primary] residual tensor
        """
        h_m_aligned = self.align_dimensions(h_m)
        h_w = h_w.to(self.device).to(self.dtype)

        if normalize:
            h_w = F.normalize(h_w, p=2, dim=-1)
            h_m_aligned = F.normalize(h_m_aligned, p=2, dim=-1)

        return h_m_aligned - h_w

    @torch.no_grad()
    def compute_residual_magnitude(
        self,
        h_w: torch.Tensor,
        h_m: torch.Tensor,
    ) -> float:
        """
        Compute scalar magnitude of residual: ||Δ||_F

        Used for rank allocation — larger residual = more transfer potential.
        """
        residual = self.compute_residual(h_w, h_m)
        return torch.norm(residual, p="fro").item()

    @torch.no_grad()
    def compute_residual_matrix(
        self,
        h_w: torch.Tensor,
        h_m: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the full residual matrix R for SVD decomposition.

        R = P(H_M) - H_W where H is [n_samples, d]

        This matrix is passed to SVD to extract the principal directions
        of the knowledge gap.

        Returns:
            [n_samples, d_primary] residual matrix
        """
        return self.compute_residual(h_w, h_m, normalize=False)

    def extract_all_layers(
        self,
        hidden_states_w: list[torch.Tensor],
        hidden_states_m: list[torch.Tensor],
        layer_matching: list[tuple[int, int, float]],
        source_name: str = "source",
    ) -> list[ResidualInfo]:
        """
        Extract residuals for all matched layer pairs.

        Args:
            hidden_states_w: Per-layer hidden states from W
            hidden_states_m: Per-layer hidden states from M
            layer_matching: List of (w_idx, m_idx, cka_score) from CKA matching
            source_name: Name of source model for tracking

        Returns:
            List of ResidualInfo for each layer pair
        """
        residuals = []

        for w_idx, m_idx, cka_score in layer_matching:
            h_w = hidden_states_w[w_idx]
            h_m = hidden_states_m[m_idx]

            residual = self.compute_residual_matrix(h_w, h_m)
            magnitude = torch.norm(residual, p="fro").item()

            residuals.append(ResidualInfo(
                w_layer=w_idx,
                m_layer=m_idx,
                source_name=source_name,
                residual=residual,
                magnitude=magnitude,
                cka_score=cka_score,
            ))

            logger.debug(
                f"Layer W[{w_idx}] ↔ M[{m_idx}]: "
                f"CKA={cka_score:.4f}, |Δ|={magnitude:.4f}"
            )

        return residuals


# Convenience import
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════
#  Compatibility aliases (trainer uses these attribute names)
# ═══════════════════════════════════════════════════════════


class ProjectionBank(nn.Module):
    """Collection of projection layers for multiple source models.

    Manages one projection per source, handling different hidden dims.
    """

    def __init__(self, d_primary: int, source_dims: dict[str, int]):
        super().__init__()
        self.projections = nn.ModuleDict()
        self.d_primary = d_primary

        for alias, d_src in source_dims.items():
            if d_src != d_primary:
                self.projections[alias] = ProjectionLayer(d_src, d_primary)
                logger.info(f"Projection [{alias}]: {d_src} → {d_primary}")
            else:
                logger.info(f"Projection [{alias}]: identity (same dim {d_src})")

    def forward(self, x: torch.Tensor, source_alias: str) -> torch.Tensor:
        if source_alias in self.projections:
            return self.projections[source_alias](x)
        return x


# ═══════════════════════════════════════════════════════════
#  Function API (used by ConfluxTrainer)
# ═══════════════════════════════════════════════════════════

@torch.no_grad()
def extract_hidden_states(
    model,
    tokenizer,
    texts: list[str],
    batch_size: int = 32,
    max_length: int = 512,
    device: str = "cuda",
) -> list[torch.Tensor]:
    """Extract hidden states from all layers for a batch of texts.

    Args:
        model: HuggingFace model (frozen, quantized)
        tokenizer: Corresponding tokenizer
        texts: Input texts
        batch_size: Batch size for forward passes
        max_length: Maximum sequence length
        device: Computation device

    Returns:
        List of tensors per layer, each shape (n_samples, hidden_dim).
        Hidden states are mean-pooled across sequence positions.
    """
    model.eval()
    all_hidden = None

    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        inputs = tokenizer(
            batch, return_tensors="pt", padding=True,
            truncation=True, max_length=max_length,
        ).to(device)

        outputs = model(**inputs, output_hidden_states=True, return_dict=True)
        mask = inputs["attention_mask"].unsqueeze(-1).float()

        batch_pooled = []
        for hs in outputs.hidden_states:
            pooled = (hs * mask).sum(dim=1) / mask.sum(dim=1)
            batch_pooled.append(pooled.cpu())

        if all_hidden is None:
            all_hidden = [[] for _ in range(len(batch_pooled))]
        for i, p in enumerate(batch_pooled):
            all_hidden[i].append(p)

    return [torch.cat(layer, dim=0) for layer in all_hidden]


def compute_residuals(
    hidden_states_w: list[torch.Tensor],
    hidden_states_m: list[torch.Tensor],
    layer_matches: list[tuple[int, int, float]],
    projection: Optional[ProjectionLayer] = None,
) -> list[ResidualInfo]:
    """Compute cross-architecture residuals for matched layer pairs.

    Args:
        hidden_states_w: Per-layer hidden states from W.
        hidden_states_m: Per-layer hidden states from M.
        layer_matches: [(w_idx, m_idx, cka_score), ...]
        projection: Optional projection for dimension alignment.

    Returns:
        List of ResidualInfo with residuals and magnitudes.
    """
    d_w = hidden_states_w[0].shape[-1]
    d_m = hidden_states_m[0].shape[-1]

    extractor = ResidualExtractor(d_primary=d_w, d_source=d_m)
    infos = extractor.extract_all_layers(
        hidden_states_w, hidden_states_m, layer_matches,
    )

    # Normalize magnitudes
    if infos:
        max_mag = max(r.residual_magnitude for r in infos)
        for r in infos:
            object.__setattr__(r, 'residual_magnitude', r.residual_magnitude / (max_mag + 1e-8))

    return infos


def aggregate_multi_source_residuals(
    residuals_per_source: dict[str, list[ResidualInfo]],
    strategy: str = "max",
) -> list[ResidualInfo]:
    """Aggregate residuals from multiple source models.

    Args:
        residuals_per_source: {source_alias: [ResidualInfo, ...]}
        strategy: "max" (largest residual), "mean", or "weighted"

    Returns:
        Single aggregated residual list.
    """
    all_layers = set()
    for infos in residuals_per_source.values():
        for r in infos:
            all_layers.add(r.layer_w_idx)

    aggregated = []
    for w_idx in sorted(all_layers):
        candidates = []
        for infos in residuals_per_source.values():
            for r in infos:
                if r.layer_w_idx == w_idx:
                    candidates.append(r)

        if not candidates:
            continue

        if strategy == "max":
            best = max(candidates, key=lambda r: r.residual_magnitude)
        elif strategy == "mean":
            best = max(candidates, key=lambda r: r.residual_magnitude)
            avg_mag = sum(r.residual_magnitude for r in candidates) / len(candidates)
            object.__setattr__(best, 'magnitude', avg_mag)
        else:
            best = max(candidates, key=lambda r: r.residual_magnitude)

        aggregated.append(best)

    return aggregated
