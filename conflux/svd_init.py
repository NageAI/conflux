"""
CONFLUX Residual-SVD Initialization (Module 2)

Initializes LoRA A/B matrices from the SVD of cross-architecture residuals.

Paper formulation:
    R = P·h_M - h_W                          (knowledge residual)
    C = (1/n) · R^T · R                      (covariance in output space)
    C = U · Σ · U^T                           (eigendecomposition)
    A_init = √Σ_r · U_r^T                    (rank, d_model)
    B_init = U_r · √Σ_r                      (d_model, rank)

Result: B·A ≈ rank-r approximation of the residual covariance.
QLoRA zero-init captures 0%. This captures explained_variance_ratio%.
"""

import torch
from typing import Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class SVDInitResult:
    """Result of Residual-SVD initialization for one layer."""
    layer_idx: int
    rank: int
    A: torch.Tensor
    B: torch.Tensor
    singular_values: torch.Tensor
    explained_variance_ratio: float
    residual_norm: float


def residual_svd_init(
    residual_matrix: torch.Tensor,
    rank: int,
    scaling: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Core SVD initialization from a residual matrix.

    Args:
        residual_matrix: (n_samples, d_model) — the knowledge residual R
        rank: target LoRA rank
        scaling: scale factor (< 1.0 for conservative start)

    Returns:
        (A, B, singular_values, explained_variance_ratio)
        A: (rank, d_model)
        B: (d_model, rank)
    """
    R = residual_matrix.float()
    n, d = R.shape

    if n >= d:
        C = (R.T @ R) / n
        eigenvalues, eigenvectors = torch.linalg.eigh(C)
        eigenvalues = eigenvalues.flip(0)
        eigenvectors = eigenvectors.flip(1)
    else:
        U, S, Vh = torch.linalg.svd(R, full_matrices=False)
        eigenvalues = (S ** 2) / n
        eigenvectors = Vh.T

    rank = min(rank, len(eigenvalues))
    eigenvalues_r = eigenvalues[:rank].clamp(min=1e-8)
    U_r = eigenvectors[:, :rank]

    total_var = eigenvalues.clamp(min=0).sum().item()
    explained_var = eigenvalues_r.sum().item()
    evr = explained_var / (total_var + 1e-10)

    sqrt_lambda = torch.sqrt(eigenvalues_r) * scaling
    B = U_r * sqrt_lambda.unsqueeze(0)
    A = (U_r * sqrt_lambda.unsqueeze(0)).T

    return A, B, eigenvalues_r, evr


def batch_svd_init(
    residual_infos: list,
    min_explained_variance: float = 0.3,
) -> list[SVDInitResult]:
    """Compute SVD initialization for all matched layer pairs.

    Args:
        residual_infos: list with .layer_w_idx, .assigned_rank, .residual_matrix
        min_explained_variance: warning threshold

    Returns:
        List of SVDInitResult per initialized layer.
    """
    results = []

    for info in residual_infos:
        if not hasattr(info, 'residual_matrix') or info.residual_matrix is None:
            continue
        if not hasattr(info, 'assigned_rank') or info.assigned_rank == 0:
            continue

        R = info.residual_matrix
        rank = info.assigned_rank
        layer_idx = info.layer_w_idx

        A, B, svals, evr = residual_svd_init(R, rank)
        residual_norm = torch.norm(R, p="fro").item()

        if evr < min_explained_variance:
            logger.warning(
                f"Layer {layer_idx}: low EVR {evr:.4f} (rank={rank})."
            )

        results.append(SVDInitResult(
            layer_idx=layer_idx,
            rank=rank,
            A=A,
            B=B,
            singular_values=svals,
            explained_variance_ratio=evr,
            residual_norm=residual_norm,
        ))

        logger.info(f"SVD init layer {layer_idx}: rank={rank}, EVR={evr:.4f}")

    avg_evr = sum(r.explained_variance_ratio for r in results) / max(len(results), 1)
    logger.info(f"Initialized {len(results)} layers. Mean EVR: {avg_evr:.4f}")
    return results


def apply_svd_init_to_peft(
    peft_model,
    svd_results: list[SVDInitResult],
    target_modules: Optional[list[str]] = None,
):
    """Apply SVD-initialized weights to a PEFT/LoRA model.

    Args:
        peft_model: HuggingFace PEFT model with LoRA adapters
        svd_results: from batch_svd_init()
        target_modules: filter to specific modules (e.g. ["q_proj"])
    """
    svd_by_layer = {r.layer_idx: r for r in svd_results}
    initialized = 0

    for name, param in peft_model.named_parameters():
        if "lora_A" not in name and "lora_B" not in name:
            continue

        layer_idx = _extract_layer_index(name)
        if layer_idx is None or layer_idx not in svd_by_layer:
            continue

        if target_modules and not any(tm in name for tm in target_modules):
            continue

        svd = svd_by_layer[layer_idx]

        if "lora_A" in name and param.shape[0] == svd.rank:
            d = min(param.shape[1], svd.A.shape[1])
            param.data.zero_()
            param.data[:, :d] = svd.A[:, :d].to(param.device, param.dtype)
            initialized += 1
        elif "lora_B" in name and param.shape[1] == svd.rank:
            d = min(param.shape[0], svd.B.shape[0])
            param.data.zero_()
            param.data[:d, :] = svd.B[:d, :].to(param.device, param.dtype)
            initialized += 1

    logger.info(f"Applied SVD init to {initialized} LoRA parameters")


def _extract_layer_index(param_name: str) -> Optional[int]:
    """Extract layer index from 'model.layers.15.self_attn.q_proj.lora_A'."""
    parts = param_name.split(".")
    for i, part in enumerate(parts):
        if part == "layers" and i + 1 < len(parts):
            try:
                return int(parts[i + 1])
            except ValueError:
                pass
    return None


class ResidualSVDInitializer:
    """Class wrapper around the function API for backward compatibility.

    Example:
        init = ResidualSVDInitializer(scaling=0.8)
        results = init.initialize_all(residual_infos, ranks={0: 16, 1: 32})
    """

    def __init__(self, scaling: float = 1.0):
        self.scaling = scaling

    def initialize(self, residual_matrix: torch.Tensor, rank: int, layer_idx: int = 0) -> SVDInitResult:
        A, B, svals, evr = residual_svd_init(residual_matrix, rank, scaling=self.scaling)
        return SVDInitResult(
            layer_idx=layer_idx, rank=rank, A=A, B=B,
            singular_values=svals, explained_variance_ratio=evr,
            residual_norm=torch.norm(residual_matrix, p="fro").item(),
        )

    def initialize_all(self, residual_infos: list, ranks: dict[int, int]) -> dict[int, SVDInitResult]:
        results = {}
        for info in residual_infos:
            layer = getattr(info, 'w_layer', getattr(info, 'layer_w_idx', 0))
            rank = ranks.get(layer, 0)
            if rank == 0:
                continue
            residual = getattr(info, 'residual', getattr(info, 'residual_matrix', None))
            if residual is None:
                continue
            results[layer] = self.initialize(residual, rank, layer_idx=layer)
        return results
