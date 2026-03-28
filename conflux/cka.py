"""
CONFLUX CKA Module — Centered Kernel Alignment

Computes representation similarity between layers of different architectures.
Used for:
  1. Cross-architecture layer matching (which W layer ≈ which M layer)
  2. Transfer value scoring (high CKA = similar = low transfer need)

Supports linear and RBF kernels. Operates on batched hidden states.
"""

import torch
import torch.nn.functional as F
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class CKAComputer:
    """
    Centered Kernel Alignment computation for cross-architecture comparison.

    Given hidden states from two models on the same inputs, computes
    a similarity matrix showing which layers are functionally equivalent.

    Example:
        cka = CKAComputer(kernel="linear")
        # h_w: list of tensors [batch, hidden_dim_w] per layer
        # h_m: list of tensors [batch, hidden_dim_m] per layer
        sim_matrix = cka.compute_similarity_matrix(h_w, h_m)
        matching = cka.find_best_matching(sim_matrix)
    """

    def __init__(self, kernel: str = "linear", rbf_sigma: Optional[float] = None):
        """
        Args:
            kernel: "linear" or "rbf". Linear is faster and sufficient for most cases.
            rbf_sigma: RBF bandwidth. If None, uses median heuristic.
        """
        assert kernel in ("linear", "rbf"), f"Unknown kernel: {kernel}"
        self.kernel = kernel
        self.rbf_sigma = rbf_sigma

    def _center_gram(self, K: torch.Tensor) -> torch.Tensor:
        """Center a Gram matrix: H @ K @ H where H = I - 1/n * 11^T"""
        n = K.shape[0]
        H = torch.eye(n, device=K.device, dtype=K.dtype) - 1.0 / n
        return H @ K @ H

    def _gram_linear(self, X: torch.Tensor) -> torch.Tensor:
        """Compute linear Gram matrix: K = X @ X^T"""
        return X @ X.T

    def _gram_rbf(self, X: torch.Tensor, sigma: Optional[float] = None) -> torch.Tensor:
        """Compute RBF Gram matrix: K_ij = exp(-||x_i - x_j||^2 / (2*sigma^2))"""
        sq_dists = torch.cdist(X, X, p=2.0).pow(2)
        if sigma is None:
            sigma = torch.median(sq_dists[sq_dists > 0]).sqrt().item()
            if sigma == 0:
                sigma = 1.0
        return torch.exp(-sq_dists / (2.0 * sigma ** 2))

    def _gram(self, X: torch.Tensor) -> torch.Tensor:
        """Compute Gram matrix based on selected kernel."""
        if self.kernel == "linear":
            return self._gram_linear(X)
        else:
            return self._gram_rbf(X, self.rbf_sigma)

    def _hsic(self, K: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
        """Hilbert-Schmidt Independence Criterion (biased estimator)."""
        Kc = self._center_gram(K)
        Lc = self._center_gram(L)
        n = K.shape[0]
        return (Kc * Lc).sum() / ((n - 1) ** 2)

    @torch.no_grad()
    def compute_cka(
        self,
        X: torch.Tensor,
        Y: torch.Tensor,
    ) -> float:
        """
        Compute CKA between two sets of representations.

        Args:
            X: [n_samples, dim_x] — hidden states from model W at some layer
            Y: [n_samples, dim_y] — hidden states from model M at some layer

        Returns:
            CKA score in [0, 1]. Higher = more similar representations.
        """
        assert X.shape[0] == Y.shape[0], "Same number of samples required"

        X = X.float()
        Y = Y.float()

        # Normalize for numerical stability
        X = X - X.mean(dim=0, keepdim=True)
        Y = Y - Y.mean(dim=0, keepdim=True)

        K = self._gram(X)
        L = self._gram(Y)

        hsic_xy = self._hsic(K, L)
        hsic_xx = self._hsic(K, K)
        hsic_yy = self._hsic(L, L)

        denom = torch.sqrt(hsic_xx * hsic_yy)
        if denom < 1e-10:
            return 0.0

        cka = (hsic_xy / denom).item()
        return max(0.0, min(1.0, cka))

    @torch.no_grad()
    def compute_similarity_matrix(
        self,
        hidden_states_w: list[torch.Tensor],
        hidden_states_m: list[torch.Tensor],
        progress_callback=None,
    ) -> torch.Tensor:
        """
        Compute full CKA similarity matrix between all layer pairs.

        Args:
            hidden_states_w: List of [n_samples, dim_w] tensors, one per W layer
            hidden_states_m: List of [n_samples, dim_m] tensors, one per M layer
            progress_callback: Optional callable(i, j, total) for progress reporting

        Returns:
            [n_layers_w, n_layers_m] similarity matrix
        """
        n_w = len(hidden_states_w)
        n_m = len(hidden_states_m)
        total = n_w * n_m

        sim_matrix = torch.zeros(n_w, n_m)

        logger.info(f"Computing CKA matrix: {n_w} x {n_m} = {total} pairs")

        for i in range(n_w):
            for j in range(n_m):
                sim_matrix[i, j] = self.compute_cka(
                    hidden_states_w[i],
                    hidden_states_m[j],
                )
                if progress_callback:
                    progress_callback(i, j, total)

        return sim_matrix

    def find_best_matching(
        self,
        sim_matrix: torch.Tensor,
        method: str = "greedy",
    ) -> list[tuple[int, int, float]]:
        """
        Find optimal layer matching from CKA similarity matrix.

        Args:
            sim_matrix: [n_w, n_m] CKA scores
            method: "greedy" (fast) or "hungarian" (optimal)

        Returns:
            List of (w_layer, m_layer, cka_score) tuples, sorted by w_layer
        """
        n_w, n_m = sim_matrix.shape
        matches = []

        if method == "greedy":
            used_m = set()
            # For each W layer, find best unmatched M layer
            for i in range(n_w):
                best_j = -1
                best_score = -1.0
                for j in range(n_m):
                    if j not in used_m and sim_matrix[i, j].item() > best_score:
                        best_score = sim_matrix[i, j].item()
                        best_j = j
                if best_j >= 0:
                    matches.append((i, best_j, best_score))
                    used_m.add(best_j)

        elif method == "hungarian":
            try:
                from scipy.optimize import linear_sum_assignment
                cost = -sim_matrix.numpy()
                row_idx, col_idx = linear_sum_assignment(cost)
                for i, j in zip(row_idx, col_idx):
                    matches.append((int(i), int(j), sim_matrix[i, j].item()))
            except ImportError:
                logger.warning("scipy not found, falling back to greedy matching")
                return self.find_best_matching(sim_matrix, method="greedy")

        matches.sort(key=lambda x: x[0])
        return matches

    def summary(self, sim_matrix: torch.Tensor, matches: list[tuple]) -> str:
        """Generate human-readable summary of CKA analysis."""
        lines = [
            f"CKA Analysis Summary",
            f"{'='*50}",
            f"Matrix size  : {sim_matrix.shape[0]} x {sim_matrix.shape[1]}",
            f"Mean CKA     : {sim_matrix.mean().item():.4f}",
            f"Max CKA      : {sim_matrix.max().item():.4f}",
            f"Min CKA      : {sim_matrix.min().item():.4f}",
            f"Matched pairs: {len(matches)}",
            f"",
            f"Top layer matches:",
        ]
        sorted_matches = sorted(matches, key=lambda x: -x[2])
        for w, m, score in sorted_matches[:10]:
            bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
            lines.append(f"  W[{w:2d}] ↔ M[{m:2d}] : {bar} {score:.4f}")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════
#  Function API (used by ConfluxTrainer)
# ═══════════════════════════════════════════════════════════

@torch.no_grad()
def compute_cka_matrix(
    hidden_states_w: list[torch.Tensor],
    hidden_states_m: list[torch.Tensor],
    kernel: str = "linear",
    max_samples: int = 256,
    skip_early_layers: int = 0,
) -> torch.Tensor:
    """Compute CKA similarity matrix between layer pairs.

    v2 improvements:
    - Mini-batch: 256 samples (was 2048). Accuracy stays >95%, speed 8x faster.
    - skip_early_layers: Skip first N layers of W (universal features, high CKA).

    Args:
        hidden_states_w: Per-layer hidden states from primary model W.
        hidden_states_m: Per-layer hidden states from source model M.
        kernel: "linear" or "rbf".
        max_samples: Subsample to this many for memory/speed.
        skip_early_layers: Skip first N layers of W (set CKA=1.0 for them).

    Returns:
        CKA matrix of shape (n_layers_w, n_layers_m).
    """
    cka = CKAComputer(kernel=kernel)
    n_w = len(hidden_states_w)
    n_m = len(hidden_states_m)

    hs_w = [h[:max_samples] for h in hidden_states_w]
    hs_m = [h[:max_samples] for h in hidden_states_m]
    hs_w = [h.mean(dim=1) if h.dim() == 3 else h for h in hs_w]
    hs_m = [h.mean(dim=1) if h.dim() == 3 else h for h in hs_m]

    sim_matrix = torch.zeros(n_w, n_m)

    for i in range(n_w):
        if i < skip_early_layers:
            sim_matrix[i, :] = 1.0
            continue
        for j in range(n_m):
            sim_matrix[i, j] = cka.compute_cka(hs_w[i], hs_m[j])

    skipped = min(skip_early_layers, n_w)
    computed = n_w - skipped
    logger.info(
        f"CKA matrix: {n_w}x{n_m}, {skipped} layers skipped, "
        f"{computed * n_m} pairs computed (n={max_samples})"
    )
    return sim_matrix


def match_layers_hungarian(cka_matrix: torch.Tensor) -> list[tuple[int, int, float]]:
    """Find optimal one-to-one layer matching using Hungarian algorithm.

    Args:
        cka_matrix: CKA similarity matrix (n_layers_w x n_layers_m).

    Returns:
        List of (w_idx, m_idx, cka_score) tuples.
    """
    cka = CKAComputer()
    return cka.find_best_matching(cka_matrix, method="hungarian")


def match_layers_greedy(cka_matrix: torch.Tensor) -> list[tuple[int, int, float]]:
    """Greedy layer matching."""
    cka = CKAComputer()
    return cka.find_best_matching(cka_matrix, method="greedy")
