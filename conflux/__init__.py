"""
CONFLUX: Cross-architecture Optimized N-source Fine-tuning
         via Low-rank Unified eXtraction

Part of the Nage AI ecosystem.

Class API:
    from conflux import CKAComputer, ResidualSVDInitializer, AdaptiveRankAllocator

Function API (used by ConfluxTrainer):
    from conflux.cka import compute_cka_matrix, match_layers_hungarian
    from conflux.svd_init import batch_svd_init, apply_svd_init_to_peft
    from conflux.rank_alloc import allocate_ranks, generate_peft_config
    from conflux.profiler import profile_layer_pairs
    from conflux.residual import extract_hidden_states, compute_residuals
"""

__version__ = "0.3.0"
__author__ = "Ömer Asım"
__license__ = "Apache-2.0"

from conflux.config import ConfluxConfig, SourceModelConfig
from conflux.cka import CKAComputer
from conflux.residual import ResidualExtractor, ProjectionBank
from conflux.svd_init import ResidualSVDInitializer
from conflux.rank_alloc import AdaptiveRankAllocator
from conflux.profiler import InformativenessProfiler
from conflux.trainer import ConfluxTrainer
from conflux.eval import forgetting_benchmark, ForgettingReport

__all__ = [
    "ConfluxConfig",
    "SourceModelConfig",
    "ConfluxTrainer",
    "CKAComputer",
    "ResidualExtractor",
    "ProjectionBank",
    "ResidualSVDInitializer",
    "AdaptiveRankAllocator",
    "InformativenessProfiler",
    "forgetting_benchmark",
    "ForgettingReport",
]
