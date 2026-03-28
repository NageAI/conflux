"""
CONFLUX Trainer

The main orchestrator that ties all five modules into a single
training pipeline. Handles the complete flow:

    Profile → Initialize → Train → Merge → Deploy

Usage:
    config = ConfluxConfig(
        primary_model="Qwen/Qwen3-8B",
        source_models=[SourceModelConfig("meta-llama/Llama-3-8B", quantization_bits=2)],
    )
    trainer = ConfluxTrainer(config)
    trainer.run()
"""

import torch
import json
import time
from pathlib import Path
from typing import Optional
import logging

from conflux.config import ConfluxConfig
from conflux.cka import compute_cka_matrix, match_layers_hungarian
from conflux.residual import (
    extract_hidden_states,
    compute_residuals,
    ProjectionBank,
    aggregate_multi_source_residuals,
)
from conflux.svd_init import batch_svd_init, apply_svd_init_to_peft
from conflux.rank_alloc import allocate_ranks, generate_peft_config, print_allocation_summary
from conflux.profiler import profile_layer_pairs, print_profiling_summary

logger = logging.getLogger(__name__)


class ConfluxTrainer:
    """Main CONFLUX training pipeline.

    Orchestrates all five modules:
    1. Profiling (Module 4) - analyze model pairs
    2. Rank Allocation (Module 1) - assign per-layer ranks
    3. SVD Initialization (Module 2) - initialize LoRA from residuals
    4. Dual-Quantized Training (Module 3) - efficient training
    5. Multi-Source Switching (Module 5) - N-source support

    Args:
        config: ConfluxConfig with all pipeline parameters
    """

    def __init__(self, config: ConfluxConfig):
        config.validate()
        self.config = config
        self.profiling_reports = {}
        self.residual_infos = {}
        self.rank_allocations = {}
        self.svd_results = {}
        self.peft_configs = {}

        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        Path(config.cache_dir).mkdir(parents=True, exist_ok=True)

        self._setup_logging()

    def _setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [CONFLUX] %(levelname)s: %(message)s",
            datefmt="%H:%M:%S",
        )

    def run(self, calibration_texts: Optional[list[str]] = None):
        """Execute the complete CONFLUX pipeline.

        Args:
            calibration_texts: Texts for profiling and SVD initialization.
                             If None, a default calibration set is generated.
        """
        logger.info("=" * 60)
        logger.info("CONFLUX Pipeline Starting")
        logger.info(f"Primary model: {self.config.primary_model}")
        logger.info(f"Source models: {[s.alias for s in self.config.source_models]}")
        logger.info(f"N-source count: {self.config.num_sources}")
        logger.info("=" * 60)

        start_time = time.time()

        if calibration_texts is None:
            calibration_texts = self._default_calibration_texts()

        logger.info("\n[Phase 1/5] Loading models and extracting hidden states...")
        primary_model, primary_tokenizer = self._load_primary_model()
        hidden_states_w = extract_hidden_states(
            primary_model, primary_tokenizer, calibration_texts,
            batch_size=self.config.profiling_batch_size,
        )

        for source_config in self.config.source_models:
            alias = source_config.alias
            logger.info(f"\n[Phase 2/5] Profiling source: {alias}...")

            source_model, source_tokenizer = self._load_source_model(source_config)
            hidden_states_m = extract_hidden_states(
                source_model, source_tokenizer, calibration_texts,
                batch_size=self.config.profiling_batch_size,
            )

            residuals, report = profile_layer_pairs(
                hidden_states_w, hidden_states_m,
                primary_model_name=self.config.primary_model,
                source_model_name=source_config.model_name_or_path,
                source_alias=alias,
                cka_kernel=self.config.cka_kernel,
                matching_method=self.config.layer_matching_method,
                min_transfer_value=self.config.min_transfer_value,
            )

            self.profiling_reports[alias] = report
            self.residual_infos[alias] = residuals

            print_profiling_summary(report)

            if self.config.save_profiling_report:
                report_path = Path(self.config.output_dir) / f"profile_{alias}.json"
                report.save(str(report_path))

            del source_model
            torch.cuda.empty_cache()

        logger.info("\n[Phase 3/5] Allocating ranks...")
        if self.config.multi_source_strategy == "aggregated" and self.config.num_sources > 1:
            aggregated = aggregate_multi_source_residuals(
                self.residual_infos,
                strategy="max",
            )
            allocations = allocate_ranks(
                aggregated,
                rank_min=self.config.rank_min,
                rank_max=self.config.rank_max,
                hidden_dim=self._get_hidden_dim(primary_model),
            )
            self.rank_allocations["aggregated"] = allocations
            print_allocation_summary(allocations, self._get_hidden_dim(primary_model))
        else:
            for alias, residuals in self.residual_infos.items():
                allocations = allocate_ranks(
                    residuals,
                    rank_min=self.config.rank_min,
                    rank_max=self.config.rank_max,
                    hidden_dim=self._get_hidden_dim(primary_model),
                )
                self.rank_allocations[alias] = allocations
                print_allocation_summary(allocations, self._get_hidden_dim(primary_model))

        logger.info("\n[Phase 4/5] SVD Initialization...")
        if self.config.svd_init_enabled:
            for alias, residuals in self.residual_infos.items():
                alloc = self.rank_allocations.get(alias, self.rank_allocations.get("aggregated"))
                if alloc:
                    for r_info in residuals:
                        matching_alloc = next((a for a in alloc if a.layer_idx == r_info.layer_w_idx), None)
                        if matching_alloc:
                            r_info.assigned_rank = matching_alloc.rank

                svd_results = batch_svd_init(residuals)
                self.svd_results[alias] = svd_results
                logger.info(f"SVD init for {alias}: {len(svd_results)} layers initialized")

        logger.info("\n[Phase 5/5] Generating PEFT configurations...")
        for alias, alloc in self.rank_allocations.items():
            peft_config = generate_peft_config(
                alloc,
                target_modules=self.config.target_modules,
                lora_alpha_multiplier=self.config.lora_alpha_multiplier,
                lora_dropout=self.config.lora_dropout,
            )
            self.peft_configs[alias] = peft_config

            config_path = Path(self.config.output_dir) / f"peft_config_{alias}.json"
            with open(config_path, "w") as f:
                json.dump(peft_config, f, indent=2)
            logger.info(f"PEFT config saved to {config_path}")

        elapsed = time.time() - start_time
        self._print_final_summary(elapsed)

        return self

    def create_peft_model(self, alias: Optional[str] = None):
        """Create a PEFT model with CONFLUX initialization.

        Call this after run() to get a ready-to-train model.

        Args:
            alias: Source model alias. If None, uses first source.

        Returns:
            PEFT model with Residual-SVD initialized LoRA adapters.
        """
        try:
            from peft import get_peft_model, LoraConfig
        except ImportError:
            raise ImportError("peft library required: pip install peft")

        if alias is None:
            alias = list(self.peft_configs.keys())[0]

        peft_config_dict = self.peft_configs[alias]

        lora_config = LoraConfig(
            r=peft_config_dict["r"],
            lora_alpha=peft_config_dict["lora_alpha"],
            lora_dropout=peft_config_dict["lora_dropout"],
            target_modules=peft_config_dict["target_modules"],
            bias=peft_config_dict["bias"],
            task_type=peft_config_dict["task_type"],
        )

        primary_model, _ = self._load_primary_model()
        peft_model = get_peft_model(primary_model, lora_config)

        if alias in self.svd_results:
            apply_svd_init_to_peft(peft_model, self.svd_results[alias])

        trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in peft_model.parameters())
        logger.info(f"PEFT model created: {trainable:,} trainable / {total:,} total params ({trainable/total:.4%})")

        return peft_model

    def _load_primary_model(self):
        """Load the primary model W with quantization."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        except ImportError:
            raise ImportError("transformers library required: pip install transformers")

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=self.config.primary_quantization_bits == 4,
            load_in_8bit=self.config.primary_quantization_bits == 8,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        model = AutoModelForCausalLM.from_pretrained(
            self.config.primary_model,
            quantization_config=bnb_config,
            device_map=self.config.primary_device_map,
            torch_dtype="auto",
        )
        tokenizer = AutoTokenizer.from_pretrained(self.config.primary_model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        return model, tokenizer

    def _load_source_model(self, source_config):
        """Load a source model M with aggressive quantization."""
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=source_config.quantization_bits <= 4,
            load_in_8bit=source_config.quantization_bits == 8,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        model = AutoModelForCausalLM.from_pretrained(
            source_config.model_name_or_path,
            quantization_config=bnb_config,
            device_map=source_config.device_map,
            torch_dtype="auto",
        )
        tokenizer = AutoTokenizer.from_pretrained(source_config.model_name_or_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        return model, tokenizer

    def _get_hidden_dim(self, model) -> int:
        """Extract hidden dimension from model config."""
        config = model.config
        return getattr(config, "hidden_size", getattr(config, "d_model", 4096))

    def _default_calibration_texts(self) -> list[str]:
        """Generate default calibration texts for profiling."""
        return [
            "The theory of relativity fundamentally changed our understanding of space and time.",
            "In machine learning, gradient descent is used to minimize the loss function.",
            "Istanbul is a city that bridges two continents, Europe and Asia.",
            "The Fibonacci sequence appears throughout nature in surprising ways.",
            "Quantum computing leverages superposition and entanglement for computation.",
            "Effective leadership requires empathy, vision, and decisive action.",
            "The stock market reflects collective expectations about future economic performance.",
            "Neural networks with attention mechanisms have revolutionized natural language processing.",
            "Climate change poses unprecedented challenges to global food security.",
            "The art of brewing coffee involves precise control of temperature and extraction time.",
        ] * 50

    def _print_final_summary(self, elapsed: float):
        """Print final pipeline summary."""
        print("\n" + "=" * 60)
        print(" CONFLUX Pipeline Complete")
        print("=" * 60)
        print(f" Total time:      {elapsed:.1f}s")
        print(f" Primary model:   {self.config.primary_model}")
        print(f" Sources:         {', '.join(r.source_alias for r in self.profiling_reports.values())}")
        print(f" Output dir:      {self.config.output_dir}")
        print()

        for alias, report in self.profiling_reports.items():
            print(f" [{alias}]")
            print(f"   Active layers: {report.recommended_active_layers}")
            print(f"   Skipped:       {report.recommended_skip_layers}")
            print(f"   Mean transfer: {report.transfer_value_stats['mean']:.4f}")

        if self.svd_results:
            total_init = sum(len(v) for v in self.svd_results.values())
            print(f"\n SVD initialized:  {total_init} layer-adapter pairs")

        print(f"\n Next steps:")
        print(f"   1. trainer.create_peft_model() → get initialized model")
        print(f"   2. Train with SFTTrainer or custom loop")
        print(f"   3. Merge adapters: model.merge_and_unload()")
        print("=" * 60)
