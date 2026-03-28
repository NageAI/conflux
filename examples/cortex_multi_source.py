"""
CONFLUX Example: Multi-Source Training for Cortex-14B

Demonstrates N-source CONFLUX with two knowledge donors:
  - Llama-3-8B: reasoning patterns
  - Mistral-7B: instruction following

The Cortex architecture uses these combined residuals to build
a superior reasoning/routing layer for the Nage ecosystem.

Requirements:
    pip install conflux[full]
    GPU: A100 80GB (online) or A100 40GB (offline mode)
"""

from conflux import ConfluxConfig, SourceModelConfig, ConfluxTrainer
from conflux.utils import estimate_vram, print_vram_estimate


def main():
    # ── Multi-source VRAM estimate ────────────────────────

    # With two sources online
    vram = estimate_vram(
        primary_bits=4,
        primary_params_b=14.0,      # Qwen3-14B
        source_bits=2,
        source_params_b=8.0,        # ~avg of two sources
        lora_rank=48,
        num_lora_layers=40,
        hidden_dim=5120,
        batch_size=2,
    )
    print_vram_estimate(vram)

    # ── Configure multi-source ────────────────────────────

    config = ConfluxConfig(
        primary_model="Qwen/Qwen3-14B",
        primary_quantization_bits=4,

        source_models=[
            SourceModelConfig(
                model_name_or_path="meta-llama/Llama-3-8B-Instruct",
                quantization_bits=2,
                alias="llama3_reasoning",
            ),
            SourceModelConfig(
                model_name_or_path="mistralai/Mistral-7B-Instruct-v0.3",
                quantization_bits=2,
                alias="mistral_instruct",
            ),
        ],

        # Adaptive ranks with higher ceiling for 14B
        rank_min=8,
        rank_max=64,

        # Profiling
        cka_kernel="linear",
        layer_matching_method="cka_hungarian",
        min_transfer_value=0.08,

        # Multi-source aggregation strategy
        multi_source_strategy="aggregated",

        # Training — lower LR for 14B
        learning_rate=1e-5,
        residual_loss_weight=0.25,
        residual_loss_annealing="cosine",
        num_train_epochs=2,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=16,
        max_seq_length=4096,

        # Offline mode — sequential source extraction
        offline_extraction=True,
        cache_dir="./conflux_cache_cortex",

        output_dir="./conflux_output_cortex",
    )

    # ── Run ───────────────────────────────────────────────

    trainer = ConfluxTrainer(config)
    trainer.run()

    print("\n✓ Multi-source CONFLUX profiling complete.")
    print("  Profiling reports saved for both sources.")
    print("  Next: trainer.create_peft_model() → SFT → DPO → GRPO")


if __name__ == "__main__":
    main()
