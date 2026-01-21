# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ms-swift (SWIFT: Scalable lightWeight Infrastructure for Fine-Tuning) is a comprehensive framework for large language model and multimodal model fine-tuning, training, inference, evaluation, quantization, and deployment. Developed by the ModelScope community in collaboration with Alibaba's DAMO team.

**Key capabilities:**
- Supports 600+ text-only LLMs and 300+ multimodal models
- Full pipeline: Training → Inference → Evaluation → Quantization → Deployment
- Training modes: Pre-training (PT), Supervised Fine-Tuning (SFT), RLHF, Reinforcement Learning
- Lightweight tuning: LoRA, QLoRA, DoRA, LoRA+, LLaMAPro, and more
- Distributed training: DDP, DeepSpeed ZeRO2/3, FSDP/FSDP2, Megatron parallelism

## Build and Development Commands

```bash
# Build
make whl          # Build wheel package
make clean        # Clean build artifacts
make docs         # Build documentation

# Testing
make test         # Run CI tests
python tests/run.py  # Run test suite

# Linting
make linter       # Run linting checks
pre-commit run --all-files  # Manual pre-commit execution
```

## CLI Commands

```bash
# Training
swift pt          # Pre-training
swift sft         # Supervised fine-tuning
swift rlhf        # RLHF training (DPO, PPO, KTO, CPO, SimPO, ORPO, RM, GRPO, etc.)

# Inference & Deployment
swift infer       # Inference
swift app         # Interactive UI for inference
swift deploy      # Model deployment

# Utilities
swift export      # Model export and quantization
swift eval        # Model evaluation
swift merge-lora  # Merge LoRA adapters
swift web-ui      # Web interface for full pipeline

# Megatron parallelism versions
megatron sft/pt/rlhf
```

All commands support `--config path/to/config.yaml` for configuration via YAML files.

## Code Architecture

### Directory Structure

```
swift/                          # Main package
├── arguments/                  # Training argument definitions (PT, SFT, RLHF, Export, Infer)
├── cli/                        # Command-line interface entry points
├── dataset/                    # Dataset loading and preprocessing
├── infer_engine/               # Inference engines (Transformers, vLLM, SGLang, LMDeploy)
├── megatron/                   # Megatron parallelism integration
├── model/                      # Model loading and processors
├── pipelines/                  # High-level training pipelines
├── rlhf_trainers/              # RLHF-specific trainers
├── template/                   # Message templates for different models
├── trainers/                   # Core trainer classes
├── tuners/                     # Tuning methods (LoRA, QLoRA, etc.)
├── tuner_plugin/               # Tuner plugin system
└── ui/                         # Web-UI components (Gradio-based)
```

### Key Architectural Patterns

**Trainer Architecture:**
- `Seq2SeqTrainer` extends HuggingFace's Seq2SeqTrainer
- Mixin pattern: `SwiftMixin`, `DataLoaderMixin` for feature composition
- `TrainerFactory` for dynamic trainer selection

**Plugin System:**
- `tuner_plugin` provides extensible tuner/adapter system
- `tuners_map` registry for available tuners

**Template System:**
- Different message templates for different models (Qwen, Llama, Mistral, etc.)
- Handles prompt formatting, special tokens, multi-turn conversations

**Pipeline Pattern:**
- High-level functions: `sft_main`, `pretrain_main`, `infer_main`, `rlhf_main`
- OmegaConf-based configuration from YAML files

## Code Style

- Line length limit: 120 columns
- Variable names: snake_case
- Class names: PascalCase
- Indentation: 4 spaces
- Pre-commit hooks enforced: flake8, isort, yapf

## Key Dependencies

- PyTorch (>=2.0)
- Transformers (>=4.33, <4.58)
- ModelScope (>=1.23)
- PEFT (>=0.11, <0.19)
- TRL (>=0.15, <0.25)
- DeepSpeed (>=0.14)
- vLLM, SGLang, LMDeploy for inference acceleration

## Documentation

- English: `/docs/source_en/`
- Chinese: `/docs/source/`
- Examples: `/examples/` (train, infer, deploy, eval, export, megatron, custom)
