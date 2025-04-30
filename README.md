# Self-refining Diffusion (TPDM)

This repository contains the implementation of the Self-refining Time Prediction Diffusion Model, a novel approach for adaptive noise scheduling in diffusion models.

## Directory Structure

```
├── configs/
│   ├── datasets/                  # Dataset configuration files
│   │   ├── example_json_dataset.yaml
│   │   ├── hf_json_list.yaml 
│   │   ├── hf_json_list_only_tpdm_lora.yaml
│   │   └── json_prompt_collator.yaml
│   ├── deepspeed/                 # DeepSpeed configuration files
│   │   ├── deepspeed_stage_0.json
│   │   ├── deepspeed_stage_2.json
│   │   ├── deepspeed_stage_2_offload.json
│   │   ├── deepspeed_stage_3.json
│   │   └── deepspeed_stage_3_offload.json
│   ├── fsdp/                      # FSDP configuration files
│   │   └── fsdp_sd3.json
│   └── models/                    # Model configuration files
│       ├── clip_reward.yaml
│       ├── image_reward.yaml
│       └── sd3_pnt.yaml
├── src/
│   ├── data/                      # Data loading and processing
│   │   ├── data_collator.py       # Collation functions for training data
│   │   ├── dummy_dataset.py       # Simple dataset for testing
│   │   ├── hf_dataset.py          # HuggingFace dataset loading utilities
│   │   └── json_dataset.py        # JSON dataset loading utilities
│   ├── models/                    # Model implementations
│   │   ├── lora_adapter.py        # LoRA adaptation for efficient fine-tuning
│   │   ├── model_utilis.py        # Common model utilities
│   │   ├── reference_distributions.py # Reference distribution implementations
│   │   └── stable_diffusion_3/    # SD3 model implementations 
│   │       ├── modeling_sd3_pnt.py # SD3 with predict-next-timestep capabilities
│   │       └── transformer_sd3.py  # SD3 transformer implementation
│   ├── reward_models/             # Reward models for training
│   │   ├── ImageReward/           # Image quality reward model
│   │   ├── PickScore/             # Pick-score reward model
│   │   ├── aesthetic_predictor_v2/ # Aesthetic prediction model v2
│   │   ├── aesthetic_predictor_v2_5/ # Aesthetic prediction model v2.5
│   │   └── clip_reward.py         # CLIP-based reward model
│   └── train/                     # Training utilities
│       ├── callbacks.py           # Training callbacks for logging and visualization
│       ├── config.py              # Training configurations
│       ├── rloo_trainer.py        # Reinforcement learning with online optimization trainer
│       └── train_utilis.py        # Training utilities
├── gradio_sd3_inference.py        # Gradio UI for SD3 model inference
├── gradio_self_refine_inference.py # Gradio UI for self-refining model inference
├── main_diff_lora_tpdm_only_trainer.py # Main script for TPDM-only LoRA training
├── main_diff_rloo_trainer.py      # Main script for RLOO training
├── main_diff_self_refine_trainer.py # Main script for self-refining model training
├── non_server_scripts/            # Scripts for local training
│   ├── launch_sd3_self_refine_train.sh # Launch self-refine training locally
│   └── launch_sd3_train.sh        # Launch SD3 training locally
├── pyproject.toml                 # Project configuration
├── requirements.txt               # Project dependencies
├── run_train_diffusion_lora_only.sh # Script to train diffusion-only LoRA
├── run_train_self_refine.sh      # Script to train self-refine model
└── run_train_tpdm_lora_only.sh   # Script to train TPDM-only LoRA
```

## Models

TPDM introduces a novel approach to diffusion model inference by adaptively determining the noise schedule on-the-fly. The key components include:

1. **Time Prediction Module (TPM)**: Predicts the optimal next noise level based on current latent features
2. **LoRA Adaptation**: Efficient fine-tuning of specific model components
3. **Self-Refinement**: Improves image quality through iterative refinement

## Training Variants

### 1. Standard TPDM Training

```bash
# Run standard TPDM training
bash non_server_scripts/launch_sd3_train.sh
```

This trains the base TPDM model with reinforcement learning from online optimization (RLOO) to learn the optimal noise scheduling policy.

### 2. TPDM-only LoRA Training

```bash
# Run TPDM-only LoRA training (only applies LoRA to the time prediction module)
bash run_train_tpdm_lora_only.sh
```

This training only applies LoRA to the Time Prediction Module, keeping the diffusion model weights frozen.

### 3. Diffusion-only LoRA Training

```bash
# Run diffusion-only LoRA training (only applies LoRA to the diffusion transformer)
bash run_train_diffusion_lora_only.sh
```

This training only applies LoRA to the diffusion transformer, keeping the time prediction module weights frozen.

### 4. Self-Refine LoRA Training

```bash
# Run self-refine LoRA training (applies LoRA to both TPDM and diffusion model)
bash run_train_self_refine.sh
```

This training applies LoRA to both the Time Prediction Module and the diffusion transformer, and incorporates a self-refinement mechanism using CLIP-based alignment scores.

## Inference

You can run inference using the provided Gradio interfaces:

```bash
# For standard SD3 inference
python gradio_sd3_inference.py

# For self-refine model inference
python gradio_self_refine_inference.py
```

## Requirements

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Configuration

Training can be customized through the configuration files in the `configs/` directory:

- Dataset configurations in `configs/datasets/`
- DeepSpeed configurations in `configs/deepspeed/`
- Model configurations in `configs/models/`
