#!/bin/bash
#SBATCH --job-name=diffusion_only_lora
#SBATCH --partition=shire-general
#SBATCH --time=48:00:00

#SBATCH --mem=500GB
#SBATCH --gres=gpu:A100_80GB:1
#SBATCH --cpus-per-task=20

#SBATCH --output=diffusion_only_lora.out
#SBATCH --error=diffusion_only_lora.err

#SBATCH --mail-type=END
#SBATCH --mail-user=mkapadni@andrew.cmu.edu

source /home/mkapadni/.bashrc  # Ensure this points to the correct file if different
conda activate gpu_env

cd /home/mkapadni/work/coursework/genai_project/TPDM_lora/TPDM

export WANDB_API_KEY="aa8f4ece8459cb64b7b96d837a17a0a435f8ddc3"

export HF_HUB_CACHE=/data/user_data/mkapadni/hf_cache/models
export HF_API_KEY=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
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Debugging Hydra Config
export HYDRA_FULL_ERROR=1

# TPDM training environment variables
export NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
export OMP_NUM_THREADS=4
export WANDB_PROJECT="genai_project"
export WANDB_MODE="online"
export RUN_NAME="training_diffusion_laion_$(date +'%Y-%m-%d')"

# Set output directory
OUTPUT_DIR="/data/user_data/mkapadni/genai_project/outputs_diffusion_lora/$(date +'%Y-%m-%d')/$RUN_NAME"

# Run training with LoRA
python -m torch.distributed.run --nproc_per_node $NUM_GPUS --nnodes 1 --standalone \
    main_diff_lora_tpdm_only_trainer.py \
    --model_config configs/models/sd3_pnt.yaml \
    --reward_model_config configs/models/image_reward.yaml \
    --train_dataset configs/datasets/hf_json_list_only_tpdm_lora.yaml \
    --data_collator configs/datasets/json_prompt_collator.yaml \
    --gamma 0.97 \
    --world_size $NUM_GPUS \
    --init_alpha 2.5 \
    --init_beta 1.0 \
    --kl_coef 0.00 \
    --self_refine false \
    --lora_tpm false \
    --lora_diffusion true \
    --lora_rank 4 \
    --lora_alpha 8.0 \
    --lora_dropout 0.0 \
    --per_device_train_batch_size 32 \
    --gradient_accumulation_steps 2 \
    --learning_rate 5e-5 \
    --lr_scheduler_type constant_with_warmup \
    --warmup_steps 0 \
    --adam_beta1 0.9 \
    --adam_beta2 0.99 \
    --adam_epsilon 1e-5 \
    --weight_decay 0.0 \
    --max_grad_norm 1.0 \
    --num_train_epochs 1 \
    --eval_steps 100 \
    --save_steps 100 \
    --torch_empty_cache_steps 10 \
    --logging_steps 1 \
    --report_to none \
    --resume_from_checkpoint true \
    --output_dir $OUTPUT_DIR \
    --run_name $RUN_NAME \
    --deepspeed configs/deepspeed/deepspeed_stage_0.json

echo '--------------------------'
echo main training task done
echo '--------------------------'