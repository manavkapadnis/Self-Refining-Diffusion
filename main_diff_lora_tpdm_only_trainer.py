import logging
import os
import pathlib

import hydra
import torch
import transformers
from omegaconf import OmegaConf

from src.train.callbacks import DiffusionWandbCallback
from src.train.config import ConfigPathArguments, CustomRLOOConfig
from src.train.rloo_trainer import CommonRLOOTrainer
from src.train.train_utilis import setup_debug


log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=log_format)

logger = logging.getLogger(__name__)


def train(cfg, training_args):
    # Load model with self-refinement capabilities
    model = hydra.utils.instantiate(
        OmegaConf.load(cfg.model_config),
        init_alpha=training_args.init_alpha,
        init_beta=training_args.init_beta,
        relative=training_args.relative,
        prediction_type=training_args.prediction_type,
        fsdp=training_args.fsdp,
        max_inference_steps=training_args.max_inference_steps,
        lambda_align=training_args.lambda_align,  # Add alignment loss weight
        lambda_id=training_args.lambda_id,        # Add identity loss weight
        self_refine=training_args.self_refine,
        lora_tpm=training_args.lora_tpm,
        lora_diffusion=training_args.lora_diffusion,
        lora_rank=training_args.lora_rank,
        lora_alpha=training_args.lora_alpha,
        lora_dropout=training_args.lora_dropout,
    )
    logger.info(f"model loaded from {cfg.model_config}")
    
    # Load aesthetic reward model
    reward_model = hydra.utils.instantiate(OmegaConf.load(cfg.reward_model_config)).eval()
    logger.info(f"reward model loaded from {cfg.reward_model_config}")
    
    train_dataset = hydra.utils.instantiate(OmegaConf.load(cfg.train_dataset))
    logger.info(f"train dataset loaded from {cfg.train_dataset}")
    data_collator = hydra.utils.instantiate(OmegaConf.load(cfg.data_collator))
    logger.info(f"data collator loaded from {cfg.data_collator}")

    # Create custom trainer with self-refinement loss
    trainer = CommonRLOOTrainer(
        config=training_args,
        policy=model,
        reward_model=reward_model,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=train_dataset,
    )

    if "wandb" in training_args.report_to:
        wandb_callback = DiffusionWandbCallback(trainer=trainer)
        logger.info("wandb callback added")
        trainer.add_callback(wandb_callback)

    if (
        list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")) and args.resume_from_checkpoint is not None
    ) or os.path.isdir(args.resume_from_checkpoint):
        if os.path.isdir(args.resume_from_checkpoint):
            trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
        else:
            trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()


if __name__ == "__main__":
    parser = transformers.HfArgumentParser((ConfigPathArguments, CustomRLOOConfig))
    cfg, args = parser.parse_args_into_dataclasses()
    
    # Add self-refinement hyperparameters
    # args.lambda_align = getattr(args, 'lambda_align', 1.0)
    # args.lambda_id = getattr(args, 'lambda_id', 0.1)
    
    train(cfg, args)