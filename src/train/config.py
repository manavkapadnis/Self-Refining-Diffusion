import os
from dataclasses import dataclass, field
from typing import Optional

from trl.trainer.rloo_config import RLOOConfig


@dataclass
class ConfigPathArguments:
    tokenizer_config: Optional[str] = field(default=None, metadata={"help": "config path of tokenizer"})
    model_config: Optional[str] = field(default=None, metadata={"help": "config path of model"})
    reward_model_config: Optional[str] = field(default=None, metadata={"help": "config path of reward model"})
    train_dataset: Optional[str] = field(default=None, metadata={"help": "config path of training dataset"})
    data_collator: Optional[str] = field(default=None, metadata={"help": "config path of data collator"})


@dataclass
class CustomRLOOConfig(RLOOConfig):
    gamma: float = 0.90
    mean_kl: bool = False
    init_alpha: float = 1.5
    init_beta: float = 0.5
    relative: bool = True
    prediction_type: str = "alpha_beta"
    max_inference_steps: int = 15
    lambda_align: float = field(
        default=1.0,
        metadata={"help": "Weight for the CLIP alignment loss in self-refinement"},
    )
    lambda_id: float = field(
        default=0.1,
        metadata={"help": "Weight for the identity preservation loss in self-refinement"},
    )
    self_refine: bool = field(
        default=True,
        metadata={"help": "Enable self-refinement with CLIP-based rewards"},
    )
    
    lora_tpm: bool = field(
        default=True,
        metadata={"help": "Apply LoRA to Time Prediction Module"},
    )
    lora_diffusion: bool = field(
        default=False,
        metadata={"help": "Apply LoRA to Diffusion Transformer"},
    )
    lora_rank: int = field(
        default=4,
        metadata={"help": "Rank for LoRA adaptation"},
    )
    lora_alpha: float = field(
        default=8.0,
        metadata={"help": "Alpha scaling factor for LoRA"},
    )
    lora_dropout: float = field(
        default=0.0,
        metadata={"help": "Dropout rate for LoRA layers"},
    )