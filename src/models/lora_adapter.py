import math
import logging
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

class LoraAdapter(nn.Module):
    """Simple LoRA adapter for linear layers"""
    def __init__(self, original_module, rank=4, alpha=8.0, dropout=0.0):
        super().__init__()
        
        self.original_module = original_module
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        if isinstance(original_module, nn.Linear):
            orig_dtype = original_module.weight.dtype
            self.lora_down = nn.Linear(original_module.in_features, rank, bias=False)
            self.lora_up = nn.Linear(rank, original_module.out_features, bias=False)
            self.dropout = nn.Dropout(dropout)
            
            # Initialize weights
            nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_up.weight)
            
            # Convert to original dtype and make contiguous
            self.lora_down.weight.data = self.lora_down.weight.data.to(orig_dtype).contiguous()
            self.lora_up.weight.data = self.lora_up.weight.data.to(orig_dtype).contiguous()
        else:
            raise ValueError(f"LoRA can only be applied to nn.Linear, not {type(original_module)}")
    
    def forward(self, x):
        # Original output
        original_output = self.original_module(x)
        
        # LoRA path (keeping the same dtype throughout)
        lora_output = self.lora_down(x)
        lora_output = self.dropout(lora_output)
        lora_output = self.lora_up(lora_output)
        lora_output = lora_output * self.scaling
        
        # Combine outputs (both should now be the same dtype)
        return original_output + lora_output

def add_lora_to_sd3_transformer(model, lora_rank=4, lora_alpha=8.0, lora_dropout=0.0):
    """Apply LoRA to SD3 transformer attention modules"""
    transformer = model.transformer
    lora_layer_count = 0
    
    # Apply LoRA to attention modules in transformer blocks
    for i, block in enumerate(transformer.transformer_blocks):
        # Apply LoRA to main attention components
        if hasattr(block, "attn"):
            # Query, Key, Value projections
            if hasattr(block.attn, "to_q"):
                block.attn.to_q = LoraAdapter(block.attn.to_q, lora_rank, lora_alpha, lora_dropout)
                lora_layer_count += 1
                
            if hasattr(block.attn, "to_k"):
                block.attn.to_k = LoraAdapter(block.attn.to_k, lora_rank, lora_alpha, lora_dropout)
                lora_layer_count += 1
                
            if hasattr(block.attn, "to_v"):
                block.attn.to_v = LoraAdapter(block.attn.to_v, lora_rank, lora_alpha, lora_dropout)
                lora_layer_count += 1
            
            # Output projection (to_out is a ModuleList with the linear layer at index 0)
            if hasattr(block.attn, "to_out") and len(block.attn.to_out) > 0:
                block.attn.to_out[0] = LoraAdapter(block.attn.to_out[0], lora_rank, lora_alpha, lora_dropout)
                lora_layer_count += 1
                
            # Additional projections (for cross-attention)
            if hasattr(block.attn, "to_add_out") and block.attn.to_add_out is not None:
                block.attn.to_add_out = LoraAdapter(block.attn.to_add_out, lora_rank, lora_alpha, lora_dropout)
                lora_layer_count += 1
    
    logger.info(f"Applied LoRA to {lora_layer_count} transformer attention layers in total")
    return model

class LoraConv2dAdapter(nn.Module):
    """LoRA adapter for Conv2d layers"""
    def __init__(self, original_module, rank=4, alpha=8.0, dropout=0.0):
        super().__init__()
        
        self.original_module = original_module
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        if isinstance(original_module, nn.Conv2d):
            orig_dtype = original_module.weight.dtype
            in_channels = original_module.in_channels
            out_channels = original_module.out_channels
            kernel_size = original_module.kernel_size
            
            # Create low-rank decomposition for Conv2d
            # We'll use 1x1 convolutions for the low-rank decomposition
            self.lora_down = nn.Conv2d(in_channels, rank, kernel_size=1, bias=False)
            self.lora_up = nn.Conv2d(rank, out_channels, kernel_size=kernel_size, 
                                    padding=original_module.padding, 
                                    stride=original_module.stride, 
                                    bias=False)
            self.dropout = nn.Dropout(dropout)
            
            # Initialize weights
            nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_up.weight)
            
            # Convert to original dtype and make contiguous
            self.lora_down.weight.data = self.lora_down.weight.data.to(orig_dtype).contiguous()
            self.lora_up.weight.data = self.lora_up.weight.data.to(orig_dtype).contiguous()
        else:
            raise ValueError(f"LoRAConv2dAdapter can only be applied to nn.Conv2d, not {type(original_module)}")
    
    def forward(self, x):
        # Original output
        original_output = self.original_module(x)
        
        # LoRA path
        lora_output = self.lora_down(x)
        lora_output = self.dropout(lora_output)
        lora_output = self.lora_up(lora_output)
        lora_output = lora_output * self.scaling
        
        # Combine outputs
        return original_output + lora_output

def add_lora_to_time_predictor(model, lora_rank=4, lora_alpha=8.0, lora_dropout=0.0):
    """Apply LoRA to Time Predictor model including conv layers"""
    time_predictor = model.time_predictor
    lora_layer_count = 0
    
    # Apply LoRA to fully connected layers
    if hasattr(time_predictor, "fc1") and isinstance(time_predictor.fc1, nn.Linear):
        time_predictor.fc1 = LoraAdapter(time_predictor.fc1, lora_rank, lora_alpha, lora_dropout)
        lora_layer_count += 1
        logger.info(f"Applied LoRA to time predictor fc1 layer (dtype: {time_predictor.fc1.original_module.weight.dtype})")
        
    if hasattr(time_predictor, "fc2") and isinstance(time_predictor.fc2, nn.Linear):
        time_predictor.fc2 = LoraAdapter(time_predictor.fc2, lora_rank, lora_alpha, lora_dropout)
        lora_layer_count += 1
        logger.info(f"Applied LoRA to time predictor fc2 layer (dtype: {time_predictor.fc2.original_module.weight.dtype})")
    
    # Apply LoRA to conv layers
    if hasattr(time_predictor, "conv1") and isinstance(time_predictor.conv1, nn.Conv2d):
        time_predictor.conv1 = LoraConv2dAdapter(time_predictor.conv1, lora_rank, lora_alpha, lora_dropout)
        lora_layer_count += 1
        logger.info(f"Applied LoRA to time predictor conv1 layer (dtype: {time_predictor.conv1.original_module.weight.dtype})")
    
    if hasattr(time_predictor, "conv2") and isinstance(time_predictor.conv2, nn.Conv2d):
        time_predictor.conv2 = LoraConv2dAdapter(time_predictor.conv2, lora_rank, lora_alpha, lora_dropout)
        lora_layer_count += 1
        logger.info(f"Applied LoRA to time predictor conv2 layer (dtype: {time_predictor.conv2.original_module.weight.dtype})")
    
    logger.info(f"Applied LoRA to {lora_layer_count} time predictor layers in total")
    return model