import torch
import torch.nn as nn
from transformers import CLIPProcessor, CLIPModel
import logging

logger = logging.getLogger(__name__)

class CLIPReward(nn.Module):
    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        super(CLIPReward, self).__init__()
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()
        
        # Freeze CLIP model parameters
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Make all parameters contiguous
        self.make_contiguous()
            
        logger.info(f"Loaded CLIP model from {model_name}")
    
    def make_contiguous(self):
        """Make all parameters and buffers contiguous in memory."""
        for module in self.modules():
            for name, param in module.named_parameters():
                if param.data.is_contiguous() == False:
                    param.data = param.data.contiguous()
            for name, buffer in module.named_buffers():
                if buffer.data.is_contiguous() == False:
                    buffer.data = buffer.data.contiguous()
                    
    @torch.no_grad()  # Add this decorator to prevent gradients
    def score(self, prompt, image):
        """Compute CLIP alignment score between text prompt and image."""
        try:
            inputs = self.processor(
                text=prompt, 
                images=image, 
                return_tensors="pt",
                padding=True,
                truncation=True
            )
            
            # Move inputs to the same device as the model
            device = next(self.model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            outputs = self.model(**inputs)
            
            # Get image and text features
            image_features = outputs.image_embeds
            text_features = outputs.text_embeds
            
            # Normalize features
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            # Compute cosine similarity (alignment score)
            alignment_score = (image_features * text_features).sum(dim=-1)
            
            return alignment_score.detach().item()  # Detach to avoid gradients
                
        except Exception as e:
            logger.error(f"Error computing CLIP score: {e}")
            return 0.0