import gradio as gr
import torch
from safetensors.torch import load_file
import logging

from src.models.stable_diffusion_3.modeling_sd3_pnt import SD3PredictNextTimeStepModelRLOOWrapper

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the model
# checkpoint_path = "/data/user_data/mkapadni/genai_project/outputs_self_refine/2025-04-27/training_diffusion_laion_2025-04-27/checkpoint-100/model.safetensors"  # Update this path
model = SD3PredictNextTimeStepModelRLOOWrapper(
    "stabilityai/stable-diffusion-3-medium-diffusers",
    torch_dtype=torch.float16,
    self_refine=True,
    lambda_align=1.0,
    lambda_id=0.1,
).eval().to("cuda")

# Load checkpoint
checkpoint = load_file(checkpoint_path)
model.load_state_dict(checkpoint, strict=False)
logger.info(f"Loaded self-refine LoRA checkpoint from {checkpoint_path}")

def generate_image(prompt, seed, max_steps, guidance_scale):
    """Generate an image with the self-refine model"""
    seed = int(seed)
    
    inputs = {
        "prompt": prompt,
        "max_inference_steps": int(max_steps),
        "guidance_scale": float(guidance_scale),
        "generator": torch.Generator("cuda").manual_seed(seed),
        "predict": True,
    }
    
    with torch.no_grad():
        outputs = model.agent_model.forward(**inputs)
    
    image = outputs.images[0][-1]
    steps = len(outputs.sigmas[0])
    
    return image, f"Inference steps: {steps}"

# Create Gradio interface
iface = gr.Interface(
    fn=generate_image,
    inputs=[
        gr.Textbox(label="Prompt", value="A beautiful sunset over mountains"),
        gr.Number(label="Seed", value=42),
        gr.Slider(minimum=5, maximum=50, value=35, step=1, label="Max Steps"),
        gr.Slider(minimum=1.0, maximum=15.0, value=7.0, step=0.5, label="Guidance Scale"),
    ],
    outputs=[
        gr.Image(label="Generated Image"),
        gr.Textbox(label="Inference Info"),
    ],
    title="Self-Refine SD3 with LoRA",
    description="Generate images using SD3 with self-refine LoRA applied to both diffusion and TPDM modules."
)

if __name__ == "__main__":
    iface.launch(share=True)