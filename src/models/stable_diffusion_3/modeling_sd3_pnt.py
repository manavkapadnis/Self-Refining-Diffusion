import logging
from typing import List, Optional, Union

import pyrootutils
import torch
import torch.distributed.checkpoint
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.fsdp import FullyShardedDataParallel
from transformers import (
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    T5EncoderModel,
    T5TokenizerFast,
)
from transformers.trainer import TrainingArguments

from diffusers.image_processor import VaeImageProcessor
from diffusers.models.autoencoders import AutoencoderKL
from diffusers.models.normalization import AdaLayerNormZeroSingle
from diffusers.utils.torch_utils import randn_tensor
from src.reward_models.clip_reward import CLIPReward


pyrootutils.setup_root(__file__, indicator="pyproject.toml", pythonpath=True)
from src.models.model_utilis import CustomDiffusionModelOutput, CustomFlowMatchEulerDiscreteScheduler
from src.models.reference_distributions import get_ref_beta
from src.models.stable_diffusion_3.transformer_sd3 import CustomSD3Transformer2DModel

# LORA
import math
import sys
import pdb
sys.path.append("src/models")
from lora_adapter import LoraAdapter, add_lora_to_time_predictor, add_lora_to_sd3_transformer
# Add this import for the CLIP reward
# from src.reward_models.clip_reward import CLIPReward

logger = logging.getLogger(__name__)


def reshape_hidden_states_to_2d(
    hidden_states: torch.Tensor,
    height: int = 64,
    width: int = 64,
    patch_size: int = 2,
) -> torch.Tensor:
    """
    Reshape the hidden states to have a 2D spatial structure.
    """
    hidden_states = hidden_states.reshape(
        shape=(
            hidden_states.shape[0],
            height // patch_size,
            width // patch_size,
            patch_size,
            patch_size,
            hidden_states.shape[-1],
        )
    )
    hidden_states = torch.einsum("nhwpqc->nchpwq", hidden_states)
    hidden_states = hidden_states.reshape(shape=(hidden_states.shape[0], hidden_states.shape[1], height, width))
    return hidden_states

class CustomAdaGroupNormZeroSingle(nn.Module):
    r"""
    Norm layer adaptive layer norm zero (adaLN-Zero).

    Parameters:
        embedding_dim (`int`): The size of each embedding vector.
        num_embeddings (`int`): The size of the embeddings dictionary.
    """

    def __init__(self, input_dim: int, embedding_dim: int, norm_type="group_norm", bias=True):
        super().__init__()

        self.silu = nn.SiLU()
        self.linear = nn.Linear(input_dim, 2 * embedding_dim, bias=bias)
        if norm_type == "group_norm":
            self.norm = nn.GroupNorm(1, embedding_dim, eps=1e-6)
        else:
            raise ValueError(
                f"Unsupported `norm_type` ({norm_type}) provided. Supported ones are: 'layer_norm', 'fp32_layer_norm'."
            )

    def forward(
        self,
        x: torch.Tensor, emb: Optional[torch.Tensor] = None):
        emb = self.linear(self.silu(emb))
        shift_msa, scale_msa = emb.chunk(2, dim=1)
        x = self.norm(x) * (1 + scale_msa[:, :, None, None]) + shift_msa[:, :, None, None]
        return x

class TimePredictor(nn.Module):
    def __init__(self, conv_out_channels, in_channels=1536 * 2, projection_dim=2, init_alpha=1.5, init_beta=0.5):
        super(TimePredictor, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=conv_out_channels, kernel_size=(3, 3), padding=1)
        self.conv2 = nn.Conv2d(conv_out_channels, conv_out_channels, kernel_size=(3, 3), padding=1, stride=2)

        self.fc1 = nn.Linear(conv_out_channels, 128)
        self.fc2 = nn.Linear(128, projection_dim)

        self.norm1 = CustomAdaGroupNormZeroSingle(in_channels // 2, conv_out_channels)
        self.epsilon = 1.0
        self.init_alpha = init_alpha
        self.init_beta = init_beta
        self._init_weights()

    def forward(self, x, temb):
        # 输入张量形状: (bs, 3072, 64, 64), (bs, 1536)

        # (bs, 3072, 64, 64) -> (bs, conv_out_channels, 64, 64)
        x = self.conv1(x)
        x = self.norm1(x, temb)
        # TODO: add timestep information by a layernorm with t-related weight & bias, similar to AdaLN
        x = F.silu(x)
        x = self.conv2(x)
        # (bs, conv_out_channels, 64, 64) -> (bs, conv_out_channels, 1, 1) -> (bs, conv_out_channels)
        x = F.adaptive_avg_pool2d(x, (16, 16))
        x = F.adaptive_max_pool2d(x, (1, 1)).view(x.size(0), -1)

        x = F.silu(self.fc1(x))  # (bs, 128)
        x = self.fc2(x)  # (bs, 2)
        return torch.exp(x) + self.epsilon

    def _init_weights(self):
        # init it by std is 0.02 and mean is 0
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    if isinstance(m, nn.Conv2d):
                        nn.init.constant_(m.bias, 0)
                    elif isinstance(m, nn.Linear):
                        nn.init.constant_(m.bias, 0)
        
        # Initialize the final layer bias more carefully to avoid NaN issues
        if hasattr(self, 'fc2'):
            nn.init.constant_(self.fc2.bias[0], self.init_alpha)
            nn.init.constant_(self.fc2.bias[1], self.init_beta)


class SD3PredictNextTimeStepModel(nn.Module):
    def __init__(
        self,
        pretrained_model_name_or_path,
        torch_dtype=torch.float16,
        min_sigma=0.001,
        init_alpha=1.5,
        init_beta=0.5,
        pre_process=False,
        relative=True,
        prediction_type="alpha_beta",
    ):
        super(SD3PredictNextTimeStepModel, self).__init__()

        # initialize the models
        self.vae = AutoencoderKL.from_pretrained(
            pretrained_model_name_or_path, subfolder="vae", torch_dtype=torch_dtype
        )
        self.transformer = CustomSD3Transformer2DModel.from_pretrained(
            pretrained_model_name_or_path, subfolder="transformer", torch_dtype=torch_dtype
        )

        self.time_predictor = TimePredictor(
            conv_out_channels=128,
            in_channels=self.transformer.config.caption_projection_dim * 2,
            projection_dim=2,
            init_alpha=init_alpha,
            init_beta=init_beta,
        ).to(dtype=self.vae.dtype)
        self.scheduler = CustomFlowMatchEulerDiscreteScheduler.from_pretrained(
            pretrained_model_name_or_path, subfolder="scheduler"
        )

        if not pre_process:
            self.text_encoder = CLIPTextModelWithProjection.from_pretrained(
                pretrained_model_name_or_path, subfolder="text_encoder", torch_dtype=torch_dtype
            ).eval()
            self.text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
                pretrained_model_name_or_path, subfolder="text_encoder_2", torch_dtype=torch_dtype
            ).eval()
            self.text_encoder_3 = (
                T5EncoderModel.from_pretrained(
                    pretrained_model_name_or_path, subfolder="text_encoder_3", torch_dtype=torch_dtype
                )
                .to(dtype=self.vae.dtype)
                .eval()
            )
            self.tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_name_or_path, subfolder="tokenizer")
            self.tokenizer_2 = CLIPTokenizer.from_pretrained(pretrained_model_name_or_path, subfolder="tokenizer_2")
            self.tokenizer_3 = T5TokenizerFast.from_pretrained(pretrained_model_name_or_path, subfolder="tokenizer_3")

        self.pre_process = pre_process
        self.vae_scale_factor = (
            2 ** (len(self.vae.config.block_out_channels) - 1) if hasattr(self, "vae") and self.vae is not None else 8
        )
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        self.tokenizer_max_length = 77
        self.default_sample_size = (
            self.transformer.config.sample_size
            if hasattr(self, "transformer") and self.transformer is not None
            else 128
        )
        self.patch_size = (
            self.transformer.config.patch_size if hasattr(self, "transformer") and self.transformer is not None else 2
        )

        self.min_sigma = min_sigma
        self.relative = relative
        self.epsilon = 1e-3
        self.prediction_type = prediction_type

        self.requires_grad_(False)
        self.eval()

    def _get_t5_prompt_embeds(
        self,
        prompt: Union[str, List[str]] = None,
        num_images_per_prompt: int = 1,
        max_sequence_length: int = 256,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        device = device or self.vae.device
        dtype = dtype or self.text_encoder.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        text_inputs = self.tokenizer_3(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids

        prompt_embeds = self.text_encoder_3(text_input_ids.to(device))[0]

        dtype = self.text_encoder_3.dtype
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

        _, seq_len, _ = prompt_embeds.shape

        # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        return prompt_embeds

    def _get_clip_prompt_embeds(
        self,
        prompt: Union[str, List[str]],
        num_images_per_prompt: int = 1,
        device: Optional[torch.device] = None,
        clip_skip: Optional[int] = None,
        clip_model_index: int = 0,
    ):
        device = device or self._execution_device

        clip_tokenizers = [self.tokenizer, self.tokenizer_2]
        clip_text_encoders = [self.text_encoder, self.text_encoder_2]

        tokenizer = clip_tokenizers[clip_model_index]
        text_encoder = clip_text_encoders[clip_model_index]

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer_max_length,
            truncation=True,
            return_tensors="pt",
        )

        text_input_ids = text_inputs.input_ids
        prompt_embeds = text_encoder(text_input_ids.to(device), output_hidden_states=True)
        pooled_prompt_embeds = prompt_embeds[0]

        if clip_skip is None:
            prompt_embeds = prompt_embeds.hidden_states[-2]
        else:
            prompt_embeds = prompt_embeds.hidden_states[-(clip_skip + 2)]

        prompt_embeds = prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)

        _, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        pooled_prompt_embeds = pooled_prompt_embeds.repeat(1, num_images_per_prompt, 1)
        pooled_prompt_embeds = pooled_prompt_embeds.view(batch_size * num_images_per_prompt, -1)

        return prompt_embeds, pooled_prompt_embeds

    @torch.inference_mode()
    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        prompt_2: Union[str, List[str]] = None,
        prompt_3: Union[str, List[str]] = None,
        device: Optional[torch.device] = None,
        num_images_per_prompt: int = 1,
        do_classifier_free_guidance: bool = True,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        negative_prompt_3: Optional[Union[str, List[str]]] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        clip_skip: Optional[int] = None,
        max_sequence_length: int = 256,
    ):
        device = device or self.vae.device

        prompt = [prompt] if isinstance(prompt, str) else prompt
        if prompt is not None:
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            prompt_2 = prompt_2 or prompt
            prompt_2 = [prompt_2] if isinstance(prompt_2, str) else prompt_2

            prompt_3 = prompt_3 or prompt
            prompt_3 = [prompt_3] if isinstance(prompt_3, str) else prompt_3

            prompt_embed, pooled_prompt_embed = self._get_clip_prompt_embeds(
                prompt=prompt,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                clip_skip=clip_skip,
                clip_model_index=0,
            )
            prompt_2_embed, pooled_prompt_2_embed = self._get_clip_prompt_embeds(
                prompt=prompt_2,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                clip_skip=clip_skip,
                clip_model_index=1,
            )
            clip_prompt_embeds = torch.cat([prompt_embed, prompt_2_embed], dim=-1)

            t5_prompt_embed = self._get_t5_prompt_embeds(
                prompt=prompt_3,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
            )

            clip_prompt_embeds = torch.nn.functional.pad(
                clip_prompt_embeds,
                (0, t5_prompt_embed.shape[-1] - clip_prompt_embeds.shape[-1]),
            )

            prompt_embeds = torch.cat([clip_prompt_embeds, t5_prompt_embed], dim=-2)
            pooled_prompt_embeds = torch.cat([pooled_prompt_embed, pooled_prompt_2_embed], dim=-1)

        if do_classifier_free_guidance and negative_prompt_embeds is None:
            negative_prompt = negative_prompt or ""
            negative_prompt_2 = negative_prompt_2 or negative_prompt
            negative_prompt_3 = negative_prompt_3 or negative_prompt

            # normalize str to list
            negative_prompt = batch_size * [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt
            negative_prompt_2 = (
                batch_size * [negative_prompt_2] if isinstance(negative_prompt_2, str) else negative_prompt_2
            )
            negative_prompt_3 = (
                batch_size * [negative_prompt_3] if isinstance(negative_prompt_3, str) else negative_prompt_3
            )

            if prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )

            negative_prompt_embed, negative_pooled_prompt_embed = self._get_clip_prompt_embeds(
                negative_prompt,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                clip_skip=None,
                clip_model_index=0,
            )
            negative_prompt_2_embed, negative_pooled_prompt_2_embed = self._get_clip_prompt_embeds(
                negative_prompt_2,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                clip_skip=None,
                clip_model_index=1,
            )
            negative_clip_prompt_embeds = torch.cat([negative_prompt_embed, negative_prompt_2_embed], dim=-1)

            t5_negative_prompt_embed = self._get_t5_prompt_embeds(
                prompt=negative_prompt_3,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
            )

            negative_clip_prompt_embeds = torch.nn.functional.pad(
                negative_clip_prompt_embeds,
                (
                    0,
                    t5_negative_prompt_embed.shape[-1] - negative_clip_prompt_embeds.shape[-1],
                ),
            )

            negative_prompt_embeds = torch.cat([negative_clip_prompt_embeds, t5_negative_prompt_embed], dim=-2)
            negative_pooled_prompt_embeds = torch.cat(
                [negative_pooled_prompt_embed, negative_pooled_prompt_2_embed], dim=-1
            )

        return (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        )

    def prepare_latents(
        self,
        batch_size,
        num_channels_latents,
        height,
        width,
        dtype,
        device,
        generator,
        latents=None,
    ):
        if latents is not None:
            return latents.to(device=device, dtype=dtype)

        shape = (
            batch_size,
            num_channels_latents,
            int(height) // self.vae_scale_factor,
            int(width) // self.vae_scale_factor,
        )

        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)

        return latents

    def forward(
        self,
        prompt: Union[str, List[str]] = None,
        negative_prompt: Union[str, List[str]] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        num_images_per_prompt: int = 1,
        max_inference_steps: int = 28,
        guidance_scale: Union[float, None] = 7.0,
        generator: Union[torch.Generator, List[torch.Generator]] = None,
        latents: Optional[torch.FloatTensor] = None,
        fix_sigmas: Optional[torch.FloatTensor] = None,
        return_full_process_images: bool = False,
        predict: bool = False,
    ) -> CustomDiffusionModelOutput:
        if prompt_embeds is None:
            prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = (
                self.encode_prompt(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    num_images_per_prompt=num_images_per_prompt,
                    device=self.vae.device,
                )
            )
        else:
            prompt_embeds = prompt_embeds.to(self.vae.device)
            pooled_prompt_embeds = pooled_prompt_embeds.to(self.vae.device)
            negative_prompt_embeds = (
                negative_prompt_embeds.to(self.vae.device) if negative_prompt_embeds is not None else None
            )
            negative_pooled_prompt_embeds = (
                negative_pooled_prompt_embeds.to(self.vae.device)
                if negative_pooled_prompt_embeds is not None
                else None
            )

        batch_size = prompt_embeds.shape[0]
        num_channels_latents = self.transformer.config.in_channels
        height = self.default_sample_size * self.vae_scale_factor
        width = self.default_sample_size * self.vae_scale_factor
        device = self.vae.device

        if latents is None:
            latents = self.prepare_latents(
                batch_size,
                num_channels_latents,
                height,
                width,
                prompt_embeds.dtype,
                device,
                generator,
                latents,
            )
        init_noise_latents = latents.clone()

        if guidance_scale is not None:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)

        sigma = torch.ones(batch_size, dtype=latents.dtype, device=device)
        history_latents = [[] for _ in range(batch_size)]
        sigmas = [[] for _ in range(batch_size)]
        logprobs = [[] for _ in range(batch_size)]
        prob_masks = [[] for _ in range(batch_size)]
        alphas = [[] for _ in range(batch_size)]
        betas = [[] for _ in range(batch_size)]
        now_step = 0

        hidden_states_combineds = []
        tembs = []
        # Denoising loop
        if fix_sigmas is not None:
            max_inference_steps = len(fix_sigmas[0])
        for step in range(max_inference_steps):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents.detach()] * 2) if guidance_scale else latents

            timestep = sigma.repeat(2) * 1000

            (noise_pred, temb, hidden_states_1, hidden_states_2) = self.transformer(
                hidden_states=latent_model_input,
                timestep=timestep,
                encoder_hidden_states=prompt_embeds,
                pooled_projections=pooled_prompt_embeds,
                return_dict=False,
            )

            if guidance_scale is not None:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                temb_uncond, temb_text = temb.chunk(2)
                temb = temb_uncond + guidance_scale * (temb_text - temb_uncond)
                hidden_states_1_uncond, hidden_states_1_text = hidden_states_1.chunk(2)
                hidden_states_1 = hidden_states_1_uncond + guidance_scale * (
                    hidden_states_1_text - hidden_states_1_uncond
                )
                hidden_states_2_uncond, hidden_states_2_text = hidden_states_2.chunk(2)
                hidden_states_2 = hidden_states_2_uncond + guidance_scale * (
                    hidden_states_2_text - hidden_states_2_uncond
                )

            hidden_states_1 = reshape_hidden_states_to_2d(hidden_states_1)
            hidden_states_2 = reshape_hidden_states_to_2d(hidden_states_2)
            hidden_states_combined = torch.cat([hidden_states_1, hidden_states_2], dim=1)
            hidden_states_combineds.append(hidden_states_combined.cpu())
            tembs.append(temb)

            time_preds = self.time_predictor(hidden_states_combined, temb)
            sigma_next = torch.zeros_like(sigma)
            for i, (param1, param2) in enumerate(time_preds):
                if self.prediction_type == "alpha_beta":
                    alpha, beta = param1, param2
                elif self.prediction_type == "mode_concentration":
                    alpha = param1 * (param2 - 2) + 1
                    beta = (1 - param1) * (param2 - 2) + 1
                beta_dist = torch.distributions.Beta(alpha, beta)

                if predict:
                    ratio = beta_dist.mode
                else:
                    ratio = beta_dist.sample()
                ratio = (
                    ratio.clamp(self.epsilon, 1 - self.epsilon)
                    if self.relative
                    else ratio.clamp(self.epsilon, sigma[i]).clamp(0, 1 - self.epsilon)
                )
                # TODO: Different map function attempts
                sigma_next[i] = sigma[i] * ratio if self.relative else sigma[i] - ratio
                sigmas[i].append(sigma_next[i])

                # if ratio is nan pdb
                prob = beta_dist.log_prob(ratio)
                logprobs[i].append(prob)
                if sigma[i] < self.min_sigma:
                    prob_masks[i].append(torch.tensor(1))
                    if predict:
                        sigma_next[i] = torch.tensor(0.0).to(sigma_next.device)
                else:
                    prob_masks[i].append(torch.tensor(0))

                alphas[i].append(alpha)
                betas[i].append(beta)

            latents = self.scheduler.custom_step(
                noise_pred,
                sigma_next=sigma_next,
                sigma=sigma,
                sample=latents,
                return_dict=False,
            )[0]

            for i in range(len(history_latents)):
                # record the latents
                if len(history_latents[i]) == 0:
                    history_latents[i] = latents[i].detach().unsqueeze(0)
                else:
                    history_latents[i] = torch.cat([history_latents[i], latents[i].detach().unsqueeze(0)], dim=0)

            # all sigma are lower than the threshold, we stop the chain
            if (sigma_next < self.min_sigma).all():
                break

            sigma = sigma_next
            now_step += 1

        # TODO: check
        INVALID_LOGPROB = 1.0
        sigmas = torch.stack([torch.stack(item) for item in sigmas])
        logprobs = torch.stack([torch.stack(item) for item in logprobs])
        prob_masks = torch.stack([torch.stack(item) for item in prob_masks]).bool().to(logprobs.device)
        alphas = torch.stack([torch.stack(item) for item in alphas])
        betas = torch.stack([torch.stack(item) for item in betas])
        logprobs = torch.masked_fill(logprobs, prob_masks, INVALID_LOGPROB)

        # (num_steps, batch_size, ...) -> (batch_size, num_steps, ...)
        hidden_states_combineds = torch.stack(hidden_states_combineds).permute(1, 0, 2, 3, 4)
        tembs = torch.stack(tembs).permute(1, 0, 2)

        images = []
        last_valid_indices = []
        if return_full_process_images:
            for latents in history_latents:
                latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
                # chunk the latents into chunks of 8
                chunk_size = (len(latents) // 8) + 1
                chunk_latents = latents.chunk(chunk_size, dim=0)
                del latents
                image = None
                for latents in chunk_latents:
                    if image is None:
                        image = self.vae.decode(latents, return_dict=False)[0].detach()
                    else:
                        image = torch.cat([image, self.vae.decode(latents, return_dict=False)[0].detach()], dim=0)
                    torch.cuda.empty_cache()
                images.append(self.image_processor.postprocess(image, output_type="pil"))
        else:
            # only decode the last valid latents
            for i in range(prob_masks.shape[0]):
                last_valid_index = torch.where(~prob_masks[i])[0][-1]
                last_valid_indices.append(last_valid_index)

            for i, latents in enumerate(history_latents):
                last_valid_index = last_valid_indices[i]
                latents = latents[last_valid_index]
                latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
                image = self.vae.decode(latents.unsqueeze(0), return_dict=False)[0].detach()
                images.append(self.image_processor.postprocess(image, output_type="pil"))

        return CustomDiffusionModelOutput(
            init_noise_latents=init_noise_latents,
            hidden_states_combineds=hidden_states_combineds,
            tembs=tembs,
            last_valid_indices=last_valid_indices,
            images=images,
            alphas=alphas,
            betas=betas,
            sigmas=sigmas,
            logprobs=logprobs,
            prob_masks=prob_masks,
        )

    def only_predict_logprobs(
        self,
        fix_sigmas: torch.Tensor,  # (bs, steps, ...)
        fix_hidden_states_combineds: torch.Tensor,  # (bs, steps, ...)
        fix_tembs: torch.Tensor, 
    ):
        if fix_sigmas is None:
            raise ValueError("fix_sigmas must be provided")
        if fix_hidden_states_combineds is None:
            raise ValueError("fix_hidden_states_combineds must be provided")

        batch_size = fix_sigmas.shape[0]
        max_inference_steps = fix_sigmas.shape[1]

        sigma = torch.ones(batch_size, dtype=self.vae.dtype, device=self.vae.device)
        sigmas = [[] for _ in range(batch_size)]
        logprobs = [[] for _ in range(batch_size)]
        prob_masks = [[] for _ in range(batch_size)]

        fix_hidden_states_combineds = fix_hidden_states_combineds.to(self.vae.device).permute(1, 0, 2, 3, 4)
        fix_tembs = fix_tembs.permute(1, 0, 2)

        for step in range(max_inference_steps):
            fix_hidden_states_combined = fix_hidden_states_combineds[step]
            fix_temb = fix_tembs[step]
            time_pred = self.time_predictor(fix_hidden_states_combined, fix_temb)
            sigma_next = torch.zeros_like(sigma)
            for i, (alpha, beta) in enumerate(time_pred):
                # Cast to float32 for numerical stability 
                alpha_32 = alpha.float()
                beta_32 = beta.float()
                
                # Clamp to ensure valid parameter values (must be > 0)
                alpha_32 = torch.clamp(alpha_32, min=1e-3, max=1e4)
                beta_32 = torch.clamp(beta_32, min=1e-3, max=1e4)
                
                try:
                    beta_dist = torch.distributions.Beta(alpha_32, beta_32)
                except (ValueError, RuntimeError) as e:
                    # Fallback in case of error
                    logger.warning(f"Error creating Beta distribution with alpha={alpha_32}, beta={beta_32}. Using fallback values.")
                    alpha_32 = torch.tensor(1.0, device=alpha.device)
                    beta_32 = torch.tensor(1.0, device=beta.device)
                    beta_dist = torch.distributions.Beta(alpha_32, beta_32)
                
                # if now sigma is smaller than min_sigma, we should not get prob from beta_dist
                if sigma[i] < self.min_sigma:
                    sigma_next[i] = fix_sigmas[i][step]
                    sigmas[i].append(sigma_next[i])
                    logprobs[i].append(torch.tensor(0.0).to(self.vae.device))
                    prob_masks[i].append(torch.tensor(1))
                    continue
                else:
                    sigma_next[i] = fix_sigmas[i][step]
                    ratio = sigma_next[i] / sigma[i] if self.relative else sigma[i] - sigma_next[i]
                
                # Ensure ratio is valid for log_prob calculation
                ratio = torch.clamp(ratio, min=self.epsilon, max=1 - self.epsilon)
                
                try:
                    prob = beta_dist.log_prob(ratio.float())  # Cast ratio to float32 too
                    if torch.isnan(prob) or torch.isinf(prob):
                        # If still NaN, use a small negative value as fallback
                        logger.warning(f"NaN or Inf in log_prob calculation with ratio={ratio}. Using fallback.")
                        prob = torch.tensor(-1.0, device=prob.device, dtype=prob.dtype)
                except (ValueError, RuntimeError) as e:
                    logger.warning(f"Error calculating log_prob with ratio={ratio}. Using fallback.")
                    prob = torch.tensor(-1.0, device=self.vae.device)
                
                sigmas[i].append(sigma_next[i])
                logprobs[i].append(prob)
                if sigma[i] < self.min_sigma:
                    prob_masks[i].append(torch.tensor(1))
                else:
                    prob_masks[i].append(torch.tensor(0))

            sigma = sigma_next

        # the value will not influence the result
        INVALID_LOGPROB = 1.0
        logprobs = torch.stack([torch.stack(item) for item in logprobs])
        prob_masks = torch.stack([torch.stack(item) for item in prob_masks]).bool().to(logprobs.device)
        logprobs = torch.masked_fill(logprobs, prob_masks, INVALID_LOGPROB)

        return {"logprobs": logprobs}


class SD3PredictNextTimeStepModelRLOOWrapper(nn.Module):
    def __init__(
        self,
        pretrained_model_name_or_path,
        torch_dtype: torch.dtype = torch.float16,
        min_sigma: float = 0.01,
        pre_process: bool = False,
        init_alpha: float = 1.5,
        init_beta: float = 0.5,
        relative: bool = True,
        prediction_type: str = "alpha_beta",
        fsdp: str = [],
        max_inference_steps: int = 28,
        lambda_align: float = 1.0,  
        lambda_id: float = 0.1,      
        self_refine: bool = False,
        lora_tpm: bool = True,
        lora_diffusion: bool = False,
        lora_rank: int = 4,
        lora_alpha: float = 8.0,
        lora_dropout: float = 0.0,
    ):
        super(SD3PredictNextTimeStepModelRLOOWrapper, self).__init__()
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.agent_model = SD3PredictNextTimeStepModel(
            pretrained_model_name_or_path,
            torch_dtype=torch_dtype,
            init_alpha=init_alpha,
            init_beta=init_beta,
            min_sigma=min_sigma,
            pre_process=pre_process,
            relative=relative,
            prediction_type=prediction_type,
        ).eval()

        self.relative = relative
        self.fsdp = fsdp
        self.max_inference_steps = max_inference_steps
        self.lambda_align = lambda_align
        self.lambda_id = lambda_id
        self.self_refine = self_refine
        # Initialize CLIP reward model
        if self_refine:
            self.clip_reward = CLIPReward().to(torch_dtype)
            self.clip_reward.eval()
            logger.info("Since self-refining is enabled, clip model loaded.")
        else:
            logger.info("Since self-refining is disabled, clip model not loaded.")
        

        # Count total parameters before LoRA
        total_params_before = sum(p.numel() for p in self.agent_model.parameters())
        logger.info(f"Total parameters before LoRA: {total_params_before:,}")
        
        # Freeze all parameters
        self.agent_model.requires_grad_(False)
        
        # Apply LoRA based on configuration parameters
        if lora_diffusion:
            logger.info(f"Applying LoRA to diffusion transformer with rank={lora_rank}")
            self.agent_model = add_lora_to_sd3_transformer(
                self.agent_model,
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout
            )
        
        # Apply LoRA to time predictor if enabled
        if lora_tpm:
            logger.info(f"Applying LoRA to time prediction module with rank={lora_rank}")
            self.agent_model.time_predictor.train()
            self.agent_model = add_lora_to_time_predictor(
                self.agent_model,
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout
            )
        
        # Make tensors contiguous for distributed training
        self.make_contiguous()
        # Count trainable parameters after LoRA
        trainable_params = sum(p.numel() for p in self.agent_model.parameters() if p.requires_grad)
        total_params_after = sum(p.numel() for p in self.agent_model.parameters())
        
        logger.info(f"Total parameters after LoRA: {total_params_after:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        logger.info(f"Percentage of trainable parameters: {trainable_params/total_params_after*100:.4f}%")

    def make_contiguous(self):
        """Make all parameters and buffers contiguous in memory."""
        for module in self.modules():
            for name, param in module.named_parameters():
                if param.data.is_contiguous() == False:
                    param.data = param.data.contiguous()
            for name, buffer in module.named_buffers():
                if buffer.data.is_contiguous() == False:
                    buffer.data = buffer.data.contiguous()

    def self_refinement_loss(self, inputs, outputs):
        """Compute self-refinement loss based on CLIP alignment scores."""
        prompts = inputs.get("prompt", None)
        images = outputs.get("images", None)
        
        if prompts is None or images is None:
            raise ValueError("Prompts and images must be provided")
        
        alignment_scores = []
        
        # Compute alignment scores for each image-prompt pair
        for prompt, image_list in zip(prompts, images):
            # Use the last valid image for self-refinement
            if isinstance(image_list, list):
                last_image = image_list[-1] if image_list else None
            else:
                last_image = image_list
                
            if last_image is not None:
                score = self.clip_reward.score(prompt, last_image)
                alignment_scores.append(score)
            else:
                alignment_scores.append(0.0)
        
        # Convert to tensor with gradient tracking
        device = next(self.agent_model.parameters()).device
        alignment_scores = torch.tensor(alignment_scores, device=device, dtype=torch.float32, requires_grad=True)
        
        # Compute self-refinement loss (negative alignment score)
        refinement_loss = -self.lambda_align * alignment_scores.mean()
        
        # Add identity preservation term (L2 regularization on LoRA parameters)
        identity_loss = 0.0
        for name, param in self.agent_model.named_parameters():
            if 'lora' in name and param.requires_grad:
                identity_loss += torch.norm(param) ** 2
        identity_loss = self.lambda_id * identity_loss
        
        total_refinement_loss = refinement_loss + identity_loss
        
        # Log the losses
        logger.info(f"Refinement loss: {refinement_loss.item():.4f}")
        logger.info(f"Identity loss: {identity_loss.item():.4f}")
        logger.info(f"Total refinement loss: {total_refinement_loss.item():.4f}")
        
        return total_refinement_loss

    def rloo_repeat(self, data, rloo_k):
        """make the data repeat rloo_k times
        Args:
            data: dict of data
            rloo_k: int
        Returns:
            data: dict of data that is repeated rloo_k times
        """
        data["prompt"] = data["prompt"] * rloo_k
        for key in [
            "prompt_embeds",
            "negative_prompt_embeds",
            "pooled_prompt_embeds",
            "negative_pooled_prompt_embeds",
        ]:
            if key in data:
                size = [rloo_k] + [1] * (len(data[key].shape) - 1)
                data[key] = data[key].repeat(*size)
        return data

    def sample(self, inputs):
        """Generate model outputs step by step for inputs
        Args:
            inputs: dict of inputs
        Returns:
            outputs: dict of final outputs after sampling
        """
        if "3.5" in self.pretrained_model_name_or_path:
            inputs["guidance_scale"] = 3.5
        inputs["max_inference_steps"] = self.max_inference_steps
        if len(self.fsdp) > 0:
            with FullyShardedDataParallel.summon_full_params(self):
                outputs = self.agent_model(**inputs)
        else:
            outputs = self.agent_model(**inputs)
        # TODO: add reward model
        # rewards = self.reward(inputs, outputs)
        # outputs.update(rewards)
        return outputs

    #### uncomment below reward for self-refine one
    # def reward(self, inputs, outputs, reward_model, gamma=0.8, return_last_reward=False):
    #     """Given a batch of model inputs and outputs, provide the rewards of the outputs, using the final image in outputs
    #     Args:
    #         inputs: dict of inputs
    #         outputs: dict of outputs
    #         reward_model: reward model (aesthetic)
    #         return_last_reward: whether to return the last reward
    #     Returns:
    #         rewards: tensor of rewards (bs, )
    #     """
    #     prompts = inputs.get("prompt", None)
    #     images = outputs.get("images", None)
    #     prob_masks = outputs.get("prob_masks", None)
    #     last_valid_indices = outputs.get("last_valid_indices", [])
    #     rewards = []
    #     last_image_rewards = []
    #     refinement_rewards = []
        
    #     if prompts is None or images is None:
    #         raise ValueError("prompt and images must be provided")
    #     elif len(prompts) != len(images):
    #         raise ValueError("prompt and images must have the same length")
        
    #     for i, (prompt, image, prob_mask) in enumerate(zip(prompts, images, prob_masks)):
    #         # use last image where prob_mask is false to calculate reward, and use gamma to discount the reward
    #         if last_valid_indices == []:
    #             last_image_idx = torch.where(~prob_mask.bool())[-1][-1].item()
    #             last_image = image[last_image_idx]
    #         else:
    #             last_image_idx = last_valid_indices[i].item()
    #             last_image = image[0]
            
    #         # Get aesthetic reward
    #         last_image_reward = reward_model.score(prompt, last_image)
            
    #         # Get CLIP alignment reward 
    #         clip_score = self.clip_reward.score(prompt, last_image)
            
    #         # Combine rewards
    #         combined_last_reward = last_image_reward + self.lambda_align * clip_score
            
    #         last_image_rewards.append(last_image_reward)
    #         refinement_rewards.append(clip_score)
            
    #         # Apply gamma discounting to the combined reward
    #         reward = 0
    #         for j in range(last_image_idx + 1):
    #             reward += combined_last_reward * (gamma ** (last_image_idx - j))
    #         reward = reward / (last_image_idx + 1)
    #         rewards.append(reward)
        
    #     rewards = torch.tensor(rewards)
    #     last_image_rewards = torch.tensor(last_image_rewards)
    #     refinement_rewards = torch.tensor(refinement_rewards)
        
    #     # Log the rewards for monitoring
    #     logger.info(f"Aesthetic rewards: {last_image_rewards.mean().item():.4f}")
    #     logger.info(f"CLIP refinement rewards: {refinement_rewards.mean().item():.4f}")
    #     logger.info(f"Combined rewards: {rewards.mean().item():.4f}")
        
    #     if return_last_reward:
    #         # Return combined rewards and original aesthetic rewards
    #         return rewards, last_image_rewards
    #     else:
    #         return rewards

    # def reward(self, inputs, outputs, reward_model, gamma=0.8, return_last_reward=False):
    #     """Given a batch of model inputs and outputs, provide the rewards of the outputs, using the final image in outputs
    #     Args:
    #         inputs: dict of inputs
    #         outputs: dict of outputs
    #         reward_model: reward model
    #         return_last_reward: whether to return the last reward
    #     Returns:
    #         rewards: tensor of rewards (bs, )
    #     """
    #     prompts = inputs.get("prompt", None)
    #     images = outputs.get("images", None)
    #     prob_masks = outputs.get("prob_masks", None)
    #     last_valid_indices = outputs.get("last_valid_indices", [])
    #     rewards = []
    #     last_image_rewards = []
    #     if prompts is None or images is None:
    #         raise ValueError("prompt and images must be provided")
    #     elif len(prompts) != len(images):
    #         raise ValueError("prompt and images must have the same length")
    #     for i, (prompt, image, prob_mask) in enumerate(zip(prompts, images, prob_masks)):
    #         # use last image where prob_mask is false to calculate reward, and use gamma to discount the reward
    #         if last_valid_indices == []:
    #             last_image_idx = torch.where(~prob_mask.bool())[-1][-1].item()
    #             last_image = image[last_image_idx]
    #         else:
    #             last_image_idx = last_valid_indices[i].item()
    #             last_image = image[0]
    #         last_image_reward = reward_model.score(prompt, last_image)
    #         last_image_rewards.append(last_image_reward)
    #         reward = 0
    #         for i in range(last_image_idx + 1):
    #             reward += last_image_reward * (gamma ** (last_image_idx - i))
    #         reward = reward / (last_image_idx + 1)
    #         rewards.append(reward)

    #     rewards = torch.tensor(rewards)
    #     last_image_rewards = torch.tensor(last_image_rewards)
    #     if return_last_reward:
    #         return rewards, last_image_rewards
    #     else:
    #         return rewards

    def reward(self, inputs, outputs, reward_model, gamma=0.8, return_last_reward=False, self_refine=False):
        """Given a batch of model inputs and outputs, provide the rewards of the outputs, using the final image in outputs
        Args:
            inputs: dict of inputs
            outputs: dict of outputs
            reward_model: reward model (aesthetic)
            gamma: discount factor
            return_last_reward: whether to return the last reward
            self_refine: whether to use CLIP-based self-refinement
        Returns:
            rewards: tensor of rewards (bs, )
        """
        prompts = inputs.get("prompt", None)
        images = outputs.get("images", None)
        prob_masks = outputs.get("prob_masks", None)
        last_valid_indices = outputs.get("last_valid_indices", [])
        rewards = []
        last_image_rewards = []
        refinement_rewards = [] if self_refine else None
        
        if prompts is None or images is None:
            raise ValueError("prompt and images must be provided")
        elif len(prompts) != len(images):
            raise ValueError("prompt and images must have the same length")
        
        for i, (prompt, image, prob_mask) in enumerate(zip(prompts, images, prob_masks)):
            # use last image where prob_mask is false to calculate reward, and use gamma to discount the reward
            if last_valid_indices == []:
                last_image_idx = torch.where(~prob_mask.bool())[-1][-1].item()
                last_image = image[last_image_idx]
            else:
                last_image_idx = last_valid_indices[i].item()
                last_image = image[0]
            
            # Get aesthetic reward
            last_image_reward = reward_model.score(prompt, last_image)
            
            # Optionally get CLIP alignment reward if self_refine is enabled
            if self_refine:
                clip_score = self.clip_reward.score(prompt, last_image)
                combined_last_reward = last_image_reward + self.lambda_align * clip_score
                refinement_rewards.append(clip_score)
            else:
                combined_last_reward = last_image_reward
            
            last_image_rewards.append(last_image_reward)
            
            # Apply gamma discounting to the reward (combined if self_refine is enabled)
            reward = 0
            for j in range(last_image_idx + 1):
                reward += combined_last_reward * (gamma ** (last_image_idx - j))
            reward = reward / (last_image_idx + 1)
            rewards.append(reward)
        
        rewards = torch.tensor(rewards)
        last_image_rewards = torch.tensor(last_image_rewards)
        
        # Log rewards if self-refine is enabled
        if self_refine:
            refinement_rewards = torch.tensor(refinement_rewards)
            logger.info(f"Aesthetic rewards: {last_image_rewards.mean().item():.4f}")
            logger.info(f"CLIP refinement rewards: {refinement_rewards.mean().item():.4f}")
            logger.info(f"Combined rewards: {rewards.mean().item():.4f}")
        else:
            logger.info(f"Rewards: {rewards.mean().item():.4f}")
        
        if return_last_reward:
            return rewards, last_image_rewards
        else:
            return rewards


    def logprobs(self, inputs, outputs):
        """Given a batch of model inputs and outputs, provide the logprobs of the outputs, using the actions in outputs
        Args:
            outputs: dict of outputs
        Returns:
            logprobs: tensor of logprobs (bs, seq_len)
            prob_masks: tensor of masks for the logprobs (bs, seq_len)
        """
        # outputs = self.agent_model(latents=outputs["init_noise_latents"], fix_sigmas=outputs["sigmas"], **inputs)
        if len(self.fsdp) > 0:
            with FullyShardedDataParallel.summon_full_params(self):
                outputs = self.agent_model.only_predict_logprobs(
                    fix_sigmas=outputs["sigmas"],
                    fix_hidden_states_combineds=outputs["hidden_states_combineds"],
                    fix_tembs=outputs["tembs"],
                )
        else:
            outputs = self.agent_model.only_predict_logprobs(
                fix_sigmas=outputs["sigmas"],
                fix_hidden_states_combineds=outputs["hidden_states_combineds"],
                fix_tembs=outputs["tembs"],
            )
        return outputs["logprobs"]

    def kl_divergence(self, outputs: CustomDiffusionModelOutput):
        """Given a batch of model outputs, provide the kl divergence of the outputs, using the alphas and betas in outputs
        Args:
            outputs: dict of outputs
        Returns:
            kl_divergence: tensor of kl divergence (bs, )
        """
        alphas = outputs["alphas"]
        betas = outputs["betas"]
        prob_masks = outputs["prob_masks"]
        kl_divergences = [[] for _ in range(len(alphas))]
        input_sigmas = F.pad(outputs["sigmas"][..., :-1], (1, 0), value=1.0)
        ref_alphas, ref_betas = get_ref_beta(input_sigmas)
        for i, (sub_alpha, sub_beta, ref_sub_alpha, ref_sub_beta) in enumerate(
            zip(alphas, betas, ref_alphas, ref_betas)
        ):
            for j, (alpha, beta, ref_alpha, ref_beta) in enumerate(
                zip(sub_alpha, sub_beta, ref_sub_alpha, ref_sub_beta)
            ):
                if prob_masks[i][j]:
                    kl_divergences[i].append(torch.tensor(0.0))
                else:
                    ref_distribution = torch.distributions.Beta(ref_alpha, ref_beta) if self.relative else torch.distributions.Beta(1.4, 11.2)
                    kl_divergences[i].append(
                        torch.distributions.kl_divergence(torch.distributions.Beta(alpha, beta), ref_distribution)
                    )
        return torch.tensor(kl_divergences)

    def subset_inputs(self, inputs, micro_batch_inds):
        subset_inputs = {}
        for key, value in inputs.items():
            if isinstance(value, torch.Tensor):
                subset_inputs[key] = value[micro_batch_inds]
            elif isinstance(value, list):
                subset_inputs[key] = [value[i] for i in micro_batch_inds]
            elif isinstance(value, float) or isinstance(value, int) or value is None:
                subset_inputs[key] = value
            else:
                raise ValueError(f"Unsupported input type: {type(value)}")
        return subset_inputs

    def subset_outputs(self, outputs, micro_batch_inds):
        subset_outputs = {}
        for key, value in outputs.items():
            if isinstance(value, torch.Tensor):
                subset_outputs[key] = value[micro_batch_inds]
            elif isinstance(value, list):
                subset_outputs[key] = [value[i] for i in micro_batch_inds]
            elif isinstance(value, dict):
                sub_dict = {}
                for k, v in value.items():
                    if isinstance(v, torch.Tensor):
                        sub_dict[k] = v[micro_batch_inds]
                    else:
                        ValueError(f"Unsupported output type: {type(v)}")
                subset_outputs[key] = sub_dict
            else:
                raise ValueError(f"Unsupported output type: {type(value)}")
        return subset_outputs


if __name__ == "__main__":
    model = SD3PredictNextTimeStepModelRLOOWrapper(
        "stabilityai/stable-diffusion-3-medium-diffusers",
    ).cuda()
    inputs = {
        "prompt": [
            "a cat is holding a paper with 'hello world'",
            "a dog is holding a paper with 'hello world'",
        ],
        "max_inference_steps": 28,
        "guidance_scale": 7.0,
    }
    inputs = model.rloo_repeat(inputs, 2)
    outputs = model.sample(
        inputs=inputs,
    )
    kl_divergence = model.kl_divergence(
        outputs=outputs,
    )
    logprobs = model.logprobs(
        inputs=inputs,
        outputs=outputs,
    )
    rewards = model.reward(
        inputs=inputs,
        outputs=outputs,
    )
