import numpy as np
import torch
from torchvision.transforms.functional import to_pil_image

from PIL import Image
from pathlib import Path
from diffusers import UNet2DConditionModel, SchedulerMixin, AutoencoderKL

import train_util
from prompt_util import PromptEmbedsXL

DEVICE_CUDA = torch.device("cuda:0")


@torch.no_grad()
# 画像の配列を -1~1 から 0~1 に変換する
def denormalize(image_tensor: torch.FloatTensor) -> torch.FloatTensor:
    return (image_tensor / 2 + 0.5).clamp(0, 1)


@torch.no_grad()
def sample_image(
    unet: UNet2DConditionModel,
    scheduler: SchedulerMixin,
    vae: AutoencoderKL,
    positive: torch.FloatTensor,
    negative: torch.FloatTensor,
    width: int = 512,
    height: int = 512,
    num_inference_steps: int = 20,
    cfg_scale: float = 0.7,
    seed: int = -1,
    weight_dtype: torch.dtype = torch.float16,
) -> Image.Image:
    generator = torch.Generator()
    if seed >= 0:
        generator.manual_seed(seed)

    scheduler.set_timesteps(num_inference_steps, device=DEVICE_CUDA)

    latents = train_util.get_initial_latents(
        scheduler, 1, height, width, 1, generator=generator
    ).to(DEVICE_CUDA, dtype=weight_dtype)

    denoised = train_util.diffusion(
        unet,
        scheduler,
        latents,
        text_embeddings=train_util.concat_embeddings(negative, positive, 1),
        total_timesteps=num_inference_steps,
        start_timesteps=0,
        guidance_scale=cfg_scale,
    ).to(torch.float16)

    image_tensor = vae.decode(
        denoised / vae.config.scaling_factor,
    ).sample
    images = denormalize(image_tensor)

    return to_pil(images)


def to_pil(image_tensor: torch.FloatTensor):
    return [to_pil_image(img) for img in image_tensor]


@torch.no_grad()
def sample_image_xl(
    unet: UNet2DConditionModel,
    scheduler: SchedulerMixin,
    vae: AutoencoderKL,
    positive: PromptEmbedsXL,
    negative: PromptEmbedsXL,
    width: int = 512,
    height: int = 512,
    num_inference_steps: int = 20,
    cfg_scale: float = 0.7,
    seed: int = -1,
    weight_dtype: torch.dtype = torch.float16,
) -> np.ndarray:
    generator = torch.Generator(device=DEVICE_CUDA)
    if seed >= 0:
        generator.manual_seed(seed)

    scheduler.set_timesteps(num_inference_steps, device=DEVICE_CUDA)

    latents = train_util.get_initial_latents(
        scheduler, 1, height, width, 1, generator=generator
    ).to(DEVICE_CUDA, dtype=weight_dtype)

    add_time_ids = train_util.get_add_time_ids(
        height, width, dynamic_crops=False, dtype=weight_dtype
    )

    denoised = train_util.diffusion_xl(
        unet,
        scheduler,
        latents,
        text_embeddings=train_util.concat_embeddings(
            negative.text_embeds, positive.text_embeds, 1
        ),
        add_text_embeddings=train_util.concat_embeddings(
            negative.pooled_embeds, positive.pooled_embeds, 1
        ),
        add_time_ids=train_util.concat_embeddings(add_time_ids, add_time_ids, 1),
        total_timesteps=num_inference_steps,
        start_timesteps=0,
        guidance_scale=cfg_scale,
    )

    image_tensor = vae.decode(
        denoised / vae.config.scaling_factor,
    ).sample
    images = denormalize(image_tensor)

    return images.detach().cpu().numpy()


def save_images(
    images_array: list[Image.Image], path: str | Path, filename_base: str = ""
):
    if isinstance(path, str):
        path = Path(path)

    path.mkdir(parents=True, exist_ok=True)

    for i, image in enumerate(images_array):
        image.save(path / f"{filename_base}_{i:04d}.png")
