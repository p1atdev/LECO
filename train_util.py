from typing import Optional

import torch

from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import UNet2DConditionModel, SchedulerMixin

from prompt_util import PromptPair

from tqdm import tqdm

UNET_IN_CHANNELS = 4  # Stable Diffusion の in_channels は 4 で固定


def get_random_noise(
    batch_size: int, height: int, width: int, generator: torch.Generator = None
) -> torch.Tensor:
    return torch.randn(
        (
            batch_size,
            UNET_IN_CHANNELS,
            height // 8,  # 縦と横これであってるのかわからないけど、どっちにしろ大きな問題は発生しないのでこれでいいや
            width // 8,
        ),
        generator=generator,
    )


def get_initial_latents(
    scheduler: SchedulerMixin,
    n_imgs: int,
    height: int,
    width: int,
    n_prompts: int,
    generator=None,
) -> torch.Tensor:
    noise = get_random_noise(n_imgs, height, width, generator=generator).repeat(
        n_prompts, 1, 1, 1
    )

    latents = noise * scheduler.init_noise_sigma

    return latents


def text_tokenize(tokenizer: CLIPTokenizer, prompts: list[str]):
    return tokenizer(
        prompts,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )


def text_encode(text_encoder: CLIPTextModel, tokens):
    return text_encoder(tokens.input_ids.to(text_encoder.device))[0]


def encode_prompts(
    tokenizer: CLIPTokenizer,
    text_encoder: CLIPTextModel,
    prompts: list[str],
):
    text_tokens = text_tokenize(tokenizer, prompts)
    text_embeddings = text_encode(text_encoder, text_tokens)

    return text_embeddings


def concat_embeddings(
    unconditional: torch.FloatTensor,
    conditional: torch.FloatTensor,
    n_imgs: int,
):
    return torch.cat([unconditional, conditional]).repeat_interleave(n_imgs, dim=0)


def get_text_embeddings(
    tokenizer: CLIPTokenizer,
    text_encoder: CLIPTextModel,
    prompts: list[str],
    n_imgs: int,
):
    text_tokens = text_tokenize(tokenizer, prompts)
    text_embeddings = text_encode(text_encoder, text_tokens)

    unconditional_tokens = text_tokenize(tokenizer, [""] * len(prompts))
    unconditional_embeddings = text_encode(text_encoder, unconditional_tokens)

    # CFG
    text_embeddings = torch.cat(
        [unconditional_embeddings, text_embeddings]
    ).repeat_interleave(n_imgs, dim=0)

    return text_embeddings


# ref: https://github.com/huggingface/diffusers/blob/0bab447670f47c28df60fbd2f6a0f833f75a16f5/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py#L721
def predict_noise(
    unet: UNet2DConditionModel,
    scheduler: SchedulerMixin,
    timestep: int,  # 現在のタイムステップ
    latents: torch.Tensor,
    text_embeddings: torch.Tensor,  # uncond な text embed と cond な text embed を結合したもの
    guidance_scale=7.5,
) -> torch.Tensor:
    # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
    latent_model_input = torch.cat([latents] * 2)

    latent_model_input = scheduler.scale_model_input(latent_model_input, timestep)

    # predict the noise residual
    noise_pred = unet(
        latent_model_input,
        timestep,
        encoder_hidden_states=text_embeddings,
    ).sample

    # perform guidance
    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
    guided_target = noise_pred_uncond + guidance_scale * (
        noise_pred_text - noise_pred_uncond
    )

    return guided_target


# ref: https://github.com/huggingface/diffusers/blob/0bab447670f47c28df60fbd2f6a0f833f75a16f5/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py#L746
@torch.no_grad()
def diffusion(
    unet: UNet2DConditionModel,
    scheduler: SchedulerMixin,
    latents: torch.FloatTensor,  # ただのノイズだけのlatents
    text_embeddings: torch.FloatTensor,
    total_timesteps: int = 1000,
    start_timesteps=0,
    **kwargs,
):
    # latents_steps = []

    for timestep in tqdm(scheduler.timesteps[start_timesteps:total_timesteps]):
        noise_pred = predict_noise(
            unet, scheduler, timestep, latents, text_embeddings, **kwargs
        )

        # compute the previous noisy sample x_t -> x_t-1
        latents = scheduler.step(noise_pred, timestep, latents).prev_sample

    # return latents_steps
    return latents


def get_optimizer(name: str):
    name = name.lower()

    if name.startswith("dadapt"):
        import dadaptation

        if name == "dadaptadam":
            return dadaptation.DAdaptAdam
        elif name == "dadaptLion":
            return dadaptation.DAdaptLion
        else:
            raise ValueError("DAdapt optimizer must be dadaptadam or dadaptlion")

    elif name.endswith("8bit"):  # 検証してない
        import bitsandbytes as bnb

        if name == "adam8bit":
            return bnb.optim.Adam8bit
        elif name == "lion8bit":
            return bnb.optim.Lion8bit
        else:
            raise ValueError("8bit optimizer must be adam8bit or lion8bit")

    else:
        if name == "adam":
            return torch.optim.Adam
        elif name == "adamw":
            return torch.optim.AdamW
        elif name == "lion":
            from lion_pytorch import Lion

            return Lion
        else:
            raise ValueError("Optimizer must be adam, adamw or lion")


def get_lr_scheduler(
    name: Optional[str],
    optimizer: torch.optim.Optimizer,
    max_iterations: Optional[int],
    lr_min: Optional[float],
    **kwargs,
):
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max_iterations, eta_min=lr_min, **kwargs
        )
    elif name == "cosine_with_restarts":
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=max_iterations // 10, T_mult=2, eta_min=lr_min, **kwargs
        )
    elif name == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=max_iterations // 100, gamma=0.999, **kwargs
        )
    elif name == "constant":
        return torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1, **kwargs)
    elif name == "linear":
        return torch.optim.lr_scheduler.LinearLR(
            optimizer, factor=0.5, total_iters=max_iterations // 100, **kwargs
        )
    else:
        raise ValueError(
            "Scheduler must be cosine, cosine_with_restarts, step, linear or constant"
        )


def get_random_resolution_in_bucket(bucket_resolution: int = 512) -> tuple[int, int]:
    max_resolution = bucket_resolution
    min_resolution = bucket_resolution // 2

    step = 64

    min_step = min_resolution // step
    max_step = max_resolution // step

    height = torch.randint(min_step, max_step, (1,)).item() * step
    width = torch.randint(min_step, max_step, (1,)).item() * step

    return height, width
