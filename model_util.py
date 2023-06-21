from typing import Literal

import torch
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import UNet2DConditionModel, SchedulerMixin
from diffusers.schedulers import (
    DDIMScheduler,
    DDPMScheduler,
    LMSDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
)

from tqdm import tqdm

TOKENIZER_V1_MODEL_NAME = "CompVis/stable-diffusion-v1-4"
TOKENIZER_V2_MODEL_NAME = "stabilityai/stable-diffusion-2-1"


UNET_IN_CHANNELS = 4  # Stable Diffusion の in_channels は 4 で固定


def load_models(
    pretrained_model_name_or_path: str, v2: bool = False, v_pred: bool = False
) -> tuple[CLIPTokenizer, CLIPTextModel, UNet2DConditionModel]:
    # VAE はいらない

    if v2:
        tokenizer = CLIPTokenizer.from_pretrained(TOKENIZER_V2_MODEL_NAME)
    else:
        tokenizer = CLIPTokenizer.from_pretrained(TOKENIZER_V1_MODEL_NAME)

    text_encoder = CLIPTextModel.from_pretrained(
        pretrained_model_name_or_path, subfolder="text_encoder"
    )

    unet = UNet2DConditionModel.from_pretrained(
        pretrained_model_name_or_path, subfolder="unet"
    )

    return tokenizer, text_encoder, unet


def create_noise_scheduler(
    name: Literal["ddim", "ddpm", "lms", "euler_a"] = "ddim",
    prediction_type: Literal["epsilon", "v_prediction"] = "epsilon",
) -> SchedulerMixin:
    # 正直、どれがいいのかわからない。元の実装だとDDIMとDDPMとLMSを選べたのだけど、どれがいいのかわからぬ。

    name = name.lower().replace(" ", "_")
    if name == "ddim":
        # https://huggingface.co/docs/diffusers/v0.17.1/en/api/schedulers/ddim
        scheduler = DDIMScheduler.from_pretrained(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            num_train_timesteps=1000,
            clip_sample=False,
            prediction_type=prediction_type,  # これでいいの？
        )
    elif name == "ddpm":
        # https://huggingface.co/docs/diffusers/v0.17.1/en/api/schedulers/ddpm
        scheduler = DDPMScheduler.from_pretrained(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            num_train_timesteps=1000,
            clip_sample=False,
            prediction_type=prediction_type,
        )
    elif name == "lms":
        # https://huggingface.co/docs/diffusers/v0.17.1/en/api/schedulers/lms_discrete
        scheduler = LMSDiscreteScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            num_train_timesteps=1000,
            prediction_type=prediction_type,
        )
    elif name == "euler_a":
        # https://huggingface.co/docs/diffusers/v0.17.1/en/api/schedulers/euler_ancestral
        scheduler = EulerAncestralDiscreteScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            num_train_timesteps=1000,
            prediction_type=prediction_type,
        )
    else:
        raise ValueError(f"Unknown scheduler name: {name}")

    return scheduler


def get_random_noise(batch_size: int, img_size: int, generator: torch.Generator = None):
    return torch.randn(
        (
            batch_size,
            UNET_IN_CHANNELS,
            img_size // 8,
            img_size // 8,
        ),
        generator=generator,
    )


def get_initial_latents(
    scheduler: SchedulerMixin,
    n_imgs: int,
    img_size: int,
    n_prompts: int,
    generator=None,
):
    noise = get_random_noise(n_imgs, img_size, generator=generator).repeat(
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
    return text_encoder(tokens.input_ids)[0]


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

    text_embeddings = torch.cat(
        [unconditional_embeddings, text_embeddings]
    ).repeat_interleave(n_imgs, dim=0)

    return text_embeddings


def predict_noise(
    unet: UNet2DConditionModel,
    scheduler: SchedulerMixin,
    iteration: int,
    latents: torch.Tensor,
    text_embeddings: torch.Tensor,
    guidance_scale=7.5,
    # v_pred=False,
):
    # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
    latents = torch.cat([latents] * 2)

    latents = scheduler.scale_model_input(latents, scheduler.timesteps[iteration])

    # predict the noise residual
    noise_prediction = unet(
        latents,
        scheduler.timesteps[iteration],
        encoder_hidden_states=text_embeddings,
    ).sample

    # perform guidance
    noise_prediction_uncond, noise_prediction_text = noise_prediction.chunk(2)
    guided_target = noise_prediction_uncond + guidance_scale * (
        noise_prediction_text - noise_prediction_uncond
    )

    return guided_target


@torch.no_grad()
def diffusion(
    unet: UNet2DConditionModel,
    scheduler: SchedulerMixin,
    latents,
    text_embeddings,
    end_iteration=1000,
    start_iteration=0,
    return_steps=False,
    **kwargs,
):
    latents_steps = []

    for iteration in tqdm(range(start_iteration, end_iteration)):
        noise_pred = predict_noise(
            unet, scheduler, iteration, latents, text_embeddings, **kwargs
        )

        # compute the previous noisy sample x_t -> x_t-1
        output = scheduler.step(noise_pred, scheduler.timesteps[iteration], latents)

        latents = output.prev_sample

        if return_steps or iteration == end_iteration - 1:
            if return_steps:
                latents_steps.append(latents.cpu())
            else:
                latents_steps.append(latents)

    return latents_steps
