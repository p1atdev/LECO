from typing import Literal

from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import UNet2DConditionModel, SchedulerMixin
from diffusers.schedulers import (
    DDIMScheduler,
    DDPMScheduler,
    LMSDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
)


TOKENIZER_V1_MODEL_NAME = "CompVis/stable-diffusion-v1-4"
TOKENIZER_V2_MODEL_NAME = "stabilityai/stable-diffusion-2-1"

AVAILABLE_SCHEDULERS = Literal["ddim", "ddpm", "lms", "euler_a"]


def load_models(
    pretrained_model_name_or_path: str,
    scheduler_name: AVAILABLE_SCHEDULERS,
    v2: bool = False,
    v_pred: bool = False,
) -> tuple[CLIPTokenizer, CLIPTextModel, UNet2DConditionModel, SchedulerMixin]:
    # VAE はいらない

    if v2:
        tokenizer = CLIPTokenizer.from_pretrained(
            TOKENIZER_V2_MODEL_NAME, subfolder="tokenizer"
        )
    else:
        tokenizer = CLIPTokenizer.from_pretrained(
            TOKENIZER_V1_MODEL_NAME, subfolder="tokenizer"
        )

    text_encoder = CLIPTextModel.from_pretrained(
        pretrained_model_name_or_path, subfolder="text_encoder"
    )

    unet = UNet2DConditionModel.from_pretrained(
        pretrained_model_name_or_path, subfolder="unet"
    )

    scheduler = create_noise_scheduler(
        scheduler_name,
        prediction_type="v_prediction" if v_pred else "epsilon",
    )

    return tokenizer, text_encoder, unet, scheduler


def create_noise_scheduler(
    scheduler_name: AVAILABLE_SCHEDULERS = "ddpm",
    prediction_type: Literal["epsilon", "v_prediction"] = "epsilon",
) -> SchedulerMixin:
    # 正直、どれがいいのかわからない。元の実装だとDDIMとDDPMとLMSを選べたのだけど、どれがいいのかわからぬ。

    name = scheduler_name.lower().replace(" ", "_")
    if name == "ddim":
        # https://huggingface.co/docs/diffusers/v0.17.1/en/api/schedulers/ddim
        scheduler = DDIMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            num_train_timesteps=1000,
            clip_sample=False,
            prediction_type=prediction_type,  # これでいいの？
        )
    elif name == "ddpm":
        # https://huggingface.co/docs/diffusers/v0.17.1/en/api/schedulers/ddpm
        scheduler = DDPMScheduler(
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
