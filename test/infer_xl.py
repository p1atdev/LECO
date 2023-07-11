# ref:
# - https://github.com/huggingface/diffusers/blob/f74d5e1c2f7caa645e95ac296b6a252276e6d185/src/diffusers/pipelines/stable_diffusion_xl/pipeline_stable_diffusion_xl.py#L655-L821

from typing import List, Optional

import gc

import torch
import torchvision

from diffusers import AutoencoderKL

import train_util
import model_util
import config_util


DEVICE_CUDA = torch.device("cuda:0")
NUM_IMAGES_PER_PROMPT = 1

SDXL_NOISE_OFFSET = 0.0357

DDIM_STEPS = 30

prompt = "a photo of lemonade"
negative_prompt = ""

precision = "fp16"


def flush():
    torch.cuda.empty_cache()
    gc.collect()


@torch.no_grad()
def train():
    weight_dtype = config_util.parse_precision(precision)

    (
        tokenizers,
        text_encoders,
        unet,
        noise_scheduler,
    ) = model_util.load_models_xl(
        "stabilityai/stable-diffusion-xl-base-0.9",
        scheduler_name="ddim",
    )

    vae = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae").to(
        DEVICE_CUDA, dtype=torch.float32
    )
    vae.eval()

    for text_encoder in text_encoders:
        text_encoder.to(DEVICE_CUDA, dtype=weight_dtype)
        text_encoder.eval()

    unet.to(DEVICE_CUDA, dtype=weight_dtype)
    unet.enable_xformers_memory_efficient_attention()
    unet.eval()

    prompt_embeds = train_util.encode_prompts_xl(
        tokenizers,
        text_encoders,
        [prompt],
        num_images_per_prompt=NUM_IMAGES_PER_PROMPT,
    )
    negative_prompt_embeds = train_util.encode_prompts_xl(
        tokenizers,
        text_encoders,
        [negative_prompt],
        num_images_per_prompt=NUM_IMAGES_PER_PROMPT,
    )

    for tokenizer, text_encoder in zip(tokenizers, text_encoders):
        del tokenizer, text_encoder

    flush()

    noise_scheduler.set_timesteps(DDIM_STEPS, device=DEVICE_CUDA)

    height, width = 1024, 768

    latents = train_util.get_initial_latents(noise_scheduler, 1, height, width, 1)
    latents = latents * noise_scheduler.init_noise_sigma
    latents = train_util.apply_noise_offset(latents, SDXL_NOISE_OFFSET)
    latents = latents.to(DEVICE_CUDA, dtype=weight_dtype)

    add_time_ids = train_util.get_add_time_ids(
        height,
        width,
        dynamic_crops=False,
        dtype=weight_dtype,
    ).to(DEVICE_CUDA, dtype=weight_dtype)

    # ちょっとデノイズされれたものが返る
    latents = train_util.diffusion_xl(
        unet,
        noise_scheduler,
        latents,  # 単純なノイズのlatentsを渡す
        text_embeddings=train_util.concat_embeddings(
            negative_prompt_embeds[0],
            prompt_embeds[0],
            1,
        ),
        add_text_embeddings=train_util.concat_embeddings(
            negative_prompt_embeds[1],
            prompt_embeds[1],
            1,
        ),
        add_time_ids=train_util.concat_embeddings(add_time_ids, add_time_ids, 1),
        total_timesteps=DDIM_STEPS,
        start_timesteps=0,
        guidance_scale=7,
    )

    vae.post_quant_conv.to(latents.dtype)
    vae.decoder.conv_in.to(latents.dtype)
    vae.decoder.mid_block.to(latents.dtype)

    image_tensor = vae.decode(
        latents / vae.config.scaling_factor,
    ).sample[0]

    image = torchvision.transforms.functional.to_pil_image(
        image_tensor.cpu(),
    )

    image.save("output.png")

    flush()

    print("Done.")


def main():
    train()


if __name__ == "__main__":
    main()
