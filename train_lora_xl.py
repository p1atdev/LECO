# ref:
# - https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py#L566
# - https://huggingface.co/spaces/baulab/Erasing-Concepts-In-Diffusion/blob/main/train.py

from typing import List, Optional
import argparse
from pathlib import Path
import gc

import torch
from tqdm import tqdm


from lora import LoRANetwork, DEFAULT_TARGET_REPLACE, UNET_TARGET_REPLACE_MODULE_CONV
import train_util
import model_util
import prompt_util
from prompt_util import PromptCache, PromptPair, PromptSettings
import debug_util
import config_util
from config_util import RootConfig

import wandb

DEVICE_CUDA = torch.device("cuda:0")
NUM_IMAGES_PER_PROMPT = 1


def flush():
    torch.cuda.empty_cache()
    gc.collect()


def train(
    config: RootConfig,
    prompts: list[PromptSettings],
):
    metadata = {
        "prompts": ",".join([prompt.json() for prompt in prompts]),
        "config": config.json(),
    }
    save_path = Path(config.save.path)

    modules = DEFAULT_TARGET_REPLACE
    if config.network.type == "c3lier":
        modules += UNET_TARGET_REPLACE_MODULE_CONV

    if config.logging.verbose:
        print(metadata)

    if config.logging.use_wandb:
        wandb.init(project=f"LECO_{config.save.name}", config=metadata)

    weight_dtype = config_util.parse_precision(config.train.precision)
    save_weight_dtype = config_util.parse_precision(config.train.precision)

    (
        tokenizers,
        text_encoders,
        unet,
        scheduler,
    ) = model_util.load_models_xl(
        config.pretrained_model.name_or_path,
        scheduler_name=config.train.noise_scheduler,
    )

    for text_encoder in text_encoders:
        text_encoder.to(DEVICE_CUDA, dtype=weight_dtype)
        text_encoder.requires_grad_(False)
        text_encoder.eval()

    unet.to(DEVICE_CUDA, dtype=weight_dtype)
    if config.other.use_xformers:
        unet.enable_xformers_memory_efficient_attention()
    unet.requires_grad_(False)
    unet.eval()

    network = LoRANetwork(
        unet,
        rank=config.network.rank,
        multiplier=1.0,
        alpha=config.network.alpha,
        train_method=config.network.training_method,
    ).to(DEVICE_CUDA, dtype=weight_dtype)

    optimizer_module = train_util.get_optimizer(config.train.optimizer)
    optimizer = optimizer_module(network.prepare_optimizer_params(), lr=config.train.lr)
    lr_scheduler = train_util.get_lr_scheduler(
        config.train.lr_scheduler,
        optimizer,
        max_iterations=config.train.iterations,
        lr_min=config.train.lr / 100,
    )
    criteria = torch.nn.MSELoss()

    print("Prompts")
    for settings in prompts:
        print(settings)

    # debug
    debug_util.check_requires_grad(network)
    debug_util.check_training_mode(network)

    cache = PromptCache()
    prompt_pairs: list[PromptPair] = []

    with torch.no_grad():
        for settings in prompts:
            print(settings)
            for prompt in [
                settings.target,
                settings.positive,
                settings.neutral,
                settings.unconditional,
            ]:
                if cache[prompt] == None:
                    cache[prompt] = train_util.encode_prompts_xl(
                        tokenizers,
                        text_encoders,
                        [prompt],
                        num_images_per_prompt=NUM_IMAGES_PER_PROMPT,
                    )

            prompt_pairs.append(
                PromptPair(
                    criteria,
                    cache[settings.target],
                    cache[settings.positive],
                    cache[settings.unconditional],
                    cache[settings.neutral],
                    settings,
                )
            )

    for tokenizer, text_encoder in zip(tokenizers, text_encoders):
        del tokenizer, text_encoder

    flush()

    pbar = tqdm(range(config.train.iterations))

    loss = None

    for i in pbar:
        with torch.no_grad():
            scheduler.set_timesteps(
                config.train.max_denoising_steps, device=DEVICE_CUDA
            )

            optimizer.zero_grad()

            prompt_pair: PromptPair = prompt_pairs[
                torch.randint(0, len(prompt_pairs), (1,)).item()
            ]

            # 1 ~ 49 からランダム
            timesteps_to = torch.randint(
                1, config.train.max_denoising_steps, (1,)
            ).item()

            height, width = prompt_pair.resolution, prompt_pair.resolution
            if prompt_pair.dynamic_resolution:
                height, width = train_util.get_random_resolution_in_bucket(
                    prompt_pair.resolution
                )

            if config.logging.verbose:
                print("gudance_scale:", prompt_pair.guidance_scale)
                print("resolution:", prompt_pair.resolution)
                print("dynamic_resolution:", prompt_pair.dynamic_resolution)
                if prompt_pair.dynamic_resolution:
                    print("bucketed resolution:", (height, width))
                print("batch_size:", prompt_pair.batch_size)

            latents = train_util.get_initial_latents(
                scheduler, prompt_pair.batch_size, height, width, 1
            ).to(DEVICE_CUDA, dtype=weight_dtype)

            add_time_ids = (
                train_util.get_add_time_ids(
                    height,
                    width,
                    dynamic_crops=prompt_pair.dynamic_crops,
                    dtype=weight_dtype,
                )
                .to(DEVICE_CUDA, dtype=weight_dtype)
                .repeat(prompt_pair.batch_size * NUM_IMAGES_PER_PROMPT, 1)
            )

            with network:
                # ちょっとデノイズされれたものが返る
                denoised_latents = train_util.diffusion_xl(
                    unet,
                    scheduler,
                    latents,  # 単純なノイズのlatentsを渡す
                    text_embeddings=train_util.concat_embeddings(
                        prompt_pair.unconditional[0],
                        prompt_pair.target[0],
                        prompt_pair.batch_size,
                    ),
                    add_text_embeddings=train_util.concat_embeddings(
                        prompt_pair.unconditional[1],
                        prompt_pair.target[1],
                        prompt_pair.batch_size,
                    ),
                    add_time_ids=add_time_ids,
                    start_timesteps=0,
                    total_timesteps=timesteps_to,
                    guidance_scale=3,
                )

            scheduler.set_timesteps(1000)

            current_timestep = scheduler.timesteps[
                int(timesteps_to * 1000 / config.train.max_denoising_steps)
            ]

            # with network: の外では空のLoRAのみが有効になる
            positive_latents = train_util.predict_noise_xl(
                unet,
                scheduler,
                current_timestep,
                denoised_latents,
                text_embeddings=train_util.concat_embeddings(
                    prompt_pair.unconditional[0],
                    prompt_pair.positive[0],
                    prompt_pair.batch_size,
                ),
                add_text_embeddings=train_util.concat_embeddings(
                    prompt_pair.unconditional[1],
                    prompt_pair.positive[1],
                    prompt_pair.batch_size,
                ),
                add_time_ids=add_time_ids,
                guidance_scale=1,
            ).to("cpu", dtype=torch.float32)
            neutral_latents = train_util.predict_noise(
                unet,
                scheduler,
                current_timestep,
                denoised_latents,
                text_embeddings=train_util.concat_embeddings(
                    prompt_pair.unconditional[0],
                    prompt_pair.neutral[0],
                    prompt_pair.batch_size,
                ),
                add_text_embeddings=train_util.concat_embeddings(
                    prompt_pair.unconditional[1],
                    prompt_pair.neutral[1],
                    prompt_pair.batch_size,
                ),
                add_time_ids=add_time_ids,
                guidance_scale=1,
            ).to("cpu", dtype=torch.float32)
            unconditional_latents = train_util.predict_noise_xl(
                unet,
                scheduler,
                current_timestep,
                denoised_latents,
                text_embeddings=train_util.concat_embeddings(
                    prompt_pair.unconditional[0],
                    prompt_pair.unconditional[0],
                    prompt_pair.batch_size,
                ),
                add_text_embeddings=train_util.concat_embeddings(
                    prompt_pair.unconditional[1],
                    prompt_pair.unconditional[1],
                    prompt_pair.batch_size,
                ),
                add_time_ids=add_time_ids,
                guidance_scale=1,
            ).to("cpu", dtype=torch.float32)

            if config.logging.verbose:
                print("positive_latents:", positive_latents[0, 0, :5, :5])
                print("neutral_latents:", neutral_latents[0, 0, :5, :5])
                print("unconditional_latents:", unconditional_latents[0, 0, :5, :5])

        with network:
            target_latents = train_util.predict_noise_xl(
                unet,
                scheduler,
                current_timestep,
                denoised_latents,
                text_embeddings=train_util.concat_embeddings(
                    prompt_pair.unconditional[0],
                    prompt_pair.target[1],
                    prompt_pair.batch_size,
                ),
                add_text_embeddings=train_util.concat_embeddings(
                    prompt_pair.unconditional[1],
                    prompt_pair.target[1],
                    prompt_pair.batch_size,
                ),
                add_time_ids=add_time_ids,
                guidance_scale=1,
            ).to("cpu", dtype=torch.float32)

            if config.logging.verbose:
                print("target_latents:", target_latents[0, 0, :5, :5])

        positive_latents.requires_grad = False
        neutral_latents.requires_grad = False
        unconditional_latents.requires_grad = False

        loss = prompt_pair.loss(
            target_latents=target_latents,
            positive_latents=positive_latents,
            neutral_latents=neutral_latents,
            unconditional_latents=unconditional_latents,
        )

        # 1000倍しないとずっと0.000...になってしまって見た目的に面白くない
        pbar.set_description(f"Loss*1k: {loss.item()*1000:.4f}")
        if config.logging.use_wandb:
            wandb.log(
                {"loss": loss, "iteration": i, "lr": lr_scheduler.get_last_lr()[0]}
            )

        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        del (
            positive_latents,
            neutral_latents,
            unconditional_latents,
            target_latents,
            latents,
        )
        flush()

        if (
            i % config.save.per_steps == 0
            and i != 0
            and i != config.train.iterations - 1
        ):
            print("Saving...")
            save_path.mkdir(parents=True, exist_ok=True)
            network.save_weights(
                save_path / f"{config.save.name}_{i}steps.safetensors",
                dtype=save_weight_dtype,
            )

    print("Saving...")
    save_path.mkdir(parents=True, exist_ok=True)
    network.save_weights(
        save_path / f"{config.save.name}_last.safetensors",
        dtype=save_weight_dtype,
    )

    del (
        unet,
        scheduler,
        loss,
        optimizer,
        network,
    )

    flush()

    print("Done.")


def main(args):
    config_file = args.config_file

    config = config_util.load_config_from_yaml(config_file)
    prompts = prompt_util.load_prompts_from_yaml(config.prompts_file)

    train(config, prompts)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_file",
        required=True,
        help="Config file for training.",
    )

    args = parser.parse_args()

    main(args)
