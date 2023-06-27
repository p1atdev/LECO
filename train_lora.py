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
DDIM_STEPS = 50


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

    if config.logging.use_wandb:
        wandb.init(project=f"LECO_{config.save.name}")
        wandb.config = metadata

    weight_dtype = config_util.parse_precision(config.train.precision)
    save_weight_dtype = config_util.parse_precision(config.train.precision)

    tokenizer, text_encoder, unet, scheduler = model_util.load_models(
        config.pretrained_model.name_or_path,
        scheduler_name=config.train.noise_scheduler,
        v2=config.pretrained_model.v2,
        v_pred=config.pretrained_model.v_pred,
    )

    text_encoder.to(DEVICE_CUDA, dtype=weight_dtype)
    text_encoder.eval()

    unet.to(DEVICE_CUDA, dtype=weight_dtype)
    unet.enable_xformers_memory_efficient_attention()
    unet.requires_grad_(False)
    unet.eval()

    network = LoRANetwork(
        unet, rank=config.network.rank, multiplier=1.0, alpha=config.network.alpha
    ).to(DEVICE_CUDA, dtype=weight_dtype)

    optimizer = torch.optim.AdamW(
        network.prepare_optimizer_params(), lr=config.train.lr
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
            for prompt in [
                settings.target,
                settings.positive,
                settings.neutral,
                settings.unconditional,
            ]:
                if cache[prompt] == None:
                    cache[prompt] = train_util.encode_prompts(
                        tokenizer, text_encoder, [prompt]
                    )

            prompt_pairs.append(
                PromptPair(
                    criteria,
                    cache[settings.target],
                    cache[settings.positive],
                    cache[settings.unconditional],
                    cache[settings.neutral],
                    settings.guidance_scale,
                    settings.action,
                )
            )

    del tokenizer
    del text_encoder

    flush()

    pbar = tqdm(range(config.train.iterations))

    n_imgs = config.train.batch_size

    for i in pbar:
        with torch.no_grad():
            scheduler.set_timesteps(DDIM_STEPS, device=DEVICE_CUDA)

            optimizer.zero_grad()

            prompt_pair: PromptPair = prompt_pairs[
                torch.randint(0, len(prompt_pairs), (1,)).item()
            ]

            # 1 ~ 49 からランダム
            timesteps_to = torch.randint(1, DDIM_STEPS, (1,)).item()

            latents = train_util.get_initial_latents(
                scheduler, n_imgs, config.train.resolution, 1
            ).to(DEVICE_CUDA, dtype=weight_dtype)

            with network:
                # ちょっとデノイズされれたものが返る
                denoised_latents = train_util.diffusion(
                    unet,
                    scheduler,
                    latents,  # 単純なノイズのlatentsを渡す
                    train_util.concat_embeddings(
                        prompt_pair.unconditional, prompt_pair.target, n_imgs
                    ),
                    start_timesteps=0,
                    total_timesteps=timesteps_to,
                    guidance_scale=3,
                )

            scheduler.set_timesteps(1000)

            current_timestep = scheduler.timesteps[
                int(timesteps_to * 1000 / DDIM_STEPS)
            ]

            # with network: の外では空のLoRAのみが有効になる
            positive_latents = train_util.predict_noise(
                unet,
                scheduler,
                current_timestep,
                denoised_latents,
                train_util.concat_embeddings(
                    prompt_pair.unconditional, prompt_pair.positive, n_imgs
                ),
                guidance_scale=1,
            ).to("cpu", dtype=torch.float32)
            neutral_latents = train_util.predict_noise(
                unet,
                scheduler,
                current_timestep,
                denoised_latents,
                train_util.concat_embeddings(
                    prompt_pair.unconditional, prompt_pair.neutral, n_imgs
                ),
                guidance_scale=1,
            ).to("cpu", dtype=torch.float32)
            unconditional_latents = train_util.predict_noise(
                unet,
                scheduler,
                current_timestep,
                denoised_latents,
                train_util.concat_embeddings(
                    prompt_pair.unconditional, prompt_pair.unconditional, n_imgs
                ),
                guidance_scale=1,
            ).to("cpu", dtype=torch.float32)

            if config.logging.verbose:
                print("positive_latents:", positive_latents[0, 0, :5, :5])
                print("neutral_latents:", neutral_latents[0, 0, :5, :5])
                print("unconditional_latents:", unconditional_latents[0, 0, :5, :5])

        with network:
            target_latents = train_util.predict_noise(
                unet,
                scheduler,
                current_timestep,
                denoised_latents,
                train_util.concat_embeddings(
                    prompt_pair.unconditional, prompt_pair.target, n_imgs
                ),
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
            wandb.log({"loss": loss, "iteration": i})

        loss.backward()
        optimizer.step()

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
