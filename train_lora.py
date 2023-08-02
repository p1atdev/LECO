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
from prompt_util import PromptEmbedsCache, PromptEmbedsPair, PromptSettings
import debug_util
import config_util
from config_util import RootConfig
import sample_util
from textual_inversion import get_all_embeddings_in_folder, filter_embeddings_by_prompt

import wandb

DEVICE_CUDA = torch.device("cuda:0")


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

    weight_dtype = config_util.parse_precision(config.train.precision)
    save_weight_dtype = config_util.parse_precision(config.train.precision)

    tokenizer, text_encoder, unet, vae, noise_scheduler = model_util.load_models(
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

    if config.logging.sample is None:
        # サンプル生成しないので
        del vae
    else:
        # VAE はサンプルにしか使わないのでずっとfp16
        vae.eval()
        vae.to(DEVICE_CUDA, dtype=torch.float16)

    print("Prompts")
    for settings in prompts:
        print(settings)

    # debug
    if config.logging.verbose:
        debug_util.check_requires_grad(network)
        debug_util.check_training_mode(network)

    cache = PromptEmbedsCache()
    sample_cache = PromptEmbedsCache()
    prompt_pairs: list[PromptEmbedsPair] = []

    # ti のディレクトリが指定されたらTIを取得
    all_embs = (
        get_all_embeddings_in_folder(config.pretrained_model.embeddings_dir)
        if config.pretrained_model.embeddings_dir is not None
        else []
    )
    used_embs = set()

    with torch.no_grad():
        print("Caching prompts...")
        for settings in prompts:
            print(settings)
            for prompt in [
                settings.target,
                settings.positive,
                settings.neutral,
                settings.unconditional,
            ]:
                if cache[prompt] == None:
                    use_embs = filter_embeddings_by_prompt(
                        all_embs, prompt
                    )  # 使うものだけにフィルター
                    for emb in use_embs:
                        if emb in used_embs:
                            # remove
                            use_embs.remove(emb)
                        else:
                            used_embs.add(emb)
                    if len(use_embs) > 0:
                        print("Loading textual inversions...")
                        embs = model_util.load_textual_inversion(use_embs)
                        embs = [embs] if not isinstance(embs, list) else embs
                        # TI を適用
                        for emb in embs:
                            tokenizer, text_encoder = emb.apply_textual_inversion(
                                tokenizer, text_encoder
                            )
                            print(f"textual inversion '{emb.token}' applied")

                    cache[prompt] = train_util.encode_prompts(
                        tokenizer, text_encoder, [prompt]
                    )

            prompt_pairs.append(
                PromptEmbedsPair(
                    criteria,
                    cache[settings.target],
                    cache[settings.positive],
                    cache[settings.unconditional],
                    cache[settings.neutral],
                    settings,
                )
            )

        if config.logging.sample is not None:
            print("Logging sample images is enabled. Caching prompts...")
            for sample in config.logging.sample:
                for prompt in [
                    sample.positive,
                    sample.negative,
                ]:
                    if sample_cache[prompt] == None:
                        sample_cache[prompt] = train_util.encode_prompts(
                            tokenizer, text_encoder, [prompt]
                        )

    del tokenizer
    del text_encoder

    flush()

    if config.logging.use_wandb:
        wandb.init(project=f"LECO_{config.save.name}", config=metadata)

    pbar = tqdm(range(config.train.iterations))

    for i in pbar:
        with torch.no_grad():
            noise_scheduler.set_timesteps(
                config.train.max_denoising_steps, device=DEVICE_CUDA
            )

            optimizer.zero_grad()

            prompt_pair: PromptEmbedsPair = prompt_pairs[
                torch.randint(0, len(prompt_pairs), (1,)).item()
            ]

            # 1 ~ 49 からランダム
            timesteps_to = torch.randint(
                1, config.train.max_denoising_steps, (1,)
            ).item()

            height, width = (
                prompt_pair.resolution,
                prompt_pair.resolution,
            )
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
                noise_scheduler, prompt_pair.batch_size, height, width, 1
            )
            if config.train.noise_offset != 0.0:  # noise offset
                latents = train_util.apply_noise_offset(
                    latents, config.train.noise_offset
                )
            if config.train.pyramid_noise_discount != 0.0:  # pyramid noise discount
                latents = train_util.apply_pyramid_noise(
                    latents, config.train.pyramid_noise_discount
                )
            latents = latents.to(DEVICE_CUDA, dtype=weight_dtype)

            with network:
                # ちょっとデノイズされれたものが返る
                denoised_latents = train_util.diffusion(
                    unet,
                    noise_scheduler,
                    latents,  # 単純なノイズのlatentsを渡す
                    train_util.concat_embeddings(
                        cache[""],  # 最初は普通にやる
                        prompt_pair.target,
                        prompt_pair.batch_size,
                    ),
                    start_timesteps=0,
                    total_timesteps=timesteps_to,
                    guidance_scale=3,
                )

            noise_scheduler.set_timesteps(1000)

            current_timestep = noise_scheduler.timesteps[
                int(timesteps_to * 1000 / config.train.max_denoising_steps)
            ]

            # with network: の外では空のLoRAのみが有効になる
            positive_latents = train_util.predict_noise(
                unet,
                noise_scheduler,
                current_timestep,
                denoised_latents,
                train_util.concat_embeddings(
                    prompt_pair.unconditional,
                    prompt_pair.positive,
                    prompt_pair.batch_size,
                ),
                guidance_scale=1,
            ).to("cpu", dtype=torch.float32)
            neutral_latents = train_util.predict_noise(
                unet,
                noise_scheduler,
                current_timestep,
                denoised_latents,
                train_util.concat_embeddings(
                    prompt_pair.unconditional,
                    prompt_pair.neutral,
                    prompt_pair.batch_size,
                ),
                guidance_scale=1,
            ).to("cpu", dtype=torch.float32)
            unconditional_latents = train_util.predict_noise(
                unet,
                noise_scheduler,
                current_timestep,
                denoised_latents,
                train_util.concat_embeddings(
                    prompt_pair.unconditional,
                    prompt_pair.unconditional,
                    prompt_pair.batch_size,
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
                noise_scheduler,
                current_timestep,
                denoised_latents,
                train_util.concat_embeddings(
                    prompt_pair.unconditional,
                    prompt_pair.target,
                    prompt_pair.batch_size,
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

        # wandb に記録するログ
        log_dict = {
            "loss": loss,
            "iteration": i,
            "lr": lr_scheduler.get_last_lr()[0],
        }

        if (
            config.logging.sample_per_steps != 0
            and i != 0
            and i % config.logging.sample_per_steps == 0
        ):
            if config.logging.sample is not None:
                print("Generating sample images...")
                total_sample_images = 0
                with network:
                    for sample in config.logging.sample:
                        print("Generating sample image:", sample.positive)
                        sample_images = sample_util.sample_image(
                            unet,
                            noise_scheduler,
                            vae,
                            sample_cache[sample.positive],
                            sample_cache[sample.negative],
                            width=sample.width,
                            height=sample.height,
                            num_inference_steps=sample.num_inference_steps,
                            cfg_scale=sample.cfg_scale,
                            seed=sample.seed,
                            weight_dtype=weight_dtype,
                        )
                        for image in sample_images:
                            # 画像をログイに追加
                            log_dict[f"sample_{total_sample_images}"] = wandb.Image(
                                image, caption=sample.positive
                            )
                            total_sample_images += 1
                # save samples
                sample_util.save_images(
                    sample_images,
                    save_path / "samples",
                    filename_base=f"sample_{i}",
                )
                del sample_images

        if config.logging.use_wandb:
            wandb.log(
                log_dict,
            )

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
        noise_scheduler,
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
