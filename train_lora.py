# ref:
# - https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py#L566
# - https://huggingface.co/spaces/baulab/Erasing-Concepts-In-Diffusion/blob/main/train.py

from typing import List, Optional
import argparse
from pathlib import Path
import json
import gc

import torch
from tqdm import tqdm


from lora import LoRANetwork, DEFAULT_TARGET_REPLACE
import train_util
import model_util
import prompt_util
from prompt_util import PromptCache, PromptPair, PromptSettings
import debug_util

import wandb

DEVICE_CUDA = "cuda"
DDIM_STEPS = 50


def flush():
    torch.cuda.empty_cache()
    gc.collect()


def train(
    prompts: list[PromptSettings],
    pretrained_model: str,
    modules: List[str],
    iterations: int,
    rank: int = 4,
    alpha: float = 1.0,
    lr: float = 1e-4,
    resolution: int = 512,
    batch_size: int = 1,
    save_path: Path = Path("./output"),
    v2: bool = False,
    v_pred: bool = False,
    precision: str = "bfloat16",
    scheduler_name: str = "lms",
    enable_wandb: bool = False,
    save_steps: int = 200,
    save_precision: str = "float",
    save_name: Optional[str] = None,
):
    if save_name == "" or save_name is None:
        # 保存用の名前
        save_name = "Untitled"
    metadata = {
        "prompts": json.dumps(prompts),
        "pretrained_model": pretrained_model,
        "modules": ", ".join(modules),
        "iterations": iterations,
        "rank": rank,
        "alpha": alpha,
        "lr": lr,
        "resolution": resolution,
        "v2": v2,
        "v_pred": v_pred,
        "precision": precision,
        "scheduler_name": scheduler_name,
        "save_path": str(save_path),
        "save_steps": save_steps,
        "save_precision": save_precision,
        "save_name": save_name,
    }

    if enable_wandb:
        wandb.init(project=f"LECO_{save_name}")
        wandb.config = metadata

    weight_dtype = torch.float32
    if precision == "float16":
        weight_dtype = torch.float16
    elif precision == "bfloat16":
        weight_dtype = torch.bfloat16

    save_weight_dtype = torch.float32
    if save_precision == "float16":
        save_weight_dtype = torch.float16
    elif save_precision == "bfloat16":
        save_weight_dtype = torch.bfloat16

    tokenizer, text_encoder, unet, scheduler = model_util.load_models(
        pretrained_model,
        scheduler_name=scheduler_name,
        v2=v2,
        v_pred=v_pred,
    )

    text_encoder.to(DEVICE_CUDA, dtype=weight_dtype)
    text_encoder.eval()

    unet.to(DEVICE_CUDA, dtype=weight_dtype)
    unet.enable_xformers_memory_efficient_attention()
    unet.requires_grad_(False)
    unet.eval()

    network = LoRANetwork(unet, rank=rank, multiplier=1.0, alpha=1).to(
        DEVICE_CUDA, dtype=weight_dtype
    )

    optimizer = torch.optim.AdamW(network.prepare_optimizer_params(), lr=lr)
    criteria = torch.nn.MSELoss()

    print("Prompts")
    for settings in prompts:
        print(settings)

    # debug
    debug_util.check_requires_grad(network)
    debug_util.check_training_mode(network)

    pbar = tqdm(range(iterations))

    cache = PromptCache()
    prompt_pairs: list[PromptPair] = []

    with torch.no_grad():
        for settings in prompts:
            for key in ["target", "positive", "neutral", "unconditional"]:
                if cache[settings[key]] == None:
                    cache[settings[key]] = train_util.encode_prompts(
                        tokenizer, text_encoder, [settings[key]]
                    )
            prompt_pairs.append(
                PromptPair(
                    criteria,
                    cache[settings["target"]],
                    cache[settings["positive"]],
                    cache[settings["unconditional"]],
                    cache[settings["neutral"]],
                    settings["guidance_scale"],
                    settings["action"],
                )
            )

    del tokenizer
    del text_encoder

    flush()

    n_imgs = batch_size

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
                scheduler, n_imgs, resolution, 1
            ).to(DEVICE_CUDA, dtype=weight_dtype)

            with network:
                # ちょっとデノイズされれたものが返る
                denoised_latents = train_util.diffusion(
                    unet,
                    scheduler,
                    latents,  # 単純なノイズのlatentsを渡す
                    train_util.concat_embeddings(
                        prompt_pair.unconditional, prompt_pair.positive, n_imgs
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
        if enable_wandb:
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

        if i % save_steps == 0 and i != 0 and i != iterations - 1:
            print("Saving...")
            save_path.mkdir(parents=True, exist_ok=True)
            network.save_weights(
                save_path / f"{save_name}_{i}steps.safetensors",
                dtype=save_weight_dtype,
            )

    print("Saving...")
    save_path.mkdir(parents=True, exist_ok=True)
    network.save_weights(
        save_path / f"{save_name}_last.safetensors",
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
    prompts_file = args.prompts_file
    pretrained_model = args.pretrained_model
    rank = args.rank
    alpha = args.alpha
    iterations = args.iterations
    lr = args.lr
    resolution = args.resolution
    batch_size = args.batch_size
    save_path = Path(args.save_path).resolve()
    v2 = args.v2
    v_pred = args.v_pred
    precision = args.precision
    scheduler_name = args.scheduler_name
    enable_wandb = args.use_wandb
    save_steps = args.save_steps
    save_precision = args.save_precision
    save_name = args.save_name

    prompts = prompt_util.load_prompts_from_yaml(prompts_file)

    train(
        prompts,
        pretrained_model,
        modules=DEFAULT_TARGET_REPLACE,
        iterations=iterations,
        rank=rank,
        alpha=alpha,
        lr=lr,
        resolution=resolution,
        batch_size=batch_size,
        save_path=save_path,
        v2=v2,
        v_pred=v_pred,
        precision=precision,
        scheduler_name=scheduler_name,
        enable_wandb=enable_wandb,
        save_steps=save_steps,
        save_precision=save_precision,
        save_name=save_name,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prompts_file",
        required=True,
        help="YAML file of settings of prompts.",
    )
    parser.add_argument(
        "--pretrained_model",
        type=str,
        required=True,
        help="Path to diffusers model or repo name",
    )
    parser.add_argument("--rank", type=int, default=4, help="rank of LoRA")
    parser.add_argument("--alpha", type=float, default=1, help="alpha of LoRA")
    parser.add_argument(
        "--iterations", type=int, default=1000, help="Number of iterations"
    )
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--resolution", type=int, default=512, help="Resolution")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--save_path", default="./output", help="Path to save weights")
    parser.add_argument("--v2", action="store_true", default=False, help="Use v2 model")
    parser.add_argument(
        "--v_pred", action="store_true", default=False, help="Use v_prediction model"
    )
    parser.add_argument(
        "--precision",
        type=str,
        choices=["float32", "float16", "bfloat16"],
        default="float16",
    )
    parser.add_argument(
        "--scheduler_name",
        type=str,
        choices=["lms", "ddim", "ddpm", "euler_a"],
        default="lms",
    )
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        default=False,
        help="Use wandb to logging.",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=200,
        help="Save weights every n steps.",
    )
    parser.add_argument(
        "--save_precision",
        type=str,
        choices=["float", "float32", "float16", "bfloat16"],
        default="float",
        help="Save weights with specified precision.",
    )
    parser.add_argument(
        "--save_name",
        type=str,
        default=None,
        help="Save weights with specified name.",
    )

    args = parser.parse_args()

    main(args)
