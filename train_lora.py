# ref:
# - https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py#L566
# - https://huggingface.co/spaces/baulab/Erasing-Concepts-In-Diffusion/blob/main/train.py

from typing import List, Optional
import argparse
from pathlib import Path

import torch
from tqdm import tqdm

from lora import DEFAULT_TARGET_REPLACE, LoRANetwork
import train_util
import model_util

import wandb

DEVICE_CUDA = "cuda"
DDIM_STEPS = 50


def train(
    prompt: str,
    pretrained_model: str,
    modules: List[str],
    iterations: int,
    neutral_prompt: str = "",
    rank: int = 4,
    alpha: float = 1.0,
    negative_guidance: float = 1.0,
    lr: float = 1e-4,
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
        save_name = prompt.replace(" ", "_")
    metadata = {
        "prompt": prompt,
        "neutral_prompt": neutral_prompt,
        "pretrained_model": pretrained_model,
        "modules": ", ".join(modules),
        "iterations": iterations,
        "rank": rank,
        "alpha": alpha,
        "negative_guidance": negative_guidance,
        "lr": lr,
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
    unet.eval()

    network = LoRANetwork(unet, rank=rank, multiplier=1.0, alpha=1).to(
        DEVICE_CUDA, dtype=weight_dtype
    )

    optimizer = torch.optim.AdamW(network.prepare_optimizer_params(), lr=lr)
    criteria = torch.nn.MSELoss()

    pbar = tqdm(range(iterations))

    with torch.no_grad():
        neutral_text_embeddings = train_util.get_text_embeddings(
            tokenizer, text_encoder, [neutral_prompt], n_imgs=1
        )
        positive_text_embeddings = train_util.get_text_embeddings(
            tokenizer, text_encoder, [prompt], n_imgs=1
        )

    del tokenizer
    del text_encoder

    torch.cuda.empty_cache()

    for i in pbar:
        with torch.no_grad():
            scheduler.set_timesteps(DDIM_STEPS, device=DEVICE_CUDA)

            optimizer.zero_grad()

            # 1 ~ 49 からランダム
            timesteps_to = torch.randint(1, DDIM_STEPS, (1,)).item()

            latents = train_util.get_initial_latents(scheduler, 1, 512, 1).to(
                DEVICE_CUDA, dtype=weight_dtype
            )
            with network:
                # ちょっとデノイズされれたものが返る
                denoised_latents = train_util.diffusion(
                    unet,
                    scheduler,
                    latents,  # 単純なノイズのlatentsを渡す
                    positive_text_embeddings,
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
                positive_text_embeddings,
                guidance_scale=1,
            ).to("cpu", dtype=torch.float32)
            neutral_latents = train_util.predict_noise(
                unet,
                scheduler,
                current_timestep,
                denoised_latents,
                neutral_text_embeddings,
                guidance_scale=1,
            ).to("cpu", dtype=torch.float32)

        with network:
            negative_latents = train_util.predict_noise(
                unet,
                scheduler,
                current_timestep,
                denoised_latents,
                positive_text_embeddings,
                guidance_scale=1,
            ).to("cpu", dtype=torch.float32)

        positive_latents.requires_grad = False
        neutral_latents.requires_grad = False

        loss = criteria(
            negative_latents,
            neutral_latents
            - (negative_guidance * (positive_latents - neutral_latents)),
        )  # loss = criteria(e_n, e_0) works the best try 5000 epochs

        pbar.set_description(f"Loss: {loss.item():.4f}")
        if enable_wandb:
            wandb.log({"loss": loss, "iteration": i})

        loss.backward()
        optimizer.step()

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
        negative_latents,
        neutral_latents,
        positive_latents,
        latents,
    )

    torch.cuda.empty_cache()

    print("Done.")


def main(args):
    prompt = args.prompt
    neutral_prompt = args.neutral_prompt
    pretrained_model = args.pretrained_model
    rank = args.rank
    alpha = args.alpha
    iterations = args.iterations
    negative_guidance = args.negative_guidance
    lr = args.lr
    save_path = Path(args.save_path).resolve()
    v2 = args.v2
    v_pred = args.v_pred
    precision = args.precision
    scheduler_name = args.scheduler_name
    enable_wandb = args.use_wandb
    save_steps = args.save_steps
    save_precision = args.save_precision
    save_name = args.save_name

    train(
        prompt,
        pretrained_model,
        modules=DEFAULT_TARGET_REPLACE,
        neutral_prompt=neutral_prompt,
        iterations=iterations,
        rank=rank,
        alpha=alpha,
        negative_guidance=negative_guidance,
        lr=lr,
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
        "--prompt",
        required=True,
        help="Prompt of a concept to delete, emphasis, or swap.",
    )
    parser.add_argument(
        "--neutral_prompt",
        default="",
        help="Prompt of neautral condition.",
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
    parser.add_argument(
        "--negative_guidance", type=float, default=1.0, help="Negative guidance"
    )
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
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
