# ref: https://huggingface.co/spaces/baulab/Erasing-Concepts-In-Diffusion/blob/main/train.py

from typing import List

import torch
from tqdm import tqdm
import wandb
import argparse
from pathlib import Path

from PIL import Image
from diffusers import AutoencoderKL
from diffusers.image_processor import VaeImageProcessor

from lora import DEFAULT_TARGET_REPLACE, LoRANetwork
import train_util
import model_util

DEVICE_CUDA = "cuda"
DDIM_STEPS = 50


# デバッグ用...
def check_requires_grad(model: torch.nn.Module):
    for name, module in list(model.named_modules())[:5]:
        if len(list(module.parameters())) > 0:
            print(f"Module: {name}")
            for name, param in list(module.named_parameters())[:2]:
                print(f"    Parameter: {name}, Requires Grad: {param.requires_grad}")


def check_training_mode(model):
    for name, module in list(model.named_modules())[:5]:
        print(f"Module: {name}, Training Mode: {module.training}")


def train(
    prompt: str,
    pretrained_model: str,
    modules: List[str],
    iterations: int,
    neutral_prompt: str = "",
    rank: int = 4,
    alpha: float = 1.0,
    negative_guidance: float = 1.0,
    lr: float = 1e-5,
    save_path: Path = Path("./output"),
    v2: bool = False,
    v_pred: bool = False,
    precision: str = "float16",
    enable_wandb: bool = False,
):
    weight_dtype = torch.float32
    if precision == "float16":
        weight_dtype = torch.float16
    elif precision == "bfloat16":
        weight_dtype = torch.bfloat16

    tokenizer, text_encoder, unet, scheduler = model_util.load_models(
        pretrained_model, scheduler_name="lms", v2=False, v_pred=v_pred  # とりあえずv1
    )

    vae = AutoencoderKL.from_pretrained(pretrained_model, subfolder="vae").to(
        DEVICE_CUDA, dtype=weight_dtype
    )
    vae.eval()

    text_encoder.to(DEVICE_CUDA, dtype=weight_dtype)
    text_encoder.eval()

    unet.to(DEVICE_CUDA, dtype=weight_dtype)
    unet.eval()

    # util.freeze(diffuser)

    network = LoRANetwork(unet, rank=rank, multiplier=1.0, alpha=1).to(
        DEVICE_CUDA, dtype=weight_dtype
    )
    # network.train()

    # print("params", network.prepare_optimizer_params())

    optimizer = torch.optim.AdamW(network.prepare_optimizer_params(), lr=lr)
    criteria = torch.nn.MSELoss()

    pbar = tqdm(range(iterations))

    with torch.no_grad():
        neutral_text_embeddings = train_util.get_text_embeddings(
            tokenizer, text_encoder, [""], n_imgs=1
        )
        positive_text_embeddings = train_util.get_text_embeddings(
            tokenizer, text_encoder, [prompt], n_imgs=1
        )

    # print("neutral", neutral_text_embeddings.shape)
    # print("positive", positive_text_embeddings.shape)

    del tokenizer
    del text_encoder

    torch.cuda.empty_cache()

    with torch.no_grad():
        scheduler.set_timesteps(DDIM_STEPS, device=DEVICE_CUDA)

        optimizer.zero_grad()

        # 1 ~ 48 からランダム
        timesteps_to = torch.randint(1, DDIM_STEPS - 1, (1,)).item()

        latents = train_util.get_initial_latents(scheduler, 1, 512, 1).to(
            DEVICE_CUDA, dtype=weight_dtype
        )
        with network:
            # ちょっとデノイズされれたものが入る
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

        current_timestep = scheduler.timesteps[int(timesteps_to * 1000 / DDIM_STEPS)]

        # with network の外では空の学習しないLoRAのみを有効にする(はず...)
        positive_latents = train_util.predict_noise(
            unet,
            scheduler,
            current_timestep,
            denoised_latents,
            positive_text_embeddings,
            guidance_scale=1,
        )
        neutral_latents = train_util.predict_noise(
            unet,
            scheduler,
            current_timestep,
            denoised_latents,
            neutral_text_embeddings,
            guidance_scale=1,
        )

        with network:
            negative_latents = train_util.predict_noise(
                unet,
                scheduler,
                current_timestep,
                denoised_latents,
                positive_text_embeddings,
                guidance_scale=1,
            )

        processor = VaeImageProcessor()

        for i, lat in enumerate(
            [denoised_latents, positive_latents, neutral_latents, negative_latents]
        ):
            img = processor.numpy_to_pil(
                processor.pt_to_numpy(
                    vae.decode(lat / vae.config.scaling_factor).sample
                )
            )[0]
            img.save(f"test{i}.png")

        print("Saving...")

    # network.save_weights(save_path / "last.safetensors", dtype=weight_dtype)

    del (
        unet,
        scheduler,
        # loss,
        optimizer,
        network,
        negative_latents,
        neutral_latents,
        positive_latents,
        # latents_steps,
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

    args = parser.parse_args()

    main(args)
