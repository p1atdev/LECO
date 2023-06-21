# ref: https://huggingface.co/spaces/baulab/Erasing-Concepts-In-Diffusion/blob/main/train.py

from typing import List

import torch
from tqdm import tqdm
import wandb
import argparse
from pathlib import Path

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
    rank: int = 4,
    scale: float = 1.0,
    dropout_p: float = 0.0,
    negative_guidance: float = 1.0,
    lr: float = 1e-5,
    save_path: Path = Path("./output"),
    verbose: bool = False,
    v_pred: bool = False,
    precision: str = "float16",
    enable_wandb: bool = False,
):
    if enable_wandb:
        wandb.init(project="LESD")
        wandb.config = {
            "prompt": prompt,
            "pretrained_model": pretrained_model,
            "modules": modules,
            "iterations": iterations,
            "rank": rank,
            "scale": scale,
            "dropout_p": dropout_p,
            "negative_guidance": negative_guidance,
            "lr": lr,
            "save_path": str(save_path),
            "verbose": verbose,
            "v_pred": v_pred,
            "precision": precision,
        }

    weight_dtype = torch.float32
    if precision == "float16":
        weight_dtype = torch.float16
    elif precision == "bfloat16":
        weight_dtype = torch.bfloat16

    tokenizer, text_encoder, unet, scheduler = model_util.load_models(
        pretrained_model, scheduler_name="ddpm", v2=False, v_pred=v_pred  # とりあえずv1
    )

    text_encoder.to(DEVICE_CUDA, dtype=weight_dtype)
    text_encoder.eval()

    unet.to(DEVICE_CUDA, dtype=weight_dtype)
    unet.eval()

    # util.freeze(diffuser)

    network = LoRANetwork(unet, rank=rank, multiplier=1.0, alpha=1).to(
        DEVICE_CUDA, dtype=weight_dtype
    )
    # network.train()

    optimizer = torch.optim.Adam(network.prepare_optimizer_params(), lr=lr)
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

    # debug
    print("grads: network")
    check_requires_grad(network)
    # print("grads: diffuser")
    # check_requires_grad(diffuser)

    print("training mode: network")
    check_training_mode(network)
    # print("training mode: diffuser")
    # check_training_mode(diffuser)

    # print("diffusion structure")
    # print(diffuser)

    for i in pbar:
        if enable_wandb:
            wandb.log({"iteration": i})

        with torch.no_grad():
            scheduler.set_timesteps(DDIM_STEPS, device=DEVICE_CUDA)

            optimizer.zero_grad()

            # 1 ~ 48 からランダム
            timesteps_from = torch.randint(1, DDIM_STEPS - 1, (1,)).item()
            timesteps_end = round((timesteps_from / DDIM_STEPS) * 1000)

            print("from", timesteps_from, "end", timesteps_end)

            latents = train_util.get_initial_latents(scheduler, 1, 512, 1).to(
                DEVICE_CUDA, dtype=weight_dtype
            )

            # デバッグ用: multiplierを確認する
            # print("before_enter_multiplier unet:", network.unet_loras[0].multiplier)
            # print("before_enter_multiplier empty:", network.empty_loras[0].multiplier)

            with network:
                # with netowrk 内では学習するほうのLoRAのみを有効にする
                # デバッグ用
                # print("in_enter_multiplier unet:", network.unet_loras[0].multiplier)
                # print("in_enter_multiplier empty:", network.empty_loras[0].multiplier)
                latents_steps = train_util.diffusion(
                    unet,
                    scheduler,
                    latents,
                    positive_text_embeddings,
                    start_timesteps=0,
                    total_timesteps=timesteps_from,
                    guidance_scale=3,
                    return_steps=False,
                    # v_pred=v_pred,
                )

            # print("latents_steps", len(latents_steps))
            # for latent in latents_steps:
            #     print(latent[0, 0, :5, :5]) # 一応ちゃんとデノイズはしているらしい

            # デバッグ用
            # print("after_exit_multiplier unet:", network.unet_loras[0].multiplier)
            # print("after_exit_multiplier empty:", network.empty_loras[0].multiplier)

            scheduler.set_timesteps(1000)

            # with network の外では空の学習しないLoRAのみを有効にする(はず...)
            positive_latents = train_util.predict_noise(
                unet,
                scheduler,
                timesteps_end,
                latents_steps[0],
                positive_text_embeddings,
                guidance_scale=1,
            ).float()
            print("positive_latents", positive_latents[0, 0, :5, :5])
            neutral_latents = train_util.predict_noise(
                unet,
                scheduler,
                timesteps_end,
                latents_steps[0],
                neutral_text_embeddings,
                guidance_scale=1,
            ).float()
            print("neutral_latents", neutral_latents[0, 0, :5, :5])

        with network:
            negative_latents = train_util.predict_noise(
                unet,
                scheduler,
                timesteps_end,
                latents_steps[0],
                positive_text_embeddings,
                guidance_scale=1,
            ).float()
            print("negative_latents", negative_latents[0, 0, :5, :5])

        positive_latents.requires_grad = False
        neutral_latents.requires_grad = False

        # FIXME: ここのロスが二回目以降nanになる (1回目も小さすぎる)
        loss = criteria(
            negative_latents,
            neutral_latents
            - (negative_guidance * (positive_latents - neutral_latents)),
        )  # loss = criteria(e_n, e_0) works the best try 5000 epochs

        pbar.set_description(f"Loss: {loss.item():.4f}")
        if enable_wandb:
            wandb.log({"loss": loss})

        loss.backward()
        optimizer.step()

    print("Saving...")

    network.save_weights(save_path / "last.safetensors", dtype=weight_dtype)

    del (
        # diffuser,
        unet,
        scheduler,
        loss,
        optimizer,
        network,
        negative_latents,
        neutral_latents,
        positive_latents,
        latents_steps,
        latents,
    )

    torch.cuda.empty_cache()

    print("Done.")


def main(args):
    prompt = args.prompt
    pretrained_model = args.pretrained_model
    rank = args.rank
    dropout_p = args.dropout_p
    scale = args.scale
    iterations = args.iterations
    negative_guidance = args.negative_guidance
    lr = args.lr
    save_path = Path(args.save_path).resolve()
    v_pred = args.v_pred
    precision = args.precision

    train(
        prompt,
        pretrained_model,
        modules=DEFAULT_TARGET_REPLACE,
        iterations=iterations,
        rank=rank,
        scale=scale,
        dropout_p=dropout_p,
        negative_guidance=negative_guidance,
        lr=lr,
        save_path=save_path,
        verbose=args.verbose,
        v_pred=v_pred,
        precision=precision,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--pretrained_model", type=str, required=True)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--dropout_p", type=float, default=0.0)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--negative_guidance", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--save_path", default="./output")
    parser.add_argument("--verbose", action="store_true", default=False)
    parser.add_argument("--v_pred", action="store_true", default=False)
    parser.add_argument(
        "--precision",
        type=str,
        choices=["float32", "float16", "bfloat16"],
        default="float16",
    )

    args = parser.parse_args()

    main(args)
