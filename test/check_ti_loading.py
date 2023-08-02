import sys

sys.path.append(".")

import argparse

import torch

from model_util import load_textual_inversion, load_models
from train_util import encode_prompts
from sample_util import sample_image

DEVICE_CUDA = torch.device("cuda:0")

SD15_MODEL = "runwayml/stable-diffusion-v1-5"


@torch.no_grad()
def main(args):
    ti_path = args.ti_path
    prompt = args.prompt
    model = args.model
    v2 = args.v2
    v_pred = args.v_pred
    width = args.width
    height = args.height

    ti = load_textual_inversion(
        ti_path,
    )

    tokenizer, text_encoder, unet, vae, scheduler = load_models(
        model,
        "euler_a",
        v2=v2,
        v_pred=v_pred,
        weight_dtype=torch.float16,
    )
    text_encoder.to(DEVICE_CUDA, dtype=torch.float16)

    encoded_1 = encode_prompts(tokenizer, text_encoder, [prompt])

    if isinstance(ti, list):
        for model in ti:
            tokenizer, text_encoder = model.apply_textual_inversion(
                tokenizer, text_encoder
            )
            print(f"textual inversion '{model.token}' applied")
    else:
        tokenizer, text_encoder = ti.apply_textual_inversion(tokenizer, text_encoder)
        print(f"textual inversion '{ti.token}' applied")

    encoded_2 = encode_prompts(tokenizer, text_encoder, [prompt])

    print(encoded_1)
    print(encoded_2)

    assert not torch.allclose(encoded_1, encoded_2)

    unet.to(DEVICE_CUDA, dtype=torch.float16)
    unet.enable_xformers_memory_efficient_attention()
    vae.to(DEVICE_CUDA, dtype=torch.float16)

    # generate
    image = sample_image(
        unet,
        scheduler,
        vae,
        positive=encoded_2,
        negative=encode_prompts(tokenizer, text_encoder, [""]),
        width=width,
        height=height,
    )

    for i, img in enumerate(image):
        img.save(f"./output/test_{i}.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("ti_path", type=str)
    parser.add_argument("prompt", type=str)
    parser.add_argument("--model", type=str, default=SD15_MODEL)
    parser.add_argument("--v2", action="store_true")
    parser.add_argument("--v_pred", action="store_true")
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)

    args = parser.parse_args()

    main(args)
