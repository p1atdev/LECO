# ref:
# - https://github.com/cloneofsimo/lora/blob/master/lora_diffusion/lora.py
# - https://github.com/kohya-ss/sd-scripts/blob/main/networks/lora.py

import os
import math
from typing import Optional, List, Type, Set
from pathlib import Path

import torch
import torch.nn as nn
from safetensors.torch import save_file
from tqdm import tqdm


# UNET_DEFAULT_TARGET_REPLACE = {"CrossAttention", "Attention", "GEGLU"}
# UNET_EXTENDED_TARGET_REPLACE = {"ResnetBlock2D", "CrossAttention", "Attention", "GEGLU"}


UNET_TARGET_REPLACE_MODULE_TRANSFORMER = [
    "CrossAttention",  # <- ESD-x (多分)
    "Attention",
    # "Transformer2DModel",
    "GEGLU",
]
# UNET_TARGET_REPLACE_MODULE_CONV = ["ResnetBlock2D", "Downsample2D", "Upsample2D"]
TEXT_ENCODER_TARGET_REPLACE_MODULE = ["CLIPAttention", "CLIPMLP"]
LORA_PREFIX_UNET = "lora_unet"
LORA_PREFIX_TEXT_ENCODER = "lora_te"

EMPTY_PREFIX_UNET = "empty_unet"

DEFAULT_TARGET_REPLACE = UNET_TARGET_REPLACE_MODULE_TRANSFORMER


class LoRAModule(nn.Module):
    """
    replaces forward method of the original Linear, instead of replacing the original Linear module.
    """

    def __init__(
        self,
        lora_name,
        org_module: nn.Module,
        multiplier=1.0,
        lora_dim=4,
        alpha=1,
    ):
        """if alpha == 0 or None, alpha is rank (no scaling)."""
        super().__init__()
        self.lora_name = lora_name
        self.lora_dim = lora_dim

        if org_module.__class__.__name__ == "Linear":
            in_dim = org_module.in_features
            out_dim = org_module.out_features
            self.lora_down = nn.Linear(in_dim, lora_dim, bias=False)
            self.lora_up = nn.Linear(lora_dim, out_dim, bias=False)

        elif org_module.__class__.__name__ == "Conv2d":
            in_dim = org_module.in_channels
            out_dim = org_module.out_channels

            self.lora_dim = min(self.lora_dim, in_dim, out_dim)
            if self.lora_dim != lora_dim:
                print(f"{lora_name} dim (rank) is changed to: {self.lora_dim}")

            kernel_size = org_module.kernel_size
            stride = org_module.stride
            padding = org_module.padding
            self.lora_down = nn.Conv2d(
                in_dim, self.lora_dim, kernel_size, stride, padding, bias=False
            )
            self.lora_up = nn.Conv2d(self.lora_dim, out_dim, (1, 1), (1, 1), bias=False)

        if type(alpha) == torch.Tensor:
            alpha = alpha.detach().numpy()
        alpha = lora_dim if alpha is None or alpha == 0 else alpha
        self.scale = alpha / self.lora_dim
        self.register_buffer("alpha", torch.tensor(alpha))  # 定数として扱える

        # same as microsoft's
        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)

        self.multiplier = multiplier
        self.org_module = org_module  # remove in applying

    def apply_to(self):
        self.org_forward = self.org_module.forward
        self.org_module.forward = self.forward
        del self.org_module

    def forward(self, x):
        return (
            self.org_forward(x)
            + self.lora_up(self.lora_down(x)) * self.multiplier * self.scale
        )


class LoRANetwork(nn.Module):
    def __init__(
        self,
        diffuser,
        rank=4,
        multiplier=1.0,
        alpha=1,
    ) -> None:
        super().__init__()

        model = diffuser
        self.multiplier = multiplier
        self.lora_dim = rank
        self.alpha = alpha
        # self.target_block = "up_blocks" if up_only else ""

        # LoRAのみ
        self.module = LoRAModule

        # util.freeze(self.model)

        # unetのloraを作る
        self.unet_loras = self.create_modules(
            LORA_PREFIX_UNET,
            model.unet,
            UNET_TARGET_REPLACE_MODULE_TRANSFORMER,
            self.lora_dim,
            self.multiplier,
            train=True,
        )
        print(f"create LoRA for U-Net: {len(self.unet_loras)} modules.")

        # 空のloraを作る(学習しない)
        self.empty_loras = self.create_modules(
            EMPTY_PREFIX_UNET,  # 名前をわけて、場所によって multiplier を変更して適用を切り替える
            model.unet,
            UNET_TARGET_REPLACE_MODULE_TRANSFORMER,
            self.lora_dim,
            multiplier=0,
            train=False,
        )

        # assertion 名前の被りがないか確認しているようだ
        lora_names = set()
        for lora in self.unet_loras:
            assert (
                lora.lora_name not in lora_names
            ), f"duplicated lora name: {lora.lora_name}"
            lora_names.add(lora.lora_name)
        empty_names = set()
        for lora in self.unet_loras:
            assert (
                lora.lora_name not in empty_names
            ), f"duplicated lora name: {lora.lora_name}"
            empty_names.add(lora.lora_name)

        # 適用する？
        for lora in self.unet_loras:
            lora.apply_to()
            self.add_module(
                lora.lora_name,
                lora,
            )

        for empty in self.empty_loras:
            empty.apply_to()
            self.add_module(
                empty.lora_name,
                empty,
            )

        self.requires_grad_(True)

        # del unet

    # 見づらいのでメソッドにしちゃう
    def create_modules(
        self,
        prefix: str,
        root_module: nn.Module,
        target_replace_modules: List[str],
        rank: int,
        multiplier: float,
        train: bool = True,
    ) -> list:
        loras = []
        for name, module in root_module.named_modules():
            if (
                module.__class__.__name__
                in target_replace_modules
                # and self.target_block in name
            ):
                for child_name, child_module in module.named_modules():
                    if child_module.__class__.__name__ in ["Linear", "Conv2d"]:
                        lora_name = prefix + "." + name + "." + child_name
                        lora_name = lora_name.replace(".", "_")
                        lora = self.module(
                            lora_name, child_module, multiplier, rank, self.alpha
                        )
                        if not train:
                            lora.requires_grad_(False)
                            lora.eval()
                        loras.append(lora)
        return loras

    # def backward(self, loss):
    #     super().backward(loss)
    #     print(self.mode)

    def prepare_optimizer_params(self):
        self.requires_grad_(True)
        all_params = []

        if self.unet_loras:  # 実質これしかない
            params = []
            [params.extend(lora.parameters()) for lora in self.unet_loras]
            param_data = {"params": params}
            all_params.append(param_data)

        return all_params

    def save_weights(self, file, dtype=None):
        state_dict = self.state_dict()

        if dtype is not None:
            for key in list(state_dict.keys()):
                v = state_dict[key]
                v = v.detach().clone().to("cpu").to(dtype)
                state_dict[key] = v

        for key in list(state_dict.keys()):
            if not key.startswith("lora"):
                # lora以外除外
                del state_dict[key]

        if os.path.splitext(file)[1] == ".safetensors":
            save_file(state_dict, file)
        else:
            torch.save(state_dict, file)

    def __enter__(self):
        for lora in self.unet_loras:
            lora.multiplier = 1.0
        for empty in self.empty_loras:
            empty.multiplier = 0

    def __exit__(self, exc_type, exc_value, tb):
        for lora in self.unet_loras:
            lora.multiplier = 0
        for empty in self.empty_loras:
            empty.multiplier = 1.0

    # def predict_noise(
    #     self, iteration, latents, text_embeddings, guidance_scale=7.5, v_pred=False
    # ):
    #     # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
    #     latents = torch.cat([latents] * 2)

    #     latents = self.model.scheduler.scale_model_input(
    #         latents, self.model.scheduler.timesteps[iteration]
    #     )

    #     # predict the noise residual
    #     noise_prediction = self.model.unet(
    #         latents,
    #         self.model.scheduler.timesteps[iteration],
    #         encoder_hidden_states=text_embeddings,
    #     ).sample

    #     # TODO: v_parameterization?
    #     if v_pred:
    #         target = self.model.scheduler.get_velocity(
    #             latents, noise_prediction, iteration
    #         )
    #     else:
    #         target = noise_prediction

    #     # perform guidance
    #     noise_prediction_uncond, noise_prediction_text = target.chunk(2)
    #     guided_target = noise_prediction_uncond + guidance_scale * (
    #         noise_prediction_text - noise_prediction_uncond
    #     )

    #     return guided_target

    # @torch.no_grad()
    # def diffusion(
    #     self,
    #     latents,
    #     text_embeddings,
    #     end_iteration=1000,
    #     start_iteration=0,
    #     return_steps=False,
    #     pred_x0=False,
    #     **kwargs,
    # ):
    #     latents_steps = []

    #     for iteration in tqdm(range(start_iteration, end_iteration)):
    #         noise_pred = self.predict_noise(
    #             iteration, latents, text_embeddings, **kwargs
    #         )

    #         # compute the previous noisy sample x_t -> x_t-1
    #         output = self.model.scheduler.step(
    #             noise_pred, self.model.scheduler.timesteps[iteration], latents
    #         )

    #         latents = output.prev_sample

    #         if return_steps or iteration == end_iteration - 1:
    #             output = output.pred_original_sample if pred_x0 else latents

    #             if return_steps:
    #                 latents_steps.append(output.cpu())
    #             else:
    #                 latents_steps.append(output)

    #     return latents_steps
