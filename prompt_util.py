from typing import Literal, TypedDict, Optional, Callable

import yaml
from pathlib import Path

import torch


class PromptCache:
    prompts: dict[str, torch.FloatTensor] = {}

    def __setitem__(self, __name: str, __value: torch.FloatTensor) -> None:
        self.prompts[__name] = __value

    def __getitem__(self, __name: str) -> torch.FloatTensor:
        if __name in self.prompts:
            return self.prompts[__name]
        else:
            return None


class PromptSettings(TypedDict):
    target: str
    positive: Optional[str]  # if None, target will be used
    unconditional: Optional[str]  # default is ""
    neutral: Optional[str]  # if None, unconditional will be used
    action: Optional[str]  # default is "erase"
    guidance_scale: Optional[float]  # default is 1.0


class PromptPair:
    ACTION_TYPES = Literal["erase", "enhance"]

    target: torch.FloatTensor  # not want to generate the concept
    positive: torch.FloatTensor  # generate the concept
    unconditional: torch.FloatTensor  # uncondition (default should be empty)
    neutral: torch.FloatTensor  # base condition (default should be empty)

    guidance_scale: float

    loss_fn: torch.nn.Module
    action: ACTION_TYPES

    def __init__(
        self,
        loss_fn: torch.nn.Module,
        target: torch.FloatTensor,
        positive: torch.FloatTensor,
        unconditional: torch.FloatTensor,
        neutral: torch.FloatTensor,
        guidance_scale: float,
        action: ACTION_TYPES,
    ) -> None:
        self.loss_fn = loss_fn
        self.target = target
        self.positive = positive
        self.unconditional = unconditional
        self.neutral = neutral
        self.guidance_scale = guidance_scale
        self.action = action

    def _erase(
        self,
        target_latents: torch.FloatTensor,  # "van gogh"
        positive_latents: torch.FloatTensor,  # "van gogh"
        unconditional_latents: torch.FloatTensor,  # ""
        neutral_latents: torch.FloatTensor,  # ""
    ) -> torch.FloatTensor:
        """Target latents are going not to have the positive concept."""

        return self.loss_fn(
            target_latents,
            neutral_latents
            - self.guidance_scale * (positive_latents - unconditional_latents),
        )

    def _enhance(
        self,
        target_latents: torch.FloatTensor,  # "van gogh"
        positive_latents: torch.FloatTensor,  # "van gogh"
        unconditional_latents: torch.FloatTensor,  # ""
        neutral_latents: torch.FloatTensor,  # ""
    ):
        """Target latents are going to have the positive concept."""

        return self.loss_fn(
            target_latents,
            neutral_latents
            + self.guidance_scale * (positive_latents - unconditional_latents),
        )

    def loss(
        self,
        **kwargs,
    ):
        if self.action == "erase":
            return self._erase(**kwargs)

        elif self.action == "enhance":
            return self._enhance(**kwargs)


def load_prompts_from_yaml(path: str | Path) -> list[PromptSettings]:
    with open(path, "r") as f:
        prompts = yaml.safe_load(f)

    if len(prompts) == 0:
        raise ValueError("prompts file is empty")

    for prompt in prompts:
        keys = prompt.keys()
        if "target" not in keys:
            raise KeyError("target is required")
        if "positive" not in keys:
            prompt["positive"] = prompt["target"]
        if "unconditional" not in keys:
            prompt["unconditional"] = ""
        if "neutral" not in keys:
            prompt["neutral"] = prompt["unconditional"]
        if "action" not in keys:
            prompt["action"] = "erase"
        if "guidance_scale" not in keys:
            prompt["guidance_scale"] = 1.0

    return prompts
