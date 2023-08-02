from typing import Literal, Optional

import yaml

from pydantic import BaseModel
import torch

from lora import TRAINING_METHODS

PRECISION_TYPES = Literal["fp32", "fp16", "bf16", "float32", "float16", "bfloat16"]
NETWORK_TYPES = Literal["lierla", "c3lier"]


class PretrainedModelConfig(BaseModel):
    name_or_path: str
    v2: bool = False
    v_pred: bool = False

    clip_skip: Optional[int] = None

    embeddings_dir: Optional[str] = None


class NetworkConfig(BaseModel):
    type: NETWORK_TYPES = "lierla"
    rank: int = 4
    alpha: float = 1.0

    training_method: TRAINING_METHODS = "full"


class TrainConfig(BaseModel):
    precision: PRECISION_TYPES = "bfloat16"
    noise_scheduler: Literal["ddim", "ddpm", "lms", "euler_a"] = "ddim"

    iterations: int = 500
    lr: float = 1e-4
    optimizer: str = "adamw"
    lr_scheduler: str = "constant"

    max_denoising_steps: int = 50

    noise_offset: float = 0.0
    pyramid_noise_discount: float = 0.0


class SaveConfig(BaseModel):
    name: str = "untitled"
    path: str = "./output"
    per_steps: int = 200
    precision: PRECISION_TYPES = "float32"


class SampleConfig(BaseModel):
    positive: str
    negative: str = ""

    width: int = 512
    height: int = 512
    num_inference_steps: int = 20
    cfg_scale: float = 7.5
    seed: int = -1


class LoggingConfig(BaseModel):
    use_wandb: bool = False

    verbose: bool = False

    sample: Optional[list[SampleConfig]] = None
    sample_per_steps: int = 100


class OtherConfig(BaseModel):
    use_xformers: bool = False


class RootConfig(BaseModel):
    prompts_file: str
    pretrained_model: PretrainedModelConfig

    network: NetworkConfig

    train: Optional[TrainConfig]

    save: Optional[SaveConfig]

    logging: Optional[LoggingConfig]

    other: Optional[OtherConfig]


def parse_precision(precision: str) -> torch.dtype:
    if precision == "fp32" or precision == "float32":
        return torch.float32
    elif precision == "fp16" or precision == "float16":
        return torch.float16
    elif precision == "bf16" or precision == "bfloat16":
        return torch.bfloat16

    raise ValueError(f"Invalid precision type: {precision}")


def load_config_from_yaml(config_path: str) -> RootConfig:
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    root = RootConfig(**config)

    if root.train is None:
        root.train = TrainConfig()

    if root.save is None:
        root.save = SaveConfig()

    if root.logging is None:
        root.logging = LoggingConfig()

    if root.other is None:
        root.other = OtherConfig()

    return root
