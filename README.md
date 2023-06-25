# LECO ✏️ (beta)

Low-rank adaptation for Erasing COncepts from diffusion models.  

The original repository: [Erasing Concepts from Diffusion Models](https://github.com/rohitgandikota/erasing/tree/main)

and the project page: https://erasing.baulab.info/

(Not only for erasing concepts, but also emphasizing or swapping them by devising prompts and LoRA weight. See [ConceptMod](https://github.com/ntc-ai/conceptmod) for more details)

## Setup

```bash
conda create -n leco python=3.10
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install xformers
pip install -r requirements.txt
```

## Train

You need 8GB VRAM at least.

```bash
python ./train_lora.py --config_file "./examples/config.yaml"
```

`config.yaml`:

```yaml
prompts_file: "./prompts.yaml"

pretrained_model:
  name_or_path: "stabilityai/stable-diffusion-2-1" # currently diffusers models only
  v2: true # true if model is v2.x
  v_pred: true # true if model uses v-prediction

network:
  type: "lierla" # or "c3lier"
  rank: 4
  alpha: 1.0

train:
  precision: "bfloat16"
  noise_scheduler: "ddim" # "ddpm", "lms" or "euler_a" are currently avaiable
  resolution: 512
  iterations: 500
  batch_size: 1
  lr: 1e-4

save:
  name: "van_gogh"
  path: "./output"
  per_steps: 200
  precision: "bfloat16"

logging:
  use_wandb: true
  verbose: false

other:
  use_xformers: true
```

`prompts.yaml`:

```yaml
- target: "van gogh" # what word for erasing the positive concept from
  positive: "van gogh" # concept to erase
  unconditional: "" # word to take the difference from the positive concept
  neutral: "" # starting point for conditioning the target
  action: "erase" # erase or enhance
  guidance_scale: 1.0
```

See the [example config](/examples/config.py) for more details.

Note: You can use float16 but it is unstable and not recommended. Please use bfloat16 or float32. 

## Pretrained weights

You can use the pretrained weights on AUTOMATIC1111's webui. 

🤗 HuggingFace: https://huggingface.co/p1atdev/leco

- [Van Gogh style](https://huggingface.co/p1atdev/leco/blob/main/van_gogh_sdv15.safetensors) (trained with "van gogh style" on SDv1.5)
- [Mona Lisa](https://huggingface.co/p1atdev/leco/blob/main/mona_lisa_sdv21v.safetensors) (trained with "mona lisa" on SDv2.1-768)



## References

I am deeply inspired by and my work relies on the outstanding efforts of the following projects. I want to express my profound gratitude to these projects and their developers:

- https://github.com/rohitgandikota/erasing: Erasing Concepts from Diffusion Models 
  - https://huggingface.co/spaces/baulab/Erasing-Concepts-In-Diffusion: Demo of ESD 

- https://github.com/cloneofsimo/lora: Low-rank Adaptation for Fast Text-to-Image Diffusion Fine-tuning

- https://github.com/kohya-ss/sd-scripts: Training, generation and utility scripts for Stable Diffusion

- https://github.com/ntc-ai/conceptmod:  Modify Concepts from Diffusion Models using a dsl 
