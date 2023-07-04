# LECO ✏️ 

<a href="https://colab.research.google.com/github/p1atdev/LECO/blob/main/train.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

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
  name_or_path: "stabilityai/stable-diffusion-2-1" # you can also use .ckpt or .safetensors models
  v2: true # true if model is v2.x
  v_pred: true # true if model uses v-prediction

network:
  type: "lierla" # or "c3lier"
  rank: 4
  alpha: 1.0

train:
  precision: "bfloat16"
  noise_scheduler: "ddim" # or "ddpm", "lms", "euler_a"
  iterations: 500
  lr: 1e-4
  optimizer: "AdamW"
  lr_scheduler: "constant"

save:
  name: "van_gogh"
  path: "./output"
  per_steps: 200
  precision: "bfloat16"

logging:
  use_wandb: false
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
  resolution: 512
  dynamic_resolution: false
  batch_size: 2
```

See the [example config](/examples/config.py) for more details.

Note: You can use float16 but it is unstable and not recommended. Please use bfloat16 or float32. 

## Pretrained weights

You can use the pretrained weights on AUTOMATIC1111's webui. 

🤗 HuggingFace: https://huggingface.co/p1atdev/leco

### SDv1.5

- [Van Gogh style](https://huggingface.co/p1atdev/leco/blob/main/van_gogh_sdv15.safetensors) (trained to erase the concept of "van gogh style" on SDv1.5)

Results of `oil painting of van gogh by himself`:

![van gogh](./images/van_gogh.jpg)

<details>
<summary>
Generation settings
</summary>

```yaml
oil painting of van gogh by himself
Steps: 20, Sampler: Euler a, CFG scale: 7, Seed: 3870472781, Size: 512x512, Model hash: cc6cb27103, Model: v1-5-pruned-emaonly, Clip skip: 2, AddNet Enabled: True, AddNet Module 1: LoRA, AddNet Model 1: van_gogh_4_last(db68853d039b), AddNet Weight A 1: -1.0, AddNet Weight B 1: -1.0, Script: X/Y/Z plot, X Type: AddNet Weight 1, X Values: "-1, 0, 1", Version: v1.3.0
```

</details>

Results of `painting of scenery by monet`:

![monet](./images/van_gogh_monet.jpg)

<details>
<summary>
Generation settings
</summary>

```yaml
painting of scenery by monet
Steps: 20, Sampler: Euler a, CFG scale: 7, Seed: 1284787312, Size: 512x512, Model hash: cc6cb27103, Model: v1-5-pruned-emaonly, Clip skip: 2, AddNet Enabled: True, AddNet Module 1: LoRA, AddNet Model 1: van_gogh_4_last(db68853d039b), AddNet Weight A 1: -1.0, AddNet Weight B 1: -1.0, Script: X/Y/Z plot, X Type: AddNet Weight 1, X Values: "-1, 0, 1", Version: v1.3.0
```

</details>

### SDv2.1-768

- [Mona Lisa](https://huggingface.co/p1atdev/leco/blob/main/mona_lisa_sdv21v.safetensors) (trained to erase the concept of "mona lisa" on SDv2.1-768)


Results of `mona lisa with jewelry`:

![mona lisa](./images/mona_lisa.jpg)


<details>
<summary>
Generation settings
</summary>

```yaml
mona lisa with jewelry
Steps: 20, Sampler: Euler a, CFG scale: 7, Seed: 3630495347, Size: 512x512, Model hash: 832eb50c0c, Model: v2-1_768-ema-pruned, Clip skip: 2, AddNet Enabled: True, AddNet Module 1: LoRA, AddNet Model 1: mona_lisa2_last(393beb35c4b1), AddNet Weight A 1: -1.0, AddNet Weight B 1: -1.0, Script: X/Y/Z plot, X Type: AddNet Weight 1, X Values: "-1, 0, 1", Version: v1.3.0
```

</details>

Results of `photo of a cute cat`:

![mona lisa](./images/mona_lisa_cat.jpg)

<details>
<summary>
Generation settings
</summary>

```yaml
photo of a cute cat
Steps: 20, Sampler: Euler a, CFG scale: 7, Seed: 900866192, Size: 512x512, Model hash: 832eb50c0c, Model: v2-1_768-ema-pruned, Clip skip: 2, AddNet Enabled: True, AddNet Module 1: LoRA, AddNet Model 1: mona_lisa2_last(393beb35c4b1), AddNet Weight A 1: -1.0, AddNet Weight B 1: -1.0, Script: X/Y/Z plot, X Type: AddNet Weight 1, X Values: "-1, 0, 1", Version: v1.3.0
```

</details>

### WD1.5 beta3

- [Cat ears](https://huggingface.co/p1atdev/leco/blob/main/cat_ears_wd15beta3.safetensors) (trained to replace "1girl" with "1girl, cat ears" on WD1.5 beta3 )

Cat ears will be attached forcibly when using with 1.0~3.0 weight. 

If -1.0~-3.0, cat ears will never appear.

Training settings: see [configs](./examples/cat_ears_config.yaml).

![cat ears](./images/cat_ears.jpg)

<details>
<summary>
Generation settings
</summary>

```yaml
masterpiece, best quality, exceptional, best aesthetic, anime, 1girl, school uniform, upper body, smile
Negative prompt: worst quality, low quality, bad aesthetic, oldest, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, jpeg artifacts, signature, watermark, username, blurry
Steps: 20, Sampler: Euler a, CFG scale: 7, Seed: 4103955758, Size: 512x512, Model hash: d38e779546, Model: wd-beta3-base-fp16, Clip skip: 2, Script: X/Y/Z plot, X Type: AddNet Weight 1, X Values: "0, 2, 3, 4", Version: v1.3.0
```

</details>


- [Unreal](https://huggingface.co/p1atdev/leco/blob/main/unreal_wd15beta3.safetensors) (trained to erase "realistic", "real life", "instagram" on WD1.5 beta3)

Training settings: see [configs](./examples/unreal_config.yaml).

With "real life, instagram":

![unreal](./images/unreal_blue_cat.jpg)

<details>
<summary>
Generation settings
</summary>

```yaml
real life, instagram, masterpiece, best quality, exceptional, best aesthetic, 1girl, cat ears, blue hair, school uniform, upper body
Negative prompt: worst quality, low quality, bad aesthetic, oldest, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, jpeg artifacts, signature, watermark, username, blurry
Steps: 20, Sampler: Euler a, CFG scale: 7, Seed: 757542759, Size: 768x768, Model hash: d38e779546, Model: wd-beta3-base-fp16, Clip skip: 2, AddNet Enabled: True, AddNet Module 1: LoRA, AddNet Model 1: unreal_6_many_prompts_200steps(fff5917285da), AddNet Weight A 1: -1.0, AddNet Weight B 1: -1.0, Script: X/Y/Z plot, X Type: AddNet Weight 1, X Values: "-1, 0, 1", Version: v1.3.0
```

</details>

Without "real life, instagram":

![unreal](./images/unreal_yellow_girl.jpg)

<details>
<summary>
Generation settings
</summary>

```yaml
masterpiece, best quality, exceptional, best aesthetic,, 1girl, aqua eyes, baseball cap, blonde hair, closed mouth, earrings, green background, hat, hoop earrings, jewelry, looking at viewer, shirt, short hair, simple background, solo, upper body, yellow shirt,
Negative prompt: worst quality, low quality, bad aesthetic, oldest, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, jpeg artifacts, signature, watermark, username, blurry
Steps: 20, Sampler: Euler a, CFG scale: 7, Seed: 2867636749, Size: 768x768, Model hash: d38e779546, Model: wd-beta3-base-fp16, Clip skip: 2, AddNet Enabled: True, AddNet Module 1: LoRA, AddNet Model 1: unreal_6_many_prompts_200steps(fff5917285da), AddNet Weight A 1: -1.0, AddNet Weight B 1: -1.0, Script: X/Y/Z plot, X Type: AddNet Weight 1, X Values: "-1, 0, 1", Version: v1.3.0
```

</details>

## References

I am deeply inspired by and my work relies on the outstanding efforts of the following projects. I want to express my profound gratitude to these projects and their developers:

- https://github.com/rohitgandikota/erasing: Erasing Concepts from Diffusion Models 
  - https://huggingface.co/spaces/baulab/Erasing-Concepts-In-Diffusion: Demo of ESD 

- https://github.com/cloneofsimo/lora: Low-rank Adaptation for Fast Text-to-Image Diffusion Fine-tuning

- https://github.com/kohya-ss/sd-scripts: Training, generation and utility scripts for Stable Diffusion

- https://github.com/ntc-ai/conceptmod:  Modify Concepts from Diffusion Models using a dsl 
