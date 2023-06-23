# LECO ✏️ (beta)

Low-rank adaptation for Erasing COncepts from diffusion models.  

The original repository: [Erasing Concepts from Diffusion Models](https://github.com/rohitgandikota/erasing/tree/main)

and the project page: https://erasing.baulab.info/

(Not only for erasing concepts, but also emphasizing and swapping them by devising prompts. See [ConceptMod](https://github.com/ntc-ai/conceptmod) for more details)

## Setup

```bash
conda create -n leco python=3.10
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

## Train

You need 8GB VRAM at least.

```bash
python .\train_lora.py --prompt "van gogh style" \
--pretrained_model "runwayml/stable-diffusion-v1-5" \
--rank 4 \
--iterations 500 \
--precision bfloat16 \
--negative_guidance 1.0 \
--lr 1e-4 \
--scheduler_name "ddim" \
--save_name "van_gogh" \
--use_wandb
```

If you are using v2.x model, use `--v2` and using v-prediction model, use `--v_pred`.

<details>
<summary>
Example settings for SDv2.1
</summary>

```bash
python .\train_lora.py --prompt "nendoroid" \
--pretrained_model "stabilityai/stable-diffusion-2-1" \
--rank 4 \
--iterations 500 \
--precision bfloat16 \
--negative_guidance 1.0 \
--lr 1e-4 \
--scheduler_name "ddim" \
--save_name "nendoroid" \
--v2 \
--v_pred \
--use_wandb
```

</details>

See the [train script](/train_lora.py) for more details.

Note: You can use float16 but it is unstable and not recommended. Please use bfloat16 or float32. 

## Pretrained weights

You can use the pretrained weights on AUTOMATIC1111's webui. 

🤗 HuggingFace: https://huggingface.co/p1atdev/leco

- Van Gogh style (trained with "van gogh style" on SDv1.5)
- De-real (trained with "realistic, instagram" on WDv1.5 beta 3)

Sample images are SOON™️.

<!-- 
| Concept trained        | Base model                        | Sample |
| ---------------------- | --------------------------------- | ------ |
| "van gogh style"       | runwayml/stable-diffusion-v1-5    |        |
| "realistic, instagram" | Birchlabs/wd-1-5-beta3-unofficial |        | --> |


## References

I am deeply inspired by and my work relies on the outstanding efforts of the following projects. I want to express my profound gratitude to these projects and their developers:

- https://github.com/rohitgandikota/erasing: Erasing Concepts from Diffusion Models 
  - https://huggingface.co/spaces/baulab/Erasing-Concepts-In-Diffusion: Demo of ESD 

- https://github.com/cloneofsimo/lora: Low-rank Adaptation for Fast Text-to-Image Diffusion Fine-tuning

- https://github.com/kohya-ss/sd-scripts: Training, generation and utility scripts for Stable Diffusion

- https://github.com/ntc-ai/conceptmod:  Modify Concepts from Diffusion Models using a dsl 
