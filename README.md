# LECO

Low-rank adaptation for Erasing COncepts from diffusion models. 

## Setup

```bash
conda create -n leco python=3.10
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

## Train

```bash
python .\train_lora.py --prompt "van gogh style" \
--pretrained_model "runwayml/stable-diffusion-v1-5" \
--rank 4 \
--iterations 500 \
--precision bfloat16 \
--negative_guidance 1.0 \
--lr 1e-4 \
--scheduler_name "ddim" \
--save_name "anime" \
--use_wandb
```

## References

- https://github.com/rohitgandikota/erasing: Erasing Concepts from Diffusion Models 
  - https://huggingface.co/spaces/baulab/Erasing-Concepts-In-Diffusion: Demo of ESD 

- https://github.com/cloneofsimo/lora: Low-rank Adaptation for Fast Text-to-Image Diffusion Fine-tuning

- https://github.com/kohya-ss/sd-scripts: Training, generation and utility scripts for Stable Diffusion

- https://github.com/ntc-ai/conceptmod:  Modify Concepts from Diffusion Models using a dsl 

