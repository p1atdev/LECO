from prompt_util import load_prompts_from_yaml
from config_util import load_config_from_yaml

config = load_config_from_yaml("config.yaml")

print(config)

prompts = load_prompts_from_yaml(config.prompts_file)

print(prompts)

# import torch
# from safetensors.torch import load_file
# import json

# ckpt = load_file(
#     "D:/Documents/Python/loesd/output/van_gogh_2/van_gogh_2_last.safetensors"
# )
# # ckpt = torch.load("./output/last.pt")


# print((ckpt))

# # kohya
# # [
# #     "lora_unet_down_blocks_0_attentions_0_proj_in.alpha",
# #     "lora_unet_down_blocks_0_attentions_0_proj_in.lora_down.weight",
# #     "lora_unet_down_blocks_0_attentions_0_proj_in.lora_up.weight",
# #     "lora_unet_down_blocks_0_attentions_0_proj_out.alpha",
# #     "lora_unet_down_blocks_0_attentions_0_proj_out.lora_down.weight",
# # ]

# # led
# # [
# #     "model.unet.conv_in.weight",
# #     "model.unet.conv_in.bias",
# #     "model.unet.time_embedding.linear_1.weight",
# #     "model.unet.time_embedding.linear_1.bias",
# #     "model.unet.time_embedding.linear_2.weight",
# # ]

# # lora_unet_mid_block_attentions_0_transformer_blocks_0_attn1_to_q.lora_down.weight
# # model.unet.up_blocks.2.attentions.0.transformer_blocks.0.attn2.to_out.0.lora_up.weight
