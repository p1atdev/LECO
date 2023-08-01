import sys

sys.path.append(".")

from model_util import load_textual_inversion

ti = load_textual_inversion(
    pretrained_model_name_or_path=[
        "D:/Documents/Models/embeds/cat-toy.bin",
        "D:/Documents/Models/embeds/rz-neg-general.pt",
    ]
)

for model in ti:
    print(model.token)
