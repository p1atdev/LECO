# ref: https://huggingface.co/spaces/baulab/Erasing-Concepts-In-Diffusion/blob/main/util.py

import torch


def get_module(module, module_name):
    if isinstance(module_name, str):
        module_name = module_name.split(".")

    if len(module_name) == 0:
        return module
    else:
        module = getattr(module, module_name[0])
        return get_module(module, module_name[1:])


def set_module(module, module_name, new_module):
    if isinstance(module_name, str):
        module_name = module_name.split(".")

    if len(module_name) == 1:
        return setattr(module, module_name[0], new_module)
    else:
        module = getattr(module, module_name[0])
        return set_module(module, module_name[1:], new_module)


def freeze(module: torch.nn.Module):
    for parameter in module.parameters():
        parameter.requires_grad = False
        parameter.requires_grad_(False)


def unfreeze(module: torch.nn.Module):
    for parameter in module.parameters():
        parameter.requires_grad = True
        parameter.requires_grad_(True)
