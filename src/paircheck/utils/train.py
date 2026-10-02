import torch


def freeze_model(model: torch.nn.Module):
    for param in model.parameters():
        param.requires_grad = False
