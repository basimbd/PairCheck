import torch
import random


def freeze_model(model: torch.nn.Module):
    for param in model.parameters():
        param.requires_grad = False


def rotating_batch_sampler(items: list, batch_size: int):
    item_pool = list(items)

    while True:
        random.shuffle(item_pool)

        for idx in range(0, len(item_pool), batch_size):
            if idx + batch_size > len(item_pool):
                remaining = (idx + batch_size) - len(item_pool)
                yield item_pool[idx:idx+batch_size] + item_pool[:remaining]
            else:
                yield item_pool[idx:idx+batch_size]
