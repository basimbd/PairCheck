import open_clip
import torch


def _validate_model_state(model: torch.nn.Module) -> None:
    invalid_entries = [
        name for name, value in model.state_dict().items()
        if value.is_floating_point() and not torch.isfinite(value).all().item()
    ]
    if invalid_entries:
        raise RuntimeError(f"CLIP checkpoint contains non-finite values: {', '.join(invalid_entries[:5])}")


def get_clip_model(model_name: str, device: str = 'cuda'):
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained='openai')
    _validate_model_state(model)

    model.eval()
    model.to(device)
    return model, preprocess
