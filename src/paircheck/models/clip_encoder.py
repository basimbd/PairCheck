import open_clip

def get_clip_model(model_name: str, device: str = 'cuda'):
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained='openai')
    model.eval()
    model.to(device)
    return model, preprocess
