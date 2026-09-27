import torch
import open_clip
from PIL import Image
from tqdm import tqdm
from torchmetrics.classification import (
    BinaryAUROC,
    BinaryAveragePrecision,
)

from paircheck.utils.logger import get_logger
from paircheck.data.sugarcrepe import sugarcrepe_loader
from paircheck.models.clip_encoder import get_clip_model

LOGGER = get_logger(__name__, "PairCheck_SC_eval.log", console=False)


def _pair_image_caption_labels(sugarcrepe_data):
    image_cap_pairs = []
    y_labels = []
    for img_dets in sugarcrepe_data:
        img_path = img_dets["image_path"]

        image_cap_pairs.append((img_path, img_dets["pos_cap"]))
        y_labels.append(0)  # Positive pair

        image_cap_pairs.append((img_path, img_dets["neg_cap"]))
        y_labels.append(1)  # Negative pair

    return image_cap_pairs, y_labels


if __name__ == "__main__":
    model_name = 'ViT-B-32-quickgelu'
    device = 'xpu'
    batch_size = 256
    sugarcrepe_data = sugarcrepe_loader(flatten=True)

    clip_model, preprocess = get_clip_model(model_name, device)
    tokenizer = open_clip.get_tokenizer(model_name) 

    img_cap_pairs, y_labels = _pair_image_caption_labels(sugarcrepe_data)
    y_labels = torch.tensor(y_labels).to(device)

    all_scores = torch.tensor([]).to(device)
    for idx in tqdm(range(0, len(img_cap_pairs), batch_size), total=len(img_cap_pairs)//batch_size, desc="Processing batches"):
        img_cap_batch = img_cap_pairs[idx:idx + batch_size]
        img_batch = torch.stack([preprocess(Image.open(img_path)) for img_path, _ in img_cap_batch]).to(device)
        cap_batch = tokenizer([caption for _, caption in img_cap_batch]).to(device)

        with torch.no_grad():
            img_emb = clip_model.encode_image(img_batch, normalize=True)
            text_emb = clip_model.encode_text(cap_batch, normalize=True)

        cos_sim = img_emb @ text_emb.T
        score = 1 - cos_sim
        all_scores = torch.cat([all_scores, score.diagonal()])
    
    auroc = BinaryAUROC().to(device)
    average_precision = BinaryAveragePrecision().to(device)

    auroc_score = auroc(all_scores, y_labels)
    average_precision_score = average_precision(all_scores, y_labels)

    LOGGER.info(f"AUROC: {auroc_score.item():.4f}")
    LOGGER.info(f"Average Precision: {average_precision_score.item():.4f}")
