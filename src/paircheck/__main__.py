import torch
import open_clip
from PIL import Image

from paircheck.utils.logger import LOGGER
from paircheck.data.coco import coco_loader
from paircheck.models.clip_encoder import get_clip_model


if __name__ == "__main__":
    model_name = 'ViT-B-32-quickgelu'
    device = 'xpu'
    coco_data = coco_loader(split_size=2)
    clip_model, preprocess = get_clip_model(model_name, device)
    tokenizer = open_clip.get_tokenizer(model_name)

    for img_dets in coco_data:
        img_path = img_dets["image_path"]
        img = Image.open(img_path)
        processed_img = preprocess(img).unsqueeze(0).to(device)
        LOGGER.debug(f"Image shape: {processed_img.shape}, dtype: {processed_img.dtype}, device: {processed_img.device}")
        cap_tokens = tokenizer([cap_dets["caption"] for cap_dets in img_dets["captions"]]).to(device)
        LOGGER.debug(f"Text shape: {cap_tokens.shape}, dtype: {cap_tokens.dtype}, device: {cap_tokens.device}")

        with torch.no_grad():
            img_emb = clip_model.encode_image(processed_img, normalize=True)
            text_emb = clip_model.encode_text(cap_tokens, normalize=True)

        cos_sim = img_emb @ text_emb.T

        LOGGER.info(f"Image ID: {img_dets['id']}, File: {img_path}")
        LOGGER.info(f"Captions:")
        for cap in img_dets['captions']:
            LOGGER.info(f"    {cap['caption']}")
        LOGGER.info(f"Cosine Similarity: {cos_sim}")
        print()
