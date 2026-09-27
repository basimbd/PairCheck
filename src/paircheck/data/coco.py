import json
from typing import Literal
from pathlib import Path

from paircheck.utils.logger import LOGGER

def _load_coco_config(config_file: str | Path) -> dict:
    config_file = Path(config_file)
    try:
        coco_config = json.loads(config_file.read_text(encoding='utf-8'))
    except json.JSONDecodeError as e:
        LOGGER.warning(f"COCO config JSON file ({str(config_file)}) loading failed: {e}")
        coco_config = {}
    return coco_config


def _form_image_path(img_file_name: str, dataset_dir: str = "dataset", split_type: Literal["train", "val", "test"] = "train"):
    return Path(dataset_dir) / "data" / "images" / split_type / img_file_name


def _build_image_index(coco: dict):
    return {
        image["id"]: image
        for image in coco["images"]
    }


def _build_captions_index(coco: dict):
    captions_index = {}
    for ann in coco["annotations"]:
        img_id = ann["image_id"]
        captions_index.setdefault(img_id, []).append(ann)
    return captions_index


def _resolve_captions_path(split_type: Literal["train", "val", "test"] = "train", dataset_dir: str = "dataset"):
    split_type = "val" if split_type == "test" else "train"
    return Path(dataset_dir) / "annotations" / f"captions_{split_type}2017.json"


def coco_loader(split_type: Literal["train", "val", "test"] = "train", split_size: int = 40_000, dataset_dir: str = "dataset"):
    coco_captions_config = _load_coco_config(_resolve_captions_path(split_type, dataset_dir))
    coco_image_indexed = _build_image_index(coco_captions_config)
    coco_captions_indexed = _build_captions_index(coco_captions_config)
    coco_split_config = _load_coco_config(f"{dataset_dir}/data/split.json")

    if not (coco_captions_config and coco_split_config):
        LOGGER.warning("COCO config loading failed. Please check the dataset directory and config files.")
        return None

    images = []
    for img_id in coco_split_config["splits"][split_type]:
        img_details = coco_image_indexed.get(img_id)
        if not img_details:
            LOGGER.warning(f"Image details not found for ID: {img_id}")
            continue
        img_path = _form_image_path(img_details["file_name"], dataset_dir, split_type)
        if not img_path.exists():
            LOGGER.warning(f"Image file not found at path: {img_path}")
            continue
        images.append({
            "id": img_id,
            "image_path": img_path,
            "height": img_details.get("height"),
            "width": img_details.get("width"),
            "captions": coco_captions_indexed.get(img_id, [])
        })
        if len(images) >= split_size:
            break
    return images
