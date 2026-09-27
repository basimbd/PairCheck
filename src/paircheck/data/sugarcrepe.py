import json
from typing import Literal
from pathlib import Path

from paircheck.utils.logger import LOGGER


def _form_image_path(img_file_name: str, dataset_dir: str = "dataset"):
    return Path(dataset_dir) / "data" / "sugarcrepe" / "images" / img_file_name


def _load_json_files_in_dir(directory: str | Path) -> dict:
    data = {}
    for file_path in Path(directory).glob("*.json"):
        with open(file_path, "r", encoding="utf-8") as f:
            data[file_path.stem] = json.load(f)

    return data


def sugarcrepe_loader(neg_type: Literal[
        "add_obj",
        "add_att",
        "replace_obj",
        "replace_att",
        "replace_rel",
        "swap_obj",
        "swap_att"
    ] | None = None, dataset_dir: str = "dataset", split_size: int | None = None, flatten: bool = False) -> list[dict] | dict[str, list[dict]]:
    annotations = {}
    if neg_type is None:
        LOGGER.info("No negative type specified. Returning all data.")
        annotations = _load_json_files_in_dir(Path(dataset_dir) / "data" / "sugarcrepe" / "annotations")
    else:
        ann_path = Path(dataset_dir) / "data" / "sugarcrepe" / "annotations" / f"{neg_type}.json"
        with open(ann_path, "r", encoding="utf-8") as f:
            annotations[neg_type] = json.load(f)

    images = {}
    images_cnt = 0
    for neg_type, idx_dict in annotations.items():
        for idx, ann_data in idx_dict.items():
            if not ann_data:
                LOGGER.warning(f"Annotation data not found for idx: {idx} in negative type: {neg_type}")
                continue
            filename = ann_data["filename"]
            img_path = _form_image_path(filename, dataset_dir)
            if not img_path.exists():
                LOGGER.warning(f"Image file not found at path: {img_path}")
                continue
            images.setdefault(neg_type, []).append({
                "id": int(Path(filename).stem),
                "image_path": img_path,
                "pos_cap": ann_data["caption"],
                "neg_cap": ann_data["negative_caption"],
            })
            images_cnt += 1
            if split_size and images_cnt >= split_size:
                break
        if split_size and images_cnt >= split_size:
            break
    if flatten:
        images = [img for sublist in images.values() for img in sublist]

    return images


if __name__ == "__main__":
    data = sugarcrepe_loader(neg_type="replace_obj", split_size=10)
    for img_list in data.values():
        for img_dets in img_list:
            LOGGER.info(f"Image ID: {img_dets['id']}, File: {img_dets['image_path']}")
            LOGGER.info(f"Positive Caption: {img_dets['pos_cap']}")
            LOGGER.info(f"Negative Caption: {img_dets['neg_cap']}")
