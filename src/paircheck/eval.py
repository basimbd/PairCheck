import os
import json
import torch
import argparse
import open_clip
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from torchmetrics.classification import (
    BinaryAUROC,
    BinaryAveragePrecision,
)

from paircheck.utils.logger import get_logger
from paircheck.data.sugarcrepe import sugarcrepe_loader
from paircheck.models.clip_encoder import get_clip_model

LOGGER = get_logger(__name__, f"RAW_CLIP_SugarCrepe_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log", console=False)


def _pair_image_caption_labels(sugarcrepe_data):
    image_cap_pairs = []
    y_labels = []
    for img_dets in sugarcrepe_data:
        img_path = img_dets["image_path"]

        image_cap_pairs.append((img_path, img_dets["pos_cap"]))
        y_labels.append(0)  # Positive pair (image-caption match)

        image_cap_pairs.append((img_path, img_dets["neg_cap"]))
        y_labels.append(1)  # Negative pair (image-caption mismatch)

    return image_cap_pairs, y_labels


def _evaluate_pairwise_accuracy(scores: torch.Tensor) -> tuple[float, float]:
    """
    scores: It is a tensor of image-caption dis-similarity scores. The scores are
    alternating between positive and corresponding negative samples. Even indices
    are positive instances and odd indices are negative instances.
    """
    assert scores.shape[0] % 2 == 0, f"Scores tensor length must be even (got {scores.shape[0]})."
    pos_scores = scores[::2]    # Even indices
    neg_scores = scores[1::2]   # Odd indices

    pair_comparison = pos_scores < neg_scores
    correct_mismatches = pair_comparison.sum()
    accuracy = pair_comparison.float().mean()

    return correct_mismatches.item(), accuracy.item()


def _evaluate_batch(img_batch: torch.Tensor, cap_batch: torch.Tensor, model: torch.nn.Module):
    with torch.no_grad():
        img_emb = model.encode_image(img_batch, normalize=True)
        text_emb = model.encode_text(cap_batch, normalize=True)

    cos_sim = img_emb @ text_emb.T
    score = 1 - cos_sim
    return score.diagonal()


def _evaluate_neg_type_set(sugarcrepe_data: dict, model: torch.nn.Module, preprocess, tokenizer, device: str, batch_size: int = 256):
    img_cap_pairs, y_labels = _pair_image_caption_labels(sugarcrepe_data)
    y_labels = torch.tensor(y_labels).to(device)

    all_scores = []
    for idx in tqdm(range(0, len(img_cap_pairs), batch_size), desc="Processing batches", position=1, leave=False):
        img_cap_batch = img_cap_pairs[idx:idx + batch_size]
        img_batch = torch.stack([preprocess(Image.open(img_path)) for img_path, _ in img_cap_batch]).to(device)
        cap_batch = tokenizer([caption for _, caption in img_cap_batch]).to(device)

        all_scores.append(_evaluate_batch(img_batch, cap_batch, model))

    return torch.cat(all_scores), y_labels


def _calculate_metrics(scores: torch.Tensor, y_labels: torch.Tensor, device: str):
    auroc = BinaryAUROC().to(device)
    average_precision = BinaryAveragePrecision().to(device)

    auroc_score = auroc(scores, y_labels)
    average_precision_score = average_precision(scores, y_labels)
    correct_mismatches, accuracy = _evaluate_pairwise_accuracy(scores)

    return {
        "auroc_score": auroc_score,
        "average_precision_score": average_precision_score,
        "accuracy": accuracy,
        "total": scores.shape[0],
        "correct_mismatches": correct_mismatches,
    }


def _print_experiment_metrics(neg_type_metrics: dict):
    LOGGER.info("\n=== Experiment Final Metrics Summary ===")
    for neg_type, metrics in neg_type_metrics.items():
        LOGGER.info(f"--- Metrics for negative type: {neg_type} ---")
        LOGGER.info(f"    AUROC: {metrics['auroc_score'].item():.4f}")
        LOGGER.info(f"    Average Precision: {metrics['average_precision_score'].item():.4f}")
        LOGGER.info(f"    Accuracy: {100 * metrics['accuracy']:.4f}")
        LOGGER.info(f"    Correct Mismatches: {metrics['correct_mismatches']}")
        LOGGER.info(f"    Total Samples: {metrics['total']}\n")


def _serialize_metrics(neg_type_metrics: dict) -> dict:
    serialized_metrics = {}
    for neg_type, metrics in neg_type_metrics.items():
        serialized_metrics[neg_type] = {
            "auroc_score": metrics["auroc_score"].item(),
            "average_precision_score": metrics["average_precision_score"].item(),
            "accuracy": metrics["accuracy"],
            "total": metrics["total"],
            "correct_mismatches": metrics["correct_mismatches"],
        }
    return serialized_metrics


def evaluate(args):
    device = args.device
    sugarcrepe_data = sugarcrepe_loader()

    clip_model, preprocess = get_clip_model(args.model_name, device)
    tokenizer = open_clip.get_tokenizer(args.model_name)

    all_scores = {"scores": [], "labels": []}
    neg_group_scores = {}
    neg_type_metrics = {}
    for neg_type, data in tqdm(sugarcrepe_data.items(), desc="Processing negative types", position=0):
        LOGGER.debug(f"Evaluating negative type: {neg_type} with {len(data)} samples.")
        scores, labels = _evaluate_neg_type_set(data, clip_model, preprocess, tokenizer, device, args.batch_size)

        current_metrics = _calculate_metrics(scores, labels, device)
        neg_type_metrics[neg_type] = current_metrics

        neg_group_key = neg_type.split("_")[0]
        neg_group_scores[neg_group_key] = neg_group_scores.get(neg_group_key, {"scores": [], "labels": []})
        neg_group_scores[neg_group_key]["scores"].append(scores)
        neg_group_scores[neg_group_key]["labels"].append(labels)

        all_scores["scores"].append(scores)
        all_scores["labels"].append(labels)

    neg_group_metrics = {}
    for neg_group, group_data in neg_group_scores.items():
        neg_group_metrics[neg_group] = _calculate_metrics(torch.cat(group_data["scores"]), torch.cat(group_data["labels"]), device)

    _print_experiment_metrics(neg_type_metrics)
    _print_experiment_metrics(neg_group_metrics)

    if args.exp_output_dir:
        output_path = "experiments/" + args.exp_output_dir + "/result_" + datetime.now().strftime('%Y%m%d_%H%M%S') + ".json"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            final_metrics = {
                **_serialize_metrics({
                    "all_scores": _calculate_metrics(torch.cat(all_scores["scores"]), torch.cat(all_scores["labels"]), device)
                }),
                "score_by_group": _serialize_metrics(neg_group_metrics),
                **_serialize_metrics(neg_type_metrics)
            }
            json.dump(final_metrics, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate CLIP model on SugarCrepe dataset.")
    parser.add_argument("--model_name", type=str, default="ViT-B-32-quickgelu", help="Name of the CLIP model to use.")
    parser.add_argument("--device", type=str, default="xpu", help="Device to run the evaluation on (e.g., 'cuda', 'xpu').")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for processing image-caption pairs.")
    parser.add_argument("--exp_output_dir", "-eod", type=str, default=None, help="Path to save the evaluation metrics as a JSON file.")
    args = parser.parse_args()

    evaluate(args)
