import json
import torch
import random
import logging
import argparse
import open_clip
import numpy as np
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import DataLoader
from torchmetrics.classification import (
    BinaryAUROC,
    BinaryAveragePrecision,
)

from paircheck.utils.logger import get_logger
from paircheck.models.clip_encoder import get_clip_model
from paircheck.data.loading import BalancedBatchSampler, ImageTextDataset
from paircheck.models.paircheck_head import PaircheckHead, PaircheckHeadConfig


SEED = 42

def _split_pos_neg_samples(data_path: str | Path) -> tuple[list[dict], list[dict]]:
    pos_list = []
    neg_list = []
    for data_line in Path(data_path).read_text(encoding="utf-8").strip().splitlines():
        if not data_line.strip():
            continue
        data = json.loads(data_line.strip())
        if data["label"] == 0:
            pos_list.append(data)
        elif data["label"] == 1:
            neg_list.append(data)

    return pos_list, neg_list


def _get_training_loader(data_path: str | Path, preprocessor, batch_size: int = 256) -> DataLoader:
    pos, neg = _split_pos_neg_samples(data_path)

    # The positive before negative samples must not be shuffled to ensure
    # that the BalancedBatchSampler can correctly sample from both lists. 
    dataset = pos + neg
    pos_indices = list(range(len(pos)))
    neg_indices = list(range(len(pos), len(pos) + len(neg)))

    image_paths = [item["image_path"] for item in dataset]
    captions = [item["caption"] for item in dataset]
    labels = [item["label"] for item in dataset]

    return DataLoader(
        ImageTextDataset(image_paths, captions, labels, preprocessor),
        batch_sampler=BalancedBatchSampler(pos_indices, neg_indices, batch_size),
        num_workers=4,
    )


def _get_validation_loader(data_path: str | Path, preprocessor, batch_size: int = 256) -> DataLoader:
    pos, neg = _split_pos_neg_samples(data_path)
    dataset = pos + neg
    random.Random(SEED).shuffle(dataset)

    image_paths = [item["image_path"] for item in dataset]
    captions = [item["caption"] for item in dataset]
    labels = [item["label"] for item in dataset]

    val_dataset = ImageTextDataset(image_paths, captions, labels, preprocessor)
    return DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )


def _enhance_features(img_emb: torch.Tensor, text_emb: torch.Tensor, dim: int = 0) -> torch.Tensor:
    return torch.cat([
        img_emb,
        text_emb,
        torch.abs(img_emb - text_emb),
        img_emb * text_emb,
    ], dim=dim)


def _print_experiment_metrics(epoch: int, metrics: dict, LOGGER: logging.Logger, type: str = "Validation"):
    LOGGER.info(f"=== {type} Metrics for Epoch {epoch} ===")
    for k, metric in metrics.items():
        LOGGER.info(f"    {k}: {metric.item() if isinstance(metric, torch.Tensor) else metric}")


def _calculate_metrics(scores: list | torch.Tensor, y_labels: list | torch.Tensor, auroc: BinaryAUROC, average_precision: BinaryAveragePrecision) -> dict:
    if isinstance(scores, list):
        scores = torch.cat(scores).to(auroc.device)
    if isinstance(y_labels, list):
        y_labels = torch.cat(y_labels).to(auroc.device)
    auroc_score = auroc(scores, y_labels)
    average_precision_score = average_precision(scores, y_labels)

    return {
        "auroc_score": auroc_score.item(),
        "average_precision_score": average_precision_score.item(),
        "total": scores.shape[0],
    }


def _plot_loss_curve(history: dict, output_path: str | Path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.plot(history["train_loss"], label="Train loss", color="blue")
    plt.plot(history["val_loss"], label="Validation loss", color="red")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()  


def _save_model_checkpoint(model: torch.nn.Module, model_config: PaircheckHeadConfig, metadata: dict, checkpoint_path: str | Path, scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau | None = None):
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "head_state_dict": model.state_dict(),
        "head_config": vars(model_config),
        "epoch": metadata["epoch"],
        "best_val_ap": metadata["best_val_ap"],
        "clip_model_name": metadata["clip_model_name"],
        "normalize_embeddings": True,
        "label_mapping": {"match": 0, "mismatch": 1},
    }
    if scheduler is not None:
        checkpoint.update({
            "optimizer_state_dict": scheduler.optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict()
        })
    torch.save(checkpoint, checkpoint_path)

def main(args: argparse.Namespace):
    EXPERIMENT_ID = args.clip_model_name + "_" + datetime.now().strftime('%Y%m%d_%H%M%S')
    LOGGER = get_logger(__name__, f"coco_dataset_train_{EXPERIMENT_ID}.log")

    # CLIP models
    clip_model, preprocess_train, preprocess_val = get_clip_model(args.clip_model_name, device=args.device, freeze=True)
    tokenizer = open_clip.get_tokenizer(args.clip_model_name)

    # Classification head
    model_config = PaircheckHeadConfig()
    head = PaircheckHead(model_config).to(args.device)
    loss_fn = BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(head.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

    # Data loading
    train_loader = _get_training_loader(args.train_set_path, preprocess_train, args.batch_size)
    val_loader = _get_validation_loader(args.val_set_path, preprocess_val, args.batch_size)

    # Training trackers
    best_val_ap = float("-inf")
    history = {"train_loss": [], "val_loss": [], "val_ap": [], "val_auroc": [], "lr": []}

    model_checkpoint_path = Path(args.model_checkpoint_path) / EXPERIMENT_ID

    # Training loop starts
    for epoch in tqdm(range(1, args.epochs + 1), desc="Epochs", position=0):
        epoch_history = {"train_loss": [], "val_loss": [], "val_scores": [], "val_labels": []}
        auroc = BinaryAUROC().to(args.device)
        average_precision = BinaryAveragePrecision().to(args.device)

        # Switch to training mode for the classification head
        head.train()
        for batch in tqdm(train_loader, desc="Batches", position=1, leave=False):
            images = batch["image"].to(args.device)
            captions = tokenizer(batch["caption"]).to(args.device)
            y_labels = batch["label"].to(args.device)
            img_emb = clip_model.encode_image(images, normalize=True).to(args.device)
            txt_emb = clip_model.encode_text(captions, normalize=True).to(args.device)

            inputs = _enhance_features(img_emb, txt_emb, dim=1)
            logits = head(inputs)
            loss = loss_fn(logits, y_labels.to(logits.dtype))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            epoch_history["train_loss"].append(loss.item())

        # Switch to evaluation mode for validation
        head.eval()
        with torch.no_grad():
            for val_batch in tqdm(val_loader, desc="Validation", position=1, leave=False):
                images = val_batch["image"].to(args.device)
                captions = tokenizer(val_batch["caption"]).to(args.device)
                labels = val_batch["label"].to(args.device)
                val_img_emb = clip_model.encode_image(images, normalize=True).to(args.device)
                val_text_emb = clip_model.encode_text(captions, normalize=True).to(args.device)

                val_inputs = _enhance_features(val_img_emb, val_text_emb, dim=1)
                val_logits = head(val_inputs)
                val_loss = loss_fn(val_logits, labels.to(val_logits.dtype))
                epoch_history["val_loss"].append(val_loss.item())
                epoch_history["val_scores"].append(torch.sigmoid(val_logits))
                epoch_history["val_labels"].append(labels)

        val_metric = _calculate_metrics(
            epoch_history["val_scores"], epoch_history["val_labels"], auroc, average_precision
        )
        _print_experiment_metrics(epoch, {"val_loss": np.mean(epoch_history["val_loss"]), **val_metric}, LOGGER, "Validation")

        history["train_loss"].append(np.mean(epoch_history["train_loss"]))
        history["val_loss"].append(np.mean(epoch_history["val_loss"]))
        history["val_ap"].append(val_metric["average_precision_score"])
        history["val_auroc"].append(val_metric["auroc_score"])
        history["lr"].append(scheduler.optimizer.param_groups[0]['lr'])

        save_metadata = {
            "epoch": epoch,
            "best_val_ap": best_val_ap,
            "clip_model_name": args.clip_model_name,
        }
        if val_metric["average_precision_score"] > best_val_ap:
            best_val_ap = val_metric["average_precision_score"]
            save_metadata["best_val_ap"] = best_val_ap
            _save_model_checkpoint(
                model=head,
                model_config=model_config,
                metadata=save_metadata,
                checkpoint_path=f"{model_checkpoint_path}/best_head.pt",
            )
        # Update the scheduler for possible learning rate adjustments
        scheduler.step(val_metric["average_precision_score"])
        _save_model_checkpoint(
            model=head,
            model_config=model_config,
            metadata=save_metadata,
            checkpoint_path=f"{model_checkpoint_path}/last_training_state.pt",
            scheduler=scheduler,
        )
        _plot_loss_curve(history, f"{model_checkpoint_path}/loss_curve.png")
        Path(f"{model_checkpoint_path}/training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

        LOGGER.info(f"Epoch {epoch} completed. Best validation AP: {best_val_ap:.4f}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate CLIP model on SugarCrepe dataset.")
    parser.add_argument("--clip-model-name", "-cmn", type=str, default="ViT-B-32-quickgelu", help="Name of the CLIP model to use.")
    parser.add_argument("--train-set-path", "-tsp", type=str, default="dataset/training/train/train.jsonl", help="Path to the training set JSONL file.")
    parser.add_argument("--val-set-path", "-vsp", type=str, default="dataset/training/val/val.jsonl", help="Path to the validation set JSONL file.")
    parser.add_argument("--device", type=str, default="xpu", help="Device to run the evaluation on (e.g., 'cuda', 'xpu').")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size for processing image-caption pairs.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs to train the model.")
    parser.add_argument("--learning-rate", "-lr", type=float, default=1e-3, help="Learning rate for the optimizer.")
    parser.add_argument("--model-checkpoint-path", "-mcp", type=str, default="experiments/hard-neg-train/model_checkpoints", help="Path to the model checkpoint directory.")
    args = parser.parse_args()

    main(args)
