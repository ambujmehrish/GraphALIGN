"""GraphAlign Training Script.

End-to-end training loop for the GraphAlign model with all 6 loss functions.
Supports mixed precision, gradient accumulation, warmup + cosine scheduling,
and multi-modal training.

Usage:
    python train.py --config configs/graphalign.yaml
    python train.py --config configs/graphalign.yaml --epochs 50 --batch_size 128
"""

import argparse
import math
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import yaml

from models.graphalign_model import GraphAlignModel
from losses.graph_losses import (
    graph_contrastive_loss,
    fusion_classification_loss,
    soft_graph_binding_loss,
    anchor_distillation_loss,
    graph_knowledge_distillation_loss,
    graph_regularization_loss,
)


# ---------------------------------------------------------------------------
# Synthetic dataset for standalone training / testing
# ---------------------------------------------------------------------------

class SyntheticMultiModalDataset(Dataset):
    """Synthetic dataset for testing and development.

    Generates random tensors for each modality with random class labels.
    Replace with real data loaders for production training.
    """

    def __init__(
        self,
        num_samples: int,
        modalities: List[str],
        modality_dims: Dict[str, int],
        modality_seq_lens: Dict[str, int],
        num_classes: int = 40,
    ):
        self.num_samples = num_samples
        self.modalities = modalities
        self.modality_dims = modality_dims
        self.modality_seq_lens = modality_seq_lens
        self.num_classes = num_classes
        self.labels = torch.randint(0, num_classes, (num_samples,))

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = {"labels": self.labels[idx]}
        for m in self.modalities:
            seq_len = self.modality_seq_lens[m]
            dim = self.modality_dims[m]
            sample[m] = torch.randn(seq_len, dim)
            sample[f"{m}_labels"] = self.labels[idx]
        return sample


# ---------------------------------------------------------------------------
# Warmup + Cosine Annealing LR Scheduler
# ---------------------------------------------------------------------------

class WarmupCosineScheduler:
    """Learning rate scheduler with linear warmup and cosine annealing."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
        base_lr: float,
        min_lr: float = 0.0,
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.current_step = 0

    def step(self):
        self.current_step += 1
        lr = self._compute_lr()
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        return lr

    def _compute_lr(self) -> float:
        if self.current_step <= self.warmup_steps:
            # Linear warmup
            return self.base_lr * self.current_step / max(1, self.warmup_steps)
        else:
            # Cosine annealing
            progress = (self.current_step - self.warmup_steps) / max(
                1, self.total_steps - self.warmup_steps
            )
            progress = min(progress, 1.0)
            return self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (
                1.0 + math.cos(math.pi * progress)
            )


# ---------------------------------------------------------------------------
# Loss computation
# ---------------------------------------------------------------------------

def compute_total_loss(
    model: GraphAlignModel,
    outputs: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    active_modalities: List[str],
    config: dict,
    labels_available: bool = True,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compute the total GraphAlign loss from all 6 loss components.

    Args:
        model: The GraphAlign model (needed for fusion classifiers).
        outputs: Model forward pass outputs.
        batch: Input batch with data and labels.
        active_modalities: List of modalities present in this batch.
        config: Training configuration dict.
        labels_available: Whether classification labels are available.

    Returns:
        total_loss: Scalar loss tensor for backprop.
        losses_dict: Dict of individual loss values for logging.
    """
    training_cfg = config["training"]
    temperature = training_cfg["temperature"]

    total_loss = torch.tensor(0.0, device=next(iter(outputs.values())).device)
    losses_dict = {}

    # Collect relationship graphs
    relationship_graphs = {}
    for m in active_modalities:
        key = f"graph_{m}"
        if key in outputs:
            relationship_graphs[m] = outputs[key]

    # 1. Graph Contrastive Learning (lambda_1)
    lambda_1 = training_cfg["lambda_1_graph_nce"]
    if lambda_1 > 0:
        for i, m1 in enumerate(active_modalities):
            for m2 in active_modalities[i + 1:]:
                if m1 in relationship_graphs and m2 in relationship_graphs:
                    loss = graph_contrastive_loss(
                        relationship_graphs[m1],
                        relationship_graphs[m2],
                        temperature=temperature,
                    )
                    total_loss = total_loss + lambda_1 * loss
                    losses_dict[f"graph_NCE_{m1}_{m2}"] = loss.item()

    # 2. Cross-Modal Fusion Classification Loss (lambda_2)
    lambda_2 = training_cfg["lambda_2_fusion"]
    if lambda_2 > 0 and labels_available and model.num_classes > 0:
        labels = batch["labels"]
        for i, m1 in enumerate(active_modalities):
            for m2 in active_modalities[i + 1:]:
                fused_key = f"fused_{m1}_{m2}"
                if fused_key not in outputs:
                    # Try reverse order
                    fused_key = f"fused_{m2}_{m1}"
                if fused_key in outputs:
                    pair_key = model._pair_key(m1, m2)
                    if pair_key in model.fusion_classifiers:
                        loss = fusion_classification_loss(
                            outputs[fused_key],
                            labels,
                            model.fusion_classifiers[pair_key],
                        )
                        total_loss = total_loss + lambda_2 * loss
                        losses_dict[f"fusion_{m1}_{m2}"] = loss.item()

    # 3. Soft Graph Binding (lambda_3)
    lambda_3 = training_cfg["lambda_3_soft_bind"]
    if lambda_3 > 0:
        for i, m1 in enumerate(active_modalities):
            for m2 in active_modalities[i + 1:]:
                if m1 in relationship_graphs and m2 in relationship_graphs:
                    y_m1 = batch.get(f"{m1}_labels")
                    y_m2 = batch.get(f"{m2}_labels")
                    if y_m1 is not None and y_m2 is not None:
                        loss = soft_graph_binding_loss(
                            relationship_graphs[m1],
                            relationship_graphs[m2],
                            y_m1,
                            y_m2,
                            temperature=temperature,
                        )
                        total_loss = total_loss + lambda_3 * loss
                        losses_dict[f"soft_bind_{m1}_{m2}"] = loss.item()

    # 4. Anchor Distillation (lambda_4)
    lambda_4 = training_cfg["lambda_4_anchor"]
    if lambda_4 > 0:
        if all(
            m in relationship_graphs for m in ["point", "text", "image"]
        ):
            loss = anchor_distillation_loss(
                relationship_graphs["point"],
                relationship_graphs["text"],
                relationship_graphs["image"],
                temperature=temperature,
            )
            total_loss = total_loss + lambda_4 * loss
            losses_dict["anchor_distil"] = loss.item()

    # 5. Teacher Distillation (lambda_5)
    lambda_5 = training_cfg["lambda_5_teacher"]
    if lambda_5 > 0 and model.use_teacher:
        for m in active_modalities:
            teacher_key = f"teacher_graph_{m}"
            if teacher_key in outputs and m in relationship_graphs:
                loss = graph_knowledge_distillation_loss(
                    outputs[teacher_key], relationship_graphs[m]
                )
                total_loss = total_loss + lambda_5 * loss
                losses_dict[f"teacher_distil_{m}"] = loss.item()

    # 6. Graph Regularization (lambda_6)
    lambda_6 = training_cfg["lambda_6_reg"]
    if lambda_6 > 0:
        for m in active_modalities:
            if m in relationship_graphs:
                loss = graph_regularization_loss(
                    relationship_graphs[m],
                    lambda_sparse=training_cfg["lambda_sparse"],
                    lambda_cluster=training_cfg["lambda_cluster"],
                    lambda_rank=training_cfg["lambda_rank"],
                )
                total_loss = total_loss + lambda_6 * loss
                losses_dict[f"graph_reg_{m}"] = loss.item()

    losses_dict["total"] = total_loss.item()
    return total_loss, losses_dict


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: GraphAlignModel,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: WarmupCosineScheduler,
    scaler: torch.cuda.amp.GradScaler,
    config: dict,
    epoch: int,
    device: torch.device,
) -> Dict[str, float]:
    """Train for one epoch.

    Args:
        model: GraphAlign model.
        dataloader: Training data loader.
        optimizer: Optimizer.
        scheduler: LR scheduler.
        scaler: GradScaler for mixed precision.
        config: Full configuration dict.
        epoch: Current epoch number.
        device: Training device.

    Returns:
        Dict of average losses for this epoch.
    """
    model.train()
    training_cfg = config["training"]
    accum_iter = training_cfg.get("accum_iter", 1)
    max_grad_norm = training_cfg.get("max_grad_norm", 1.0)
    log_interval = training_cfg.get("log_interval", 50)
    use_amp = training_cfg.get("use_amp", True) and device.type == "cuda"

    epoch_losses = {}
    num_batches = 0

    for step, batch in enumerate(dataloader):
        # Move batch to device
        batch = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

        # Determine active modalities in this batch
        active_modalities = [
            m for m in model.modalities if m in batch
        ]

        # Forward pass with mixed precision
        with torch.amp.autocast("cuda" if device.type == "cuda" else "cpu", enabled=use_amp):
            outputs = model(batch)
            loss, losses_dict = compute_total_loss(
                model, outputs, batch, active_modalities, config
            )
            loss = loss / accum_iter

        # Backward pass
        if use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # Optimizer step (with gradient accumulation)
        if (step + 1) % accum_iter == 0 or (step + 1) == len(dataloader):
            if use_amp:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_grad_norm
                )
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_grad_norm
                )
                optimizer.step()

            optimizer.zero_grad()
            lr = scheduler.step()

        # Accumulate losses
        for k, v in losses_dict.items():
            if k not in epoch_losses:
                epoch_losses[k] = 0.0
            epoch_losses[k] += v
        num_batches += 1

        # Log
        if (step + 1) % log_interval == 0:
            avg_total = epoch_losses.get("total", 0.0) / num_batches
            current_lr = optimizer.param_groups[0]["lr"]
            print(
                f"  Epoch {epoch} | Step {step + 1}/{len(dataloader)} | "
                f"Loss: {avg_total:.4f} | LR: {current_lr:.2e}"
            )

    # Average over epoch
    for k in epoch_losses:
        epoch_losses[k] /= max(num_batches, 1)

    return epoch_losses


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> dict:
    """Load YAML configuration."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="GraphAlign Training")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/graphalign.yaml",
        help="Path to config file",
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--num_samples",
        type=int,
        default=2048,
        help="Number of synthetic samples (for testing)",
    )
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Override from command line
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    if args.batch_size is not None:
        config["training"]["batch_size"] = args.batch_size
    if args.lr is not None:
        config["training"]["learning_rate"] = args.lr

    # Device
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Build model
    modalities = config["modalities"]
    modality_input_dims = config["modality_input_dims"]
    encoder_cfg = config["encoder"]
    pooling_cfg = config["pooling"]
    expansion_cfg = config["expansion"]
    training_cfg = config["training"]

    model = GraphAlignModel(
        modalities=modalities,
        modality_input_dims=modality_input_dims,
        encoder_hidden_dim=encoder_cfg["hidden_dim"],
        encoder_output_dim=encoder_cfg["output_dim"],
        encoder_num_layers=encoder_cfg["num_layers"],
        encoder_num_heads=encoder_cfg["num_heads"],
        encoder_dropout=encoder_cfg["dropout"],
        graph_target_length=pooling_cfg["target_length"],
        graph_dim=pooling_cfg["graph_dim"],
        graph_pooling_heads=pooling_cfg["num_heads"],
        graph_pooling_dropout=pooling_cfg["dropout"],
        expansion_order=expansion_cfg["order"],
        num_classes=config.get("num_classes", 0),
        use_teacher=config.get("use_teacher", False),
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,} total, {num_trainable:,} trainable")

    # Build synthetic dataset
    data_cfg = config.get("data", {})
    modality_seq_lens = {
        "image": data_cfg.get("image_seq_len", 197),
        "text": data_cfg.get("text_seq_len", 77),
        "point": data_cfg.get("point_seq_len", 1024),
    }
    # Only include configured modalities
    modality_seq_lens = {
        m: modality_seq_lens.get(m, 128) for m in modalities
    }

    dataset = SyntheticMultiModalDataset(
        num_samples=args.num_samples,
        modalities=modalities,
        modality_dims=modality_input_dims,
        modality_seq_lens=modality_seq_lens,
        num_classes=config.get("num_classes", 40),
    )

    dataloader = DataLoader(
        dataset,
        batch_size=training_cfg["batch_size"],
        shuffle=True,
        num_workers=0,
        drop_last=True,
        pin_memory=(device.type == "cuda"),
    )

    # Build optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_cfg["learning_rate"],
        weight_decay=training_cfg["weight_decay"],
        betas=tuple(training_cfg.get("betas", [0.9, 0.999])),
        eps=training_cfg.get("eps", 1e-8),
    )

    # Build scheduler
    total_steps = len(dataloader) * training_cfg["epochs"]
    scheduler = WarmupCosineScheduler(
        optimizer=optimizer,
        warmup_steps=training_cfg["warmup_steps"],
        total_steps=total_steps,
        base_lr=training_cfg["learning_rate"],
        min_lr=training_cfg.get("min_lr", 0.0),
    )

    # GradScaler for mixed precision
    scaler = torch.amp.GradScaler("cuda", enabled=(
        training_cfg.get("use_amp", True) and device.type == "cuda"
    ))

    # Output directory
    output_dir = config.get("output_dir", "./output/graphalign")
    checkpoint_dir = config.get("checkpoint_dir", os.path.join(output_dir, "checkpoints"))
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Training loop
    print(f"\nStarting training for {training_cfg['epochs']} epochs")
    print(f"Batch size: {training_cfg['batch_size']}, Accum iter: {training_cfg.get('accum_iter', 1)}")
    print(f"Total steps: {total_steps}, Warmup steps: {training_cfg['warmup_steps']}")
    print(f"Modalities: {modalities}")
    print("-" * 60)

    best_loss = float("inf")

    for epoch in range(training_cfg["epochs"]):
        t0 = time.time()

        epoch_losses = train_one_epoch(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            config=config,
            epoch=epoch,
            device=device,
        )

        elapsed = time.time() - t0
        total_loss = epoch_losses.get("total", 0.0)

        print(f"\nEpoch {epoch} complete | Total Loss: {total_loss:.4f} | Time: {elapsed:.1f}s")

        # Print individual losses
        for k, v in sorted(epoch_losses.items()):
            if k != "total":
                print(f"  {k}: {v:.4f}")

        # Save checkpoint
        if (epoch + 1) % max(1, training_cfg.get("save_interval", 10)) == 0:
            ckpt_path = os.path.join(checkpoint_dir, f"epoch_{epoch:04d}.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": total_loss,
                    "config": config,
                },
                ckpt_path,
            )
            print(f"  Saved checkpoint: {ckpt_path}")

        # Track best
        if total_loss < best_loss:
            best_loss = total_loss
            best_path = os.path.join(checkpoint_dir, "best.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": total_loss,
                    "config": config,
                },
                best_path,
            )
            print(f"  New best loss: {best_loss:.4f}")

        print("-" * 60)

    print(f"\nTraining complete. Best loss: {best_loss:.4f}")


if __name__ == "__main__":
    main()
