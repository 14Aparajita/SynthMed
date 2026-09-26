import os
import numpy as np
import torch
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Any
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report,
)

logger = logging.getLogger("synthmed.evaluation")


def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error for multiclass classification."""
    if len(y_true) == 0:
        return 0.0
    confidences = np.max(y_prob, axis=1)
    predictions = np.argmax(y_prob, axis=1)
    correct = (predictions == y_true).astype(float)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (confidences > lo) & (confidences <= hi)
        if mask.sum() == 0:
            continue
        bin_acc = correct[mask].mean()
        bin_conf = confidences[mask].mean()
        ece += (mask.sum() / n) * abs(bin_acc - bin_conf)
    return float(ece)


@dataclass
class ExperimentMetrics:
    experiment_name: str = ""
    accuracy: float = 0.0
    f1_score: float = 0.0
    roc_auc: float = 0.0
    ece: float = 0.0
    confusion_matrix: np.ndarray = field(default_factory=lambda: np.zeros((5, 5)))
    per_class_report: Dict[str, Any] = field(default_factory=dict)
    schema_validity_rate: float = 0.0
    repair_success_rate: float = 0.0
    mean_grounding_score: float = 0.0
    fid: float = -1.0
    train_loss: List[float] = field(default_factory=list)
    val_loss: List[float] = field(default_factory=list)
    training_time: float = 0.0
    num_synthetic_metadata: int = 0
    num_synthetic_images: int = 0
    schema_repair_enabled: bool = False
    rag_grounding_enabled: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "experiment_name": self.experiment_name,
            "accuracy": self.accuracy,
            "f1_score": self.f1_score,
            "roc_auc": self.roc_auc,
            "ece": self.ece,
            "confusion_matrix": self.confusion_matrix.tolist(),
            "per_class_report": self.per_class_report,
            "schema_validity_rate": self.schema_validity_rate,
            "repair_success_rate": self.repair_success_rate,
            "mean_grounding_score": self.mean_grounding_score,
            "fid": self.fid,
            "num_synthetic_metadata": self.num_synthetic_metadata,
            "num_synthetic_images": self.num_synthetic_images,
            "schema_repair_enabled": self.schema_repair_enabled,
            "rag_grounding_enabled": self.rag_grounding_enabled,
            "final_train_loss": self.train_loss[-1] if self.train_loss else 0.0,
            "final_val_loss": self.val_loss[-1] if self.val_loss else 0.0,
            "training_time": self.training_time,
        }


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    schema_validity: float = 0.0,
    repair_success: float = 0.0,
    grounding_score: float = 0.0,
    experiment_name: str = "",
) -> ExperimentMetrics:
    metrics = ExperimentMetrics(
        experiment_name=experiment_name,
        accuracy=float(accuracy_score(y_true, y_pred)),
        f1_score=float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        schema_validity_rate=float(schema_validity),
        repair_success_rate=float(repair_success),
        mean_grounding_score=float(grounding_score),
    )
    try:
        if len(np.unique(y_true)) > 1:
            metrics.roc_auc = float(roc_auc_score(
                y_true, y_prob, multi_class="ovr", average="weighted"
            ))
        else:
            metrics.roc_auc = 0.5
    except Exception as e:
        logger.warning(f"ROC-AUC error: {e}")
        metrics.roc_auc = 0.0

    try:
        metrics.ece = compute_ece(y_true, y_prob, n_bins=10)
    except Exception as e:
        logger.warning(f"ECE error: {e}")
        metrics.ece = 0.0

    metrics.confusion_matrix = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3, 4])
    try:
        metrics.per_class_report = classification_report(
            y_true, y_pred, output_dict=True, zero_division=0
        )
    except Exception:
        metrics.per_class_report = {}
    return metrics


def compute_fid(real_images, synthetic_images, device="cuda"):
    """FID computation. Set SYNTHMED_SKIP_FID=1 to skip. No CPU retry."""
    if os.environ.get("SYNTHMED_SKIP_FID", "0") == "1":
        return -1.0
    try:
        from torchmetrics.image.fid import FrechetInceptionDistance
    except ImportError:
        return -1.0
    if real_images.shape[0] < 2 or synthetic_images.shape[0] < 2:
        return -1.0
    if real_images.dtype != torch.uint8:
        real_images = (torch.clamp(real_images.float(), 0, 1) * 255).to(torch.uint8)
    if synthetic_images.dtype != torch.uint8:
        synthetic_images = (torch.clamp(synthetic_images.float(), 0, 1) * 255).to(torch.uint8)
    real_images = real_images.contiguous()
    synthetic_images = synthetic_images.contiguous()
    try:
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
        fid = FrechetInceptionDistance(feature=64, normalize=False).to(device)
        fid.update(real_images.to(device), real=True)
        fid.update(synthetic_images.to(device), real=False)
        score = float(fid.compute().item())
        del fid
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
        return score
    except Exception as e:
        logger.warning(f"FID failed on {device}: {e}")
        return -1.0