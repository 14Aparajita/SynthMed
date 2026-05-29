import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Any
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix

@dataclass
class ExperimentMetrics:
    """Container for experiment evaluation metrics."""
    experiment_name: str = ""
    
    # Classification metrics
    accuracy: float = 0.0
    f1_score: float = 0.0
    roc_auc: float = 0.0
    confusion_matrix: np.ndarray = field(default_factory=lambda: np.zeros((5, 5)))
    
    # Schema metrics
    schema_validity_rate: float = 0.0
    repair_success_rate: float = 0.0
    
    # Grounding metrics
    mean_grounding_score: float = 0.0
    
    # Training info
    train_loss: List[float] = field(default_factory=list)
    val_loss: List[float] = field(default_factory=list)
    training_time: float = 0.0
    
    # Synthetic data stats
    num_synthetic_metadata: int = 0
    num_synthetic_images: int = 0
    schema_repair_enabled: bool = False
    rag_grounding_enabled: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to serializable dictionary."""
        return {
            'experiment_name': self.experiment_name,
            'accuracy': self.accuracy,
            'f1_score': self.f1_score,
            'roc_auc': self.roc_auc,
            'schema_validity_rate': self.schema_validity_rate,
            'repair_success_rate': self.repair_success_rate,
            'mean_grounding_score': self.mean_grounding_score,
            'num_synthetic_metadata': self.num_synthetic_metadata,
            'num_synthetic_images': self.num_synthetic_images,
            'schema_repair_enabled': self.schema_repair_enabled,
            'rag_grounding_enabled': self.rag_grounding_enabled,
            'final_train_loss': self.train_loss[-1] if self.train_loss else 0.0,
            'final_val_loss': self.val_loss[-1] if self.val_loss else 0.0,
            'training_time': self.training_time,
        }

def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    schema_validity: float = 0.0,
    repair_success: float = 0.0,
    grounding_score: float = 0.0,
    experiment_name: str = ""
) -> ExperimentMetrics:
    """Compute all evaluation metrics."""
    metrics = ExperimentMetrics(
        experiment_name=experiment_name,
        accuracy=accuracy_score(y_true, y_pred),
        f1_score=f1_score(y_true, y_pred, average='weighted'),
        schema_validity_rate=schema_validity,
        repair_success_rate=repair_success,
        mean_grounding_score=grounding_score,
    )
    
    # ROC-AUC
    try:
        metrics.roc_auc = roc_auc_score(
            y_true, y_prob, multi_class='ovr', average='weighted'
        )
    except ValueError:
        metrics.roc_auc = 0.0
    
    # Confusion matrix
    metrics.confusion_matrix = confusion_matrix(y_true, y_pred)
    
    return metrics