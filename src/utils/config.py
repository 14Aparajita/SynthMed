import yaml
from pathlib import Path
from typing import Any, Dict
from dataclasses import dataclass, field
import torch

@dataclass
class DataConfig:
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    knowledge_base_dir: str = "data/knowledge_base"
    image_size: int = 128
    num_real_train: int = 400
    num_real_test: int = 100
    num_synthetic_metadata: int = 200
    num_synthetic_images: int = 200

@dataclass
class SchemaConfig:
    schema_path: str = "config/schema/clinical_metadata.json"
    repair_enabled: bool = True
    repair_max_iterations: int = 3

@dataclass
class RetrievalConfig:
    embedder_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    top_k: int = 5
    fusion_weights: list = field(default_factory=lambda: [0.4, 0.3, 0.3])
    index_path: str = "outputs/models/faiss_index.bin"

@dataclass
class GenerationConfig:
    metadata_model: str = "distilgpt2"
    metadata_max_length: int = 256
    temperature: float = 0.7
    diffusion_timesteps: int = 100
    diffusion_image_size: int = 32
    diffusion_checkpoint: str = "outputs/models/diffusion_unet.pt"

@dataclass
class ClassifierConfig:
    model_name: str = "mobilenet_v2"
    num_classes: int = 5
    batch_size: int = 16
    epochs: int = 30
    learning_rate: float = 0.001
    weight_decay: float = 0.0001

@dataclass
class EvaluationConfig:
    metrics: list = field(default_factory=lambda: ["accuracy", "f1", "roc_auc"])
    save_results: bool = True
    results_path: str = "outputs/results"

@dataclass
class ExperimentConfig:
    name: str = "synthmed"
    seed: int = 42
    device: str = "cpu"

@dataclass
class Config:
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    data: DataConfig = field(default_factory=DataConfig)
    schema: SchemaConfig = field(default_factory=SchemaConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    classifier: ClassifierConfig = field(default_factory=ClassifierConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

def load_config(config_path: str) -> Config:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    config = Config()
    for section, values in config_dict.items():
        if hasattr(config, section):
            section_config = getattr(config, section)
            for key, value in values.items():
                if hasattr(section_config, key):
                    setattr(section_config, key, value)
    
    # Auto-detect device
    if config.experiment.device == "cpu":
        config.experiment.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    return config