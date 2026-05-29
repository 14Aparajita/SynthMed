"""
Main experiment pipeline for SynthMed.
Runs the complete synthetic data generation and evaluation workflow.
"""

import sys
import json
import time
import argparse
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import load_config, setup_logging, set_seed
from src.data import preprocess_images, load_clinical_data, DRDataset, get_augmentation_pipeline
from src.schema import SchemaValidator, JSONRepairer, load_schema
from src.retrieval import DocumentEmbedder, FAISSIndexer, RAGFusion
from src.generation import MetadataGenerator, LightweightDiffusion, GroundedGenerator
from src.classifier import DRClassifier, ClassifierTrainer
from src.evaluation import compute_all_metrics, ResultsReporter, ExperimentMetrics

logger = setup_logging()

def run_pipeline(config_path: str):
    """Run complete SynthMed pipeline."""
    config = load_config(config_path)
    set_seed(config.experiment.seed)
    
    logger.info(f"Starting SynthMed pipeline: {config.experiment.name}")
    logger.info(f"Device: {config.experiment.device}")
    
    # ============================================================
    # Step 1: Data Preparation
    # ============================================================
    logger.info("="*60)
    logger.info("Step 1: Data Preparation")
    logger.info("="*60)
    
    # Preprocess images - convert to .npy format
    image_files = preprocess_images(
        config.data.raw_dir,
        config.data.processed_dir,
        config.data.image_size
    )
    
    if not image_files:
        logger.error("No images found! Please check your data directory.")
        return None
    
    # Create image mapping (stem -> full path)
    image_mapping = {Path(f).stem: f for f in image_files}
    logger.info(f"Processed {len(image_mapping)} images")
    
    # Load clinical data
    clinical_df = load_clinical_data(
        str(Path(config.data.raw_dir) / "clinical.csv"),
        image_mapping
    )
    
    # Ensure image paths point to processed files
    if 'image_path' in clinical_df.columns:
        # Convert raw paths to processed .npy paths
        def convert_to_processed_path(path):
            if pd.isna(path):
                return path
            path_str = str(path)
            stem = Path(path_str).stem
            processed_path = Path(config.data.processed_dir) / f"{stem}.npy"
            if processed_path.exists():
                return str(processed_path)
            return path_str
        
        clinical_df['image_path'] = clinical_df['image_path'].apply(convert_to_processed_path)
    
    # Split real data
    n_train = config.data.num_real_train
    n_test = config.data.num_real_test
    
    # Use all available data if requested numbers exceed dataset
    total_available = len(clinical_df)
    n_train = min(n_train, int(total_available * 0.7))
    n_test = min(n_test, total_available - n_train)
    
    train_indices = range(n_train)
    test_indices = range(n_train, n_train + n_test)
    
    train_df = clinical_df.iloc[train_indices].reset_index(drop=True)
    test_df = clinical_df.iloc[test_indices].reset_index(drop=True)
    
    logger.info(f"Train samples: {len(train_df)}, Test samples: {len(test_df)}")
    logger.info(f"Class distribution - Train: {train_df['dr_grade'].value_counts().to_dict()}")
    
        
    # ============================================================
    # Step 2: Schema Validation Setup
    # ============================================================
    logger.info("="*60)
    logger.info("Step 2: Schema Validation Setup")
    logger.info("="*60)
    
    schema = load_schema(config.schema.schema_path)
    validator = SchemaValidator(schema)
    repairer = JSONRepairer(schema, config.schema.repair_max_iterations)
    
    # ============================================================
    # Step 3: Knowledge Base & Retrieval Setup
    # ============================================================
    logger.info("="*60)
    logger.info("Step 3: Retrieval System Setup")
    logger.info("="*60)
    
    # Load knowledge base
    kb_dir = Path(config.data.knowledge_base_dir)
    kb_files = list(kb_dir.glob("*.txt")) + list(kb_dir.glob("*.jsonl"))
    
    documents = []
    for kb_file in kb_files:
        with open(kb_file, 'r') as f:
            if kb_file.suffix == '.jsonl':
                for line in f:
                    data = json.loads(line)
                    documents.append(data.get('text', ''))
            else:
                documents.append(f.read())
    
    if not documents:
        # Create default knowledge base
        documents = _create_default_knowledge_base()
        logger.warning("No knowledge base found. Using default clinical documents.")
    
    logger.info(f"Loaded {len(documents)} knowledge base documents")
    
    # Setup retrieval
    embedder = DocumentEmbedder(config.retrieval.embedder_model)
    doc_embeddings = embedder.embed_documents(documents)
    
    indexer = FAISSIndexer(embedder.embedding_dim)
    indexer.add_documents(documents, doc_embeddings)
    
    rag_fusion = RAGFusion(
        embedder,
        indexer,
        config.retrieval.fusion_weights,
        config.retrieval.top_k
    )
    
    # ============================================================
    # Step 4: Synthetic Metadata Generation
    # ============================================================
    logger.info("="*60)
    logger.info("Step 4: Synthetic Metadata Generation")
    logger.info("="*60)
    
    metadata_gen = MetadataGenerator(
        config.generation.metadata_model,
        config.experiment.device,
        config.generation.metadata_max_length,
        config.generation.temperature
    )
    
    grounded_gen = GroundedGenerator(metadata_gen, rag_fusion)
    
    synthetic_records = []
    if config.data.num_synthetic_metadata > 0:
        # Generate records for each DR grade
        records_per_grade = config.data.num_synthetic_metadata // 5
        
        for grade in range(5):
            generated = grounded_gen.generate_grounded(
                dr_grade=grade,
                num_records=records_per_grade,
                use_grounding=True
            )
            
            # Validate and repair
            for record in generated:
                is_valid, errors = validator.validate(record)
                if not is_valid and config.schema.repair_enabled:
                    repaired, success = repairer.repair(record)
                    if success:
                        record.update(repaired)
                    record['_repaired'] = success
                record['_valid'] = is_valid or record.get('_repaired', False)
            
            synthetic_records.extend(generated)
        
        logger.info(f"Generated {len(synthetic_records)} synthetic metadata records")
        logger.info(f"Schema validity rate: {validator.validity_rate:.3f}")
        logger.info(f"Repair success rate: {repairer.repair_success_rate:.3f}")
    
    # ============================================================
    # Step 5: Synthetic Image Generation
    # ============================================================
    logger.info("="*60)
    logger.info("Step 5: Synthetic Image Generation")
    logger.info("="*60)
    
    synthetic_images = []
    synthetic_labels = []
    
    if config.data.num_synthetic_images > 0:
        diffusion = LightweightDiffusion(
            image_size=config.generation.diffusion_image_size,
            num_timesteps=config.generation.diffusion_timesteps
        ).to(config.experiment.device)
        
        # Train diffusion model on real images (simplified for laptop)
        logger.info("Training diffusion model (minimal training for demo)...")
        _train_diffusion_minimal(
            diffusion,
            train_df,
            config.experiment.device,
            epochs=5  # Minimal epochs for laptop
        )
        
        # Generate images
        images_per_grade = config.data.num_synthetic_images // 5
        
        for grade in range(5):
            # Generate 32x32 images
            generated = diffusion.sample(
                batch_size=images_per_grade,
                device=config.experiment.device,
                progress=True
            )
            
            # Upscale to target size
            generated = diffusion.upscale(
                generated,
                config.data.image_size
            )
            
            # Convert to numpy
            gen_np = generated.cpu().numpy()
            gen_np = gen_np.transpose(0, 2, 3, 1)  # NHWC
            
            for img in gen_np:
                synthetic_images.append(img)
                synthetic_labels.append(grade)
        
        logger.info(f"Generated {len(synthetic_images)} synthetic images")
    
    # ============================================================
    # Step 6: Classifier Training
    # ============================================================
    logger.info("="*60)
    logger.info("Step 6: Classifier Training")
    logger.info("="*60)
    
    # Prepare datasets
    augmentation = get_augmentation_pipeline()
    
    # Real train dataset
    train_paths = train_df['image_path'].tolist()
    train_labels = train_df['dr_grade'].tolist()
    
    real_train_dataset = DRDataset(
        train_paths,
        train_labels,
        transform=augmentation
    )
    
    # Test dataset
    test_paths = test_df['image_path'].tolist()
    test_labels = test_df['dr_grade'].tolist()
    
    test_dataset = DRDataset(test_paths, test_labels)
    
    # Create loaders
    train_loader = DataLoader(
        real_train_dataset,
        batch_size=config.classifier.batch_size,
        shuffle=True,
        num_workers=0  # Laptop-friendly
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.classifier.batch_size,
        shuffle=False,
        num_workers=0
    )
    
    # Train classifier
    model = DRClassifier(num_classes=config.classifier.num_classes)
    trainer = ClassifierTrainer(
        model,
        config.experiment.device,
        config.classifier.learning_rate,
        config.classifier.weight_decay
    )
    
    logger.info("Training baseline classifier...")
    start_time = time.time()
    
    history = trainer.train(
        train_loader,
        test_loader,
        epochs=config.classifier.epochs,
        save_dir="outputs/models"
    )
    
    training_time = time.time() - start_time
    
    # ============================================================
    # Step 7: Evaluation
    # ============================================================
    logger.info("="*60)
    logger.info("Step 7: Evaluation")
    logger.info("="*60)
    
    # Evaluate on test set
    trainer.model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(config.experiment.device)
            outputs = trainer.model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Compute metrics
    metrics = compute_all_metrics(
        all_labels,
        all_preds,
        all_probs,
        schema_validity=validator.validity_rate,
        repair_success=repairer.repair_success_rate,
        grounding_score=rag_fusion.mean_grounding_score,
        experiment_name=config.experiment.name
    )
    
    metrics.num_synthetic_metadata = config.data.num_synthetic_metadata
    metrics.num_synthetic_images = config.data.num_synthetic_images
    metrics.schema_repair_enabled = config.schema.repair_enabled
    metrics.rag_grounding_enabled = True
    metrics.train_loss = history['train_loss']
    metrics.val_loss = history['val_loss']
    metrics.training_time = training_time
    
    # Generate report
    reporter = ResultsReporter()
    reporter.add_experiment(metrics)
    reporter.generate_report()
    
    logger.info("="*60)
    logger.info("Pipeline complete!")
    logger.info(f"Accuracy: {metrics.accuracy:.4f}")
    logger.info(f"F1 Score: {metrics.f1_score:.4f}")
    logger.info(f"ROC-AUC: {metrics.roc_auc:.4f}")
    logger.info(f"Schema Validity: {metrics.schema_validity_rate:.4f}")
    logger.info(f"Repair Success: {metrics.repair_success_rate:.4f}")
    logger.info(f"Grounding Score: {metrics.mean_grounding_score:.4f}")
    logger.info("="*60)
    
    return metrics

def _create_default_knowledge_base() -> list:
    """Create default DR knowledge base documents."""
    return [
        "Diabetic retinopathy (DR) is a microvascular complication of diabetes mellitus. "
        "It is characterized by progressive damage to retinal blood vessels.",
        
        "Non-proliferative diabetic retinopathy (NPDR) is the early stage of DR. "
        "Findings include microaneurysms, dot and blot hemorrhages, and hard exudates.",
        
        "Proliferative diabetic retinopathy (PDR) is the advanced stage characterized "
        "by neovascularization of the optic disc or elsewhere in the retina.",
        
        "Microaneurysms are the earliest clinical sign of diabetic retinopathy. "
        "They appear as small, round, red dots in the retina.",
        
        "Hard exudates are yellow-white deposits of lipoproteins in the retina, "
        "often arranged in a circinate pattern around leaking microaneurysms.",
        
        "Cotton wool spots represent areas of retinal ischemia and appear as "
        "fluffy white lesions in the nerve fiber layer.",
        
        "Venous beading and intraretinal microvascular abnormalities (IRMA) "
        "are signs of severe NPDR and indicate high risk of progression to PDR.",
        
        "Diabetic macular edema (DME) is the leading cause of vision loss in "
        "patients with diabetic retinopathy, characterized by retinal thickening.",
        
        "Treatment options include anti-VEGF injections, laser photocoagulation, "
        "and vitrectomy for advanced cases with vitreous hemorrhage.",
        
        "Regular screening is crucial as early stages of DR are often asymptomatic "
        "but treatment can prevent vision loss.",
    ]

def _train_diffusion_minimal(
    model: LightweightDiffusion,
    train_df: pd.DataFrame,
    device: str,
    epochs: int = 5
):
    """Minimal diffusion model training for laptop."""
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    image_paths = train_df['image_path'].tolist()[:100]  # Use subset
    
    for epoch in range(epochs):
        total_loss = 0.0
        for img_path in image_paths:
            try:
                img = np.load(img_path)
                img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
                img = img.to(device)
                
                # Sample timestep
                t = torch.randint(0, model.num_timesteps, (1,), device=device)
                
                # Add noise
                noise = torch.randn_like(img)
                x_noisy, noise = model.add_noise(img, t, noise)
                
                # Predict noise
                predicted_noise = model(x_noisy, t)
                
                # Loss
                loss = torch.nn.functional.mse_loss(predicted_noise, noise)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            except Exception as e:
                continue
        
        avg_loss = total_loss / max(len(image_paths), 1)
        logger.info(f"Diffusion Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
    
    model.save_checkpoint("outputs/models/diffusion_unet.pt")

def main():
    parser = argparse.ArgumentParser(description="SynthMed Pipeline")
    parser.add_argument(
        "--config",
        type=str,
        default="config/default.yaml",
        help="Path to configuration file"
    )
    args = parser.parse_args()
    
    run_pipeline(args.config)

if __name__ == "__main__":
    main()