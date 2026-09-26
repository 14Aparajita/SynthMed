"""
Main experiment pipeline for SynthMed.
Runs synthetic data generation and evaluation.

FID is NOT computed here. During A2-family runs, 20 real + 20 synthetic
images are saved to outputs/fid_samples/ for offline FID computation via
scripts/compute_fid_offline.py. This keeps the pipeline crash-resistant.
"""

import sys
import json
import time
import argparse
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import load_config, setup_logging, set_seed
from src.data import (
    preprocess_images, load_clinical_data,
    DRDataset, SyntheticDataset, get_augmentation_pipeline,
)
from src.schema import SchemaValidator, JSONRepairer, load_schema
from src.retrieval import DocumentEmbedder, FAISSIndexer, RAGFusion
from src.generation import MetadataGenerator, LightweightDiffusion, GroundedGenerator
from src.classifier import DRClassifier, ClassifierTrainer
from src.evaluation import compute_all_metrics, ResultsReporter, ExperimentMetrics

logger = setup_logging()


def _default_kb() -> list:
    return [
        "Diabetic retinopathy (DR) is a microvascular complication of diabetes mellitus. It is characterized by progressive damage to retinal blood vessels.",
        "Non-proliferative diabetic retinopathy (NPDR) is the early stage of DR. Findings include microaneurysms, dot and blot hemorrhages, and hard exudates.",
        "Proliferative diabetic retinopathy (PDR) is the advanced stage characterized by neovascularization of the optic disc or elsewhere in the retina.",
        "Microaneurysms are the earliest clinical sign of diabetic retinopathy. They appear as small, round, red dots in the retina.",
        "Hard exudates are yellow-white deposits of lipoproteins in the retina, often arranged in a circinate pattern around leaking microaneurysms.",
        "Cotton wool spots represent areas of retinal ischemia and appear as fluffy white lesions in the nerve fiber layer.",
        "Venous beading and intraretinal microvascular abnormalities (IRMA) are signs of severe NPDR and indicate high risk of progression to PDR.",
        "Diabetic macular edema (DME) is the leading cause of vision loss in patients with diabetic retinopathy, characterized by retinal thickening.",
        "Treatment options include anti-VEGF injections, laser photocoagulation, and vitrectomy for advanced cases with vitreous hemorrhage.",
        "Regular screening is crucial as early stages of DR are often asymptomatic but treatment can prevent vision loss.",
    ]


def _train_diffusion_minimal(model, train_df, device, epochs=20, conditioning=False):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    image_paths = train_df["image_path"].tolist()[:100]
    grade_labels = train_df["dr_grade"].tolist()[:100] if conditioning else None

    for epoch in range(epochs):
        total_loss = 0.0
        for idx, img_path in enumerate(image_paths):
            try:
                img = np.load(img_path)
                img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)
                t = torch.randint(0, model.num_timesteps, (1,), device=device)
                noise = torch.randn_like(img)
                x_noisy, noise = model.add_noise(img, t, noise)
                cond_label = None
                if conditioning and grade_labels is not None:
                    cond_label = torch.tensor([grade_labels[idx]], dtype=torch.long, device=device)
                predicted_noise = model(x_noisy, t, labels=cond_label)
                loss = torch.nn.functional.mse_loss(predicted_noise, noise)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                del img, x_noisy, predicted_noise, loss
            except Exception:
                continue
        avg_loss = total_loss / max(len(image_paths), 1)
        logger.info(f"Diffusion Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.6f}")
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    model.save_checkpoint("outputs/models/diffusion_unet.pt")


def _generate_real_pseudo_metadata(metadata_gen, train_df, validator, repairer, config):
    records = []
    for _, row in train_df.iterrows():
        grade = int(row["dr_grade"])
        try:
            recs = metadata_gen.generate_structured(dr_grade=grade, context=None, num_records=1)
            rec = recs[0] if recs else None
        except Exception as e:
            logger.warning(f"Pseudo-metadata generation failed for grade {grade}: {e}")
            rec = None

        if rec is None:
            rec = {
                "patient_id": "P00000", "age": 50, "sex": "M", "dr_grade": grade,
                "image_quality": 0.5, "left_eye": True,
                "anatomical_findings": {"microaneurysms": "none", "hemorrhages": "none", "exudates": "none"},
            }
        is_valid, _ = validator.validate(rec)
        if not is_valid and config.schema.repair_enabled:
            repaired, success = repairer.repair(rec)
            if success:
                rec.update(repaired)
        records.append(rec)
    logger.info(f"Generated {len(records)} pseudo-metadata records for real images")
    return records


def run_pipeline(config_path: str):
    config = load_config(config_path)
    set_seed(config.experiment.seed)

    logger.info(f"Starting SynthMed pipeline: {config.experiment.name}")
    logger.info(f"Device: {config.experiment.device}")

    # Step 1: Data
    logger.info("=" * 60)
    logger.info("Step 1: Data Preparation")
    logger.info("=" * 60)

    image_files = preprocess_images(
        config.data.raw_dir, config.data.processed_dir, config.data.image_size
    )
    if not image_files:
        logger.error("No images found.")
        return None

    image_mapping = {Path(f).stem: f for f in image_files}
    clinical_df = load_clinical_data(str(Path(config.data.raw_dir) / "clinical.csv"), image_mapping)

    if "image_path" in clinical_df.columns:
        def convert(p):
            if pd.isna(p):
                return p
            stem = Path(str(p)).stem
            candidate = Path(config.data.processed_dir) / f"{stem}.npy"
            return str(candidate) if candidate.exists() else str(p)
        clinical_df["image_path"] = clinical_df["image_path"].apply(convert)

    if "split" in clinical_df.columns:
        train_df = clinical_df[clinical_df["split"] == "train"].reset_index(drop=True)
        test_df = clinical_df[clinical_df["split"] == "test"].reset_index(drop=True)
    else:
        n_train = min(config.data.num_real_train, int(len(clinical_df) * 0.7))
        n_test = min(config.data.num_real_test, len(clinical_df) - n_train)
        train_df = clinical_df.iloc[:n_train].reset_index(drop=True)
        test_df = clinical_df.iloc[n_train:n_train + n_test].reset_index(drop=True)

    logger.info(f"Train: {len(train_df)}, Test: {len(test_df)}")

    # Step 2: Schema
    schema = load_schema(config.schema.schema_path)
    validator = SchemaValidator(schema)
    repairer = JSONRepairer(schema, config.schema.repair_max_iterations)

    # Step 3: Retrieval
    kb_dir = Path(config.data.knowledge_base_dir)
    documents = []
    for kb_file in list(kb_dir.glob("*.txt")) + list(kb_dir.glob("*.jsonl")):
        with open(kb_file, "r") as f:
            if kb_file.suffix == ".jsonl":
                for line in f:
                    documents.append(json.loads(line).get("text", ""))
            else:
                documents.append(f.read())
    if not documents:
        documents = _default_kb()

    embedder = DocumentEmbedder(config.retrieval.embedder_model)
    doc_embs = embedder.embed_documents(documents)
    indexer = FAISSIndexer(embedder.embedding_dim)
    indexer.add_documents(documents, doc_embs)
    rag_fusion = RAGFusion(embedder, indexer, config.retrieval.fusion_weights, config.retrieval.top_k)

    # Step 4: Metadata
    metadata_gen = MetadataGenerator(
        config.generation.metadata_model,
        config.experiment.device,
        config.generation.metadata_max_length,
        config.generation.temperature,
    )
    grounded_gen = GroundedGenerator(metadata_gen, rag_fusion)

    synthetic_records = []
    if config.data.num_synthetic_metadata > 0:
        per_grade = config.data.num_synthetic_metadata // 5
        for grade in range(5):
            generated = grounded_gen.generate_grounded(
                dr_grade=grade, num_records=per_grade,
                use_grounding=config.retrieval.rag_enabled,
            )
            for record in generated:
                is_valid, _ = validator.validate(record)
                if not is_valid and config.schema.repair_enabled:
                    repaired, success = repairer.repair(record)
                    if success:
                        record.update(repaired)
                    record["_repaired"] = success
                record["_valid"] = is_valid or record.get("_repaired", False)
            synthetic_records.extend(generated)
        logger.info(f"Generated {len(synthetic_records)} synthetic metadata records")
        logger.info(f"Schema validity: {validator.validity_rate:.3f}")
        logger.info(f"Repair success: {repairer.repair_success_rate:.3f}")

    # Step 5: Images
    synthetic_images = []
    synthetic_labels = []
    if config.data.num_synthetic_images > 0:
        diffusion = LightweightDiffusion(
            image_size=config.generation.diffusion_image_size,
            num_timesteps=config.generation.diffusion_timesteps,
            base_channels=32,
            num_classes=5,
        ).to(config.experiment.device)

        _train_diffusion_minimal(
            diffusion, train_df, config.experiment.device,
            epochs=config.generation.diffusion_epochs,
            conditioning=config.generation.conditioning_enabled,
        )

        per_grade = config.data.num_synthetic_images // 5
        for grade in range(5):
            labels_tensor = None
            if config.generation.conditioning_enabled:
                labels_tensor = torch.full(
                    (per_grade,), grade, dtype=torch.long, device=config.experiment.device
                )
            generated = diffusion.sample(
                batch_size=per_grade,
                device=config.experiment.device,
                progress=True,
                labels=labels_tensor,
                chunk_size=10,
            )
            if config.generation.diffusion_image_size != config.data.image_size:
                generated = diffusion.upscale(generated, config.data.image_size)
            gen_np = generated.cpu().numpy().transpose(0, 2, 3, 1)
            for img in gen_np:
                synthetic_images.append(img)
                synthetic_labels.append(grade)
            del generated, gen_np
            if config.experiment.device.startswith("cuda"):
                torch.cuda.empty_cache()
        logger.info(f"Generated {len(synthetic_images)} synthetic images")

        # Save samples for offline FID (no FID computed here)
        try:
            fid_dir_real = Path("outputs/fid_samples/real")
            fid_dir_synth = Path("outputs/fid_samples/synth")
            fid_dir_real.mkdir(parents=True, exist_ok=True)
            fid_dir_synth.mkdir(parents=True, exist_ok=True)

            saved_real = 0
            for p in train_df["image_path"].tolist():
                if saved_real >= 20:
                    break
                try:
                    arr = np.load(p)
                except Exception:
                    continue
                if arr.ndim != 3 or arr.shape[-1] != 3:
                    continue
                np.save(fid_dir_real / f"real_{saved_real:03d}.npy", arr.astype(np.float32))
                saved_real += 1

            for i, img in enumerate(synthetic_images[:20]):
                np.save(fid_dir_synth / f"synth_{i:03d}.npy", img.astype(np.float32))

            logger.info(f"Saved {saved_real} real + {min(20, len(synthetic_images))} synthetic samples to outputs/fid_samples/ for offline FID")
        except Exception as e:
            logger.warning(f"Failed to save FID samples: {e}")

    # Step 6: Classifier
    augmentation = get_augmentation_pipeline(strength=config.data.augmentation_strength)

    train_paths = train_df["image_path"].tolist()
    train_labels = train_df["dr_grade"].tolist()
    test_paths = test_df["image_path"].tolist()
    test_labels = test_df["dr_grade"].tolist()

    real_train = DRDataset(
        image_paths=train_paths, labels=train_labels,
        transform=augmentation, return_metadata=config.classifier.use_metadata,
    )
    test_ds = DRDataset(
        image_paths=test_paths, labels=test_labels,
        transform=None, return_metadata=config.classifier.use_metadata,
    )

    real_pseudo_metadata = []
    if config.classifier.use_metadata:
        real_pseudo_metadata = _generate_real_pseudo_metadata(
            metadata_gen, train_df, validator, repairer, config
        )
    else:
        real_pseudo_metadata = [None] * len(train_df)

    if len(synthetic_images) > 0:
        if len(synthetic_records) == 0:
            synthetic_records = [None] * len(synthetic_images)
        elif len(synthetic_records) < len(synthetic_images):
            synthetic_records = synthetic_records + [None] * (len(synthetic_images) - len(synthetic_records))
        else:
            synthetic_records = synthetic_records[:len(synthetic_images)]

        train_dataset = SyntheticDataset(
            real_dataset=real_train,
            real_metadata=real_pseudo_metadata,
            synthetic_images=synthetic_images,
            synthetic_labels=synthetic_labels,
            synthetic_metadata=synthetic_records,
        )
    else:
        train_dataset = real_train

    train_loader = DataLoader(
        train_dataset, batch_size=config.classifier.batch_size, shuffle=True, num_workers=0
    )
    test_loader = DataLoader(
        test_ds, batch_size=config.classifier.batch_size, shuffle=False, num_workers=0
    )

    model = DRClassifier(
        num_classes=config.classifier.num_classes,
        pretrained=True,
        use_metadata=config.classifier.use_metadata,
        metadata_dim=7,
    )
    trainer = ClassifierTrainer(
        model=model,
        device=config.experiment.device,
        learning_rate=config.classifier.learning_rate,
        weight_decay=config.classifier.weight_decay,
    )

    start_time = time.time()
    history = trainer.train(
        train_loader=train_loader,
        val_loader=test_loader,
        epochs=config.classifier.epochs,
        save_dir="outputs/models",
    )
    training_time = time.time() - start_time

    # Step 7: Evaluation
    trainer.model.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            if len(batch) == 3:
                images, labels, metadata = batch
                images = images.to(config.experiment.device)
                metadata = metadata.to(config.experiment.device)
                labels = labels.to(config.experiment.device)
                outputs = trainer.model(images, metadata) if config.classifier.use_metadata else trainer.model(images)
            else:
                images, labels = batch
                images = images.to(config.experiment.device)
                labels = labels.to(config.experiment.device)
                outputs = trainer.model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    metrics = compute_all_metrics(
        all_labels, all_preds, all_probs,
        schema_validity=validator.validity_rate,
        repair_success=repairer.repair_success_rate,
        grounding_score=rag_fusion.mean_grounding_score,
        experiment_name=config.experiment.name,
    )
    metrics.num_synthetic_metadata = config.data.num_synthetic_metadata
    metrics.num_synthetic_images = config.data.num_synthetic_images
    metrics.schema_repair_enabled = config.schema.repair_enabled
    metrics.rag_grounding_enabled = config.retrieval.rag_enabled
    metrics.train_loss = history["train_loss"]
    metrics.val_loss = history["val_loss"]
    metrics.training_time = training_time
    metrics.fid = -1.0

    Path("outputs/results").mkdir(parents=True, exist_ok=True)
    with open(f"outputs/results/{config.experiment.name}_metrics.json", "w") as f:
        json.dump(metrics.to_dict(), f, indent=2, default=str)

    reporter = ResultsReporter()
    reporter.add_experiment(metrics)
    reporter.generate_report()

    logger.info("=" * 60)
    logger.info("Pipeline complete!")
    logger.info(f"Accuracy: {metrics.accuracy:.4f}")
    logger.info(f"F1: {metrics.f1_score:.4f}")
    logger.info(f"ROC-AUC: {metrics.roc_auc:.4f}")
    logger.info("=" * 60)
    return metrics


def main():
    parser = argparse.ArgumentParser(description="SynthMed Pipeline")
    parser.add_argument("--config", type=str, default="config/default.yaml")
    args = parser.parse_args()
    run_pipeline(args.config)


if __name__ == "__main__":
    main()