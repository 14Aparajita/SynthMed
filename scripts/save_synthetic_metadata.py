"""
Generate synthetic metadata (500 records) using the exact SynthMed pipeline
and save it to disk for downstream analysis.

Runs the same DistilGPT-2 + schema repair + RAG pipeline used in the paper,
so the saved records are identical in distribution to the ones used in
experiment A2.
"""

import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from src.utils import load_config, setup_logging, set_seed
from src.schema import SchemaValidator, JSONRepairer, load_schema
from src.retrieval import DocumentEmbedder, FAISSIndexer, RAGFusion
from src.generation import MetadataGenerator, GroundedGenerator

logger = setup_logging()

def main():
    config = load_config("config/exp_A2_synthmed_100real_500syn.yaml")
    set_seed(config.experiment.seed)

    # Schema
    schema = load_schema(config.schema.schema_path)
    validator = SchemaValidator(schema)
    repairer = JSONRepairer(schema, config.schema.repair_max_iterations)

    # Retrieval
    kb_dir = Path(config.data.knowledge_base_dir)
    docs = []
    for f in list(kb_dir.glob("*.jsonl")) + list(kb_dir.glob("*.txt")):
        if f.suffix == ".jsonl":
            with open(f) as fh:
                for line in fh:
                    docs.append(json.loads(line).get("text", ""))
        else:
            docs.append(f.read_text())

    embedder = DocumentEmbedder(config.retrieval.embedder_model)
    doc_embs = embedder.embed_documents(docs)
    indexer = FAISSIndexer(embedder.embedding_dim)
    indexer.add_documents(docs, doc_embs)
    rag = RAGFusion(embedder, indexer,
                    config.retrieval.fusion_weights,
                    config.retrieval.top_k)

    # Generation
    meta_gen = MetadataGenerator(
        config.generation.metadata_model,
        config.experiment.device,
        config.generation.metadata_max_length,
        config.generation.temperature,
    )
    grounded = GroundedGenerator(meta_gen, rag)

    all_records = []
    per_grade = config.data.num_synthetic_metadata // 5
    for grade in range(5):
        generated = grounded.generate_grounded(
            dr_grade=grade,
            num_records=per_grade,
            use_grounding=config.retrieval.rag_enabled,
        )
        for rec in generated:
            is_valid, _ = validator.validate(rec)
            if not is_valid and config.schema.repair_enabled:
                repaired, success = repairer.repair(rec)
                if success:
                    rec.update(repaired)
                rec["_repaired"] = success
            rec["_valid"] = is_valid or rec.get("_repaired", False)
            rec["_assigned_grade"] = grade
            all_records.append(rec)

    out = Path("outputs/results/synthetic_metadata_A2.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for rec in all_records:
            f.write(json.dumps(rec) + "\n")

    print(f"Saved {len(all_records)} records to {out}")
    print(f"Schema validity rate: {validator.validity_rate:.3f}")
    print(f"Repair success rate: {repairer.repair_success_rate:.3f}")

if __name__ == "__main__":
    main()