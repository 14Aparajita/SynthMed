import numpy as np
import json
import re
from typing import List, Dict, Tuple
from .embedder import DocumentEmbedder
from .indexer import FAISSIndexer
import logging

logger = logging.getLogger("synthmed.retrieval")


def _clinical_values_text(record_json_text: str) -> str:
    """
    Extract clinical values from a JSON record so grounding is measured on
    content, not on JSON syntax or field names.
    """
    try:
        rec = json.loads(record_json_text)
    except Exception:
        return record_json_text

    values = []
    for k in ["age", "sex", "dr_grade", "image_quality", "left_eye"]:
        if k in rec:
            values.append(str(rec[k]))
    anat = rec.get("anatomical_findings", {})
    if isinstance(anat, dict):
        for k, v in anat.items():
            values.append(f"{k} {v}")
    return " ".join(values)


class RAGFusion:
    """
    RAG Fusion for knowledge-grounded generation.
    Combines multiple retrieval strategies with reciprocal rank fusion.
    """

    def __init__(self, embedder: DocumentEmbedder, indexer: FAISSIndexer,
                 fusion_weights: List[float] = None, top_k: int = 5):
        self.embedder = embedder
        self.indexer = indexer
        self.fusion_weights = fusion_weights if fusion_weights is not None else [0.4, 0.3, 0.3]
        self.top_k = top_k
        self.grounding_scores: List[float] = []

    def retrieve(self, query: str, context: Dict[str, str] = None) -> List[Tuple[str, float, str]]:
        semantic = self._semantic_search(query)
        keyword = self._keyword_search(query)
        clinical = self._clinical_search(query, context)
        fused = self._reciprocal_rank_fusion([semantic, keyword, clinical])
        return fused[:self.top_k]

    def _semantic_search(self, query: str) -> List[Tuple[str, float]]:
        q_emb = self.embedder.embed_query(query)
        return self.indexer.search(q_emb, k=self.top_k * 2)

    def _keyword_search(self, query: str) -> List[Tuple[str, float]]:
        query_terms = set(query.lower().split())
        scores = []
        for doc in self.indexer.documents:
            doc_terms = set(doc.lower().split())
            union = len(query_terms | doc_terms)
            score = len(query_terms & doc_terms) / union if union else 0.0
            scores.append((doc, score))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:self.top_k * 2]

    def _clinical_search(self, query: str, context: Dict[str, str] = None) -> List[Tuple[str, float]]:
        expansions = {
            "dr": ["diabetic retinopathy", "retinal", "fundus"],
            "grade": ["severity", "level", "stage", "classification"],
            "microaneurysm": ["lesion", "dot", "hemorrhage"],
            "exudate": ["hard exudate", "soft exudate", "cotton wool"],
        }
        expanded = query
        for term, extra in expansions.items():
            if term.lower() in query.lower():
                expanded += " " + " ".join(extra)
        return self._semantic_search(expanded)

    def _reciprocal_rank_fusion(self, result_lists, k: int = 60):
        fused = {}
        for weight, results in zip(self.fusion_weights, result_lists):
            for rank, (doc, score) in enumerate(results, 1):
                if doc not in fused:
                    fused[doc] = {"score": 0.0, "strategies": []}
                fused[doc]["score"] += weight * (1.0 / (k + rank))
                fused[doc]["strategies"].append(score)
        sorted_docs = sorted(fused.items(), key=lambda x: x[1]["score"], reverse=True)
        return [(doc, s["score"], ", ".join(map(str, s["strategies"]))) for doc, s in sorted_docs]

    def compute_grounding_score(self, generated_text: str, retrieved_docs: List[str]) -> float:
        """
        Mean Jaccard-style overlap between clinical values of the generated
        record and the retrieved passages.
        """
        if not retrieved_docs:
            return 0.0

        # Extract only clinical values from the record; strip JSON syntax and keys
        cleaned = _clinical_values_text(generated_text)
        gen_tokens = set(re.findall(r"[a-z]+", cleaned.lower()))
        gen_tokens -= {"none", "true", "false"}  # drop non-informative tokens

        if not gen_tokens:
            return 0.0

        doc_scores = []
        for doc in retrieved_docs:
            doc_tokens = set(re.findall(r"[a-z]+", doc.lower()))
            if not doc_tokens:
                doc_scores.append(0.0)
                continue
            overlap = len(gen_tokens & doc_tokens)
            union = len(gen_tokens | doc_tokens)
            doc_scores.append(overlap / union if union else 0.0)

        score = float(np.mean(doc_scores))
        self.grounding_scores.append(score)
        return score

    @property
    def mean_grounding_score(self) -> float:
        if not self.grounding_scores:
            return 0.0
        return float(np.mean(self.grounding_scores))

    def reset_scores(self):
        self.grounding_scores = []