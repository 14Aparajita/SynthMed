import numpy as np
from typing import List, Dict, Tuple
from .embedder import DocumentEmbedder
from .indexer import FAISSIndexer
import logging

logger = logging.getLogger("synthmed.retrieval")

class RAGFusion:
    """
    RAG Fusion for knowledge-grounded generation.
    Combines multiple retrieval strategies with reciprocal rank fusion.
    """
    
    def __init__(
        self,
        embedder: DocumentEmbedder,
        indexer: FAISSIndexer,
        fusion_weights: List[float] = [0.4, 0.3, 0.3],
        top_k: int = 5
    ):
        self.embedder = embedder
        self.indexer = indexer
        self.fusion_weights = fusion_weights
        self.top_k = top_k
        self.grounding_scores = []
    
    def retrieve(
        self,
        query: str,
        context: Dict[str, str] = None
    ) -> List[Tuple[str, float, str]]:
        """
        Retrieve and fuse documents using multiple strategies.
        Returns list of (document, score, strategy) tuples.
        """
        # Strategy 1: Semantic search
        semantic_results = self._semantic_search(query)
        
        # Strategy 2: Keyword-based search
        keyword_results = self._keyword_search(query)
        
        # Strategy 3: Clinical concept search
        clinical_results = self._clinical_search(query, context)
        
        # Fusion
        fused = self._reciprocal_rank_fusion([
            semantic_results,
            keyword_results,
            clinical_results
        ])
        
        return fused[:self.top_k]
    
    def _semantic_search(self, query: str) -> List[Tuple[str, float]]:
        """Dense retrieval using embeddings."""
        query_embedding = self.embedder.embed_query(query)
        return self.indexer.search(query_embedding, k=self.top_k * 2)
    
    def _keyword_search(self, query: str) -> List[Tuple[str, float]]:
        """Sparse retrieval using keyword overlap."""
        from collections import Counter
        
        query_terms = set(query.lower().split())
        scores = []
        
        for doc in self.indexer.documents:
            doc_terms = set(doc.lower().split())
            overlap = len(query_terms & doc_terms)
            # Jaccard similarity
            union = len(query_terms | doc_terms)
            score = overlap / union if union > 0 else 0.0
            scores.append((doc, score))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:self.top_k * 2]
    
    def _clinical_search(
        self, query: str, context: Dict[str, str] = None
    ) -> List[Tuple[str, float]]:
        """Search using clinical concept expansion."""
        clinical_terms = {
            "dr": ["diabetic retinopathy", "retinal", "fundus"],
            "grade": ["severity", "level", "stage", "classification"],
            "microaneurysm": ["lesion", "dot", "hemorrhage"],
            "exudate": ["hard exudate", "soft exudate", "cotton wool"],
        }
        
        expanded_query = query
        for term, expansions in clinical_terms.items():
            if term.lower() in query.lower():
                expanded_query += " " + " ".join(expansions)
        
        # Use semantic search on expanded query
        return self._semantic_search(expanded_query)
    
    def _reciprocal_rank_fusion(
        self,
        result_lists: List[List[Tuple[str, float]]],
        k: int = 60
    ) -> List[Tuple[str, float, str]]:
        """
        Reciprocal Rank Fusion algorithm.
        Combines multiple ranked lists.
        """
        fused_scores = {}
        
        for weight, results in zip(self.fusion_weights, result_lists):
            for rank, (doc, score) in enumerate(results, 1):
                if doc not in fused_scores:
                    fused_scores[doc] = {"score": 0, "strategies": []}
                
                rrf_score = weight * (1.0 / (k + rank))
                fused_scores[doc]["score"] += rrf_score
                fused_scores[doc]["strategies"].append(score)
        
        # Sort by fused score
        sorted_docs = sorted(
            fused_scores.items(),
            key=lambda x: x[1]["score"],
            reverse=True
        )
        
        return [
            (doc, scores["score"], ", ".join(map(str, scores["strategies"])))
            for doc, scores in sorted_docs
        ]
    
    def compute_grounding_score(
        self,
        generated_text: str,
        retrieved_docs: List[str]
    ) -> float:
        """
        Compute grounding score: how well generated text is grounded in retrieved docs.
        Simple overlap-based metric.
        """
        generated_terms = set(generated_text.lower().split())
        
        if not retrieved_docs:
            return 0.0
        
        doc_scores = []
        for doc in retrieved_docs:
            doc_terms = set(doc.lower().split())
            overlap = len(generated_terms & doc_terms)
            total = len(generated_terms)
            doc_scores.append(overlap / total if total > 0 else 0.0)
        
        grounding_score = np.mean(doc_scores)
        self.grounding_scores.append(grounding_score)
        
        return grounding_score
    
    @property
    def mean_grounding_score(self) -> float:
        """Average grounding score across all generations."""
        if not self.grounding_scores:
            return 0.0
        return np.mean(self.grounding_scores)
    
    def reset_scores(self):
        """Reset grounding score history."""
        self.grounding_scores = []