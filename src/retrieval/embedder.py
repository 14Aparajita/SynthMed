import numpy as np
from typing import List, Union
from sentence_transformers import SentenceTransformer
import logging

logger = logging.getLogger("synthmed.retrieval")

class DocumentEmbedder:
    """Embed documents and queries using sentence transformers."""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        logger.info(f"Loading embedder model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
    
    def embed_documents(self, documents: List[str]) -> np.ndarray:
        """Embed a list of documents."""
        embeddings = self.model.encode(
            documents,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        return embeddings
    
    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query."""
        embedding = self.model.encode([query], convert_to_numpy=True)
        return embedding[0]
    
    def embed_queries(self, queries: List[str]) -> np.ndarray:
        """Embed multiple queries."""
        return self.model.encode(queries, convert_to_numpy=True)