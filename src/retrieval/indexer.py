import faiss
import numpy as np
import pickle
from pathlib import Path
from typing import List, Tuple
import logging

logger = logging.getLogger("synthmed.retrieval")

class FAISSIndexer:
    """FAISS-based vector index for document retrieval."""
    
    def __init__(self, dimension: int, index_type: str = "Flat"):
        self.dimension = dimension
        self.index_type = index_type
        self.index = None
        self.documents = []
        self._build_index()
    
    def _build_index(self):
        """Initialize FAISS index."""
        if self.index_type == "Flat":
            self.index = faiss.IndexFlatL2(self.dimension)
        elif self.index_type == "IVF":
            quantizer = faiss.IndexFlatL2(self.dimension)
            nlist = min(100, self.dimension // 10)
            self.index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist)
        else:
            self.index = faiss.IndexFlatL2(self.dimension)
        
        logger.info(f"Built FAISS index: {self.index_type}, dim={self.dimension}")
    
    def add_documents(self, documents: List[str], embeddings: np.ndarray):
        """Add documents and their embeddings to the index."""
        if embeddings.shape[1] != self.dimension:
            raise ValueError(f"Embedding dimension {embeddings.shape[1]} != {self.dimension}")
        
        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings)
        
        if not self.index.is_trained and self.index_type == "IVF":
            self.index.train(embeddings)
        
        self.index.add(embeddings.astype(np.float32))
        self.documents.extend(documents)
        
        logger.info(f"Added {len(documents)} documents. Total: {self.index.ntotal}")
    
    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Tuple[str, float]]:
        """Search for k nearest documents."""
        query = query_embedding.reshape(1, -1).astype(np.float32)
        faiss.normalize_L2(query)
        
        distances, indices = self.index.search(query, k)
        
        results = []
        for i, dist in zip(indices[0], distances[0]):
            if i < len(self.documents) and i >= 0:
                similarity = 1.0 - (dist / 2.0)  # Convert L2 to cosine similarity
                results.append((self.documents[i], similarity))
        
        return results
    
    def save(self, path: str):
        """Save index and documents."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        faiss.write_index(self.index, path)
        with open(path + ".docs", 'wb') as f:
            pickle.dump(self.documents, f)
        
        logger.info(f"Saved index to {path}")
    
    def load(self, path: str):
        """Load index and documents."""
        self.index = faiss.read_index(path)
        with open(path + ".docs", 'rb') as f:
            self.documents = pickle.load(f)
        
        logger.info(f"Loaded index from {path}: {self.index.ntotal} documents")