"""Test retrieval and RAG fusion."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from src.retrieval import DocumentEmbedder, FAISSIndexer, RAGFusion

def test_embedder():
    embedder = DocumentEmbedder("sentence-transformers/all-MiniLM-L6-v2")
    
    docs = ["Test document one", "Test document two"]
    embeddings = embedder.embed_documents(docs)
    
    assert embeddings.shape == (2, 384)  # MiniLM-L6-v2 dimension
    assert embeddings.dtype == np.float32

def test_indexer():
    embedder = DocumentEmbedder("sentence-transformers/all-MiniLM-L6-v2")
    indexer = FAISSIndexer(embedder.embedding_dim)
    
    docs = ["Document A", "Document B", "Document C"]
    embeddings = embedder.embed_documents(docs)
    indexer.add_documents(docs, embeddings)
    
    query_embedding = embedder.embed_query("Document A")
    results = indexer.search(query_embedding, k=2)
    
    assert len(results) == 2
    assert results[0][0] == "Document A"  # Closest match

def test_rag_fusion():
    embedder = DocumentEmbedder("sentence-transformers/all-MiniLM-L6-v2")
    indexer = FAISSIndexer(embedder.embedding_dim)
    
    docs = [
        "Diabetic retinopathy causes microaneurysms in the retina",
        "Regular exercise helps control blood sugar levels",
        "Retinal examination detects early signs of DR",
        "Hypertension is common in diabetic patients",
        "Microaneurysms are the first visible sign of DR",
    ]
    embeddings = embedder.embed_documents(docs)
    indexer.add_documents(docs, embeddings)
    
    rag = RAGFusion(embedder, indexer)
    results = rag.retrieve("microaneurysms in diabetic retinopathy")
    
    assert len(results) > 0
    # Should retrieve relevant documents
    assert any("microaneurysms" in r[0].lower() for r in results)