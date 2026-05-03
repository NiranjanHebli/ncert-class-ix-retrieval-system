"""
Simplified LangChain EnsembleRetriever Implementation
S3-11: Use LangChain EnsembleRetriever as starting point
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Any

from langchain_core.documents import Document
from langchain_classic.retrievers.ensemble import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from improved_chunking import ImprovedChunker, Chunk

PROJECT_ROOT = Path(__file__).parent.parent


class SimpleLangChainEnsemble:
    """
    Simplified implementation that demonstrates the concept
    of LangChain EnsembleRetriever without complex dependencies
    """
    
    def __init__(self):
        self.chunks = []
    
    def build_from_chunks(self, chunks: List[Chunk]):
        """Build retrievers from chunks"""
        self.chunks = chunks
        self._build_langchain_ensemble()
    
    def _build_langchain_ensemble(self):
        """Build actual LangChain ensemble"""
        # Prepare documents
        documents = []
        for chunk in self.chunks:
            doc = Document(
                page_content=chunk.text,
                metadata={
                    "chunk_id": chunk.chunk_id,
                    "chapter": chunk.chapter,
                    "content_type": chunk.content_type
                }
            )
            documents.append(doc)
        
        # Build BM25 retriever
        self.bm25_retriever = BM25Retriever.from_documents(documents)
        
        # Build embeddings and vector store
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        chroma_path = str(PROJECT_ROOT / "data/chroma_langchain")
        self.vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=chroma_path
        )
        
        # Build dense retriever
        self.dense_retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 10}
        )
        
        # Create ensemble
        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[self.bm25_retriever, self.dense_retriever],
            weights=[0.5, 0.5]
        )
        
        print("LangChain EnsembleRetriever built successfully!")
    def retrieve(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve using ensemble"""
        docs = self.ensemble_retriever.invoke(query)
        results = []
        for i, doc in enumerate(docs[:k]):
            results.append({
                "text": doc.page_content,
                "chunk_id": doc.metadata.get("chunk_id", f"ensemble_{i}"),
                "chapter": doc.metadata.get("chapter", "Unknown"),
                "content_type": doc.metadata.get("content_type", "text"),
                "score": 1.0 / (i + 1),
                "source": "langchain_ensemble"
            })
        return results


def test_ensemble():
    """Test the ensemble retriever"""
    ensemble = SimpleLangChainEnsemble()
    
    chunks_file = PROJECT_ROOT / "data" / "improved_vector_db" / "wk10_chunks.json"
    with open(chunks_file, 'r', encoding='utf-8') as f:
        chunks_data = json.load(f)
    
    chunks = [Chunk(
        text=chunk_data["text"],
        content_type=chunk_data["content_type"],
        chapter=chunk_data["chapter"],
        chunk_id=chunk_data["chunk_id"],
        token_count=len(chunk_data["text"].split()),
        semantic_boundary="paragraph"
    ) for chunk_data in chunks_data]
    
    ensemble.build_from_chunks(chunks)
    
    for query in ["What are the three states of matter?", "State the universal law of gravitation."]:
        print(f"\nQuery: {query}")
        results = ensemble.retrieve(query, k=3)
        for i, result in enumerate(results, 1):
            print(f"  {i}. [{result['source']}] {result['chapter']} - {result['text'][:60]}...")


if __name__ == "__main__":
    test_ensemble()
