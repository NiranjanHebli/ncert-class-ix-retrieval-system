import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"

from typing import List, Dict, Any
from pathlib import Path
import logging
import chromadb
from chromadb.config import Settings
from rank_bm25 import BM25Okapi
import re
import numpy as np
from improved_chunking import ImprovedChunker, Chunk

PROJECT_ROOT = Path(__file__).parent.parent

# Suppress annoying ChromaDB telemetry errors
logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.CRITICAL)

class HybridRetriever:
    """
    Implements Hybrid Retrieval (BM25 + Dense) with Reciprocal Rank Fusion (RRF).
    """

    def __init__(self, chroma_collection_name: str, chroma_path: str = "data/chroma_db"):
        self.chroma_client = chromadb.PersistentClient(path=chroma_path)

        model_map = {
            "all_mpnet_base_v2": "sentence-transformers/all-mpnet-base-v2",
            "bge_small_en_v1.5": "BAAI/bge-small-en-v1.5"
        }

        embedding_function = None
        if chroma_collection_name in model_map:
            from chromadb.utils import embedding_functions
            embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=model_map[chroma_collection_name]
            )
            print(f"Using embedding model: {model_map[chroma_collection_name]}")

        try:
            self.collection = self.chroma_client.get_collection(
                name=chroma_collection_name,
                embedding_function=embedding_function
            )
        except Exception as e:
            print(f"Warning: Could not get collection '{chroma_collection_name}'. Error: {e}")
            print("Falling back to 'all_mpnet_base_v2' if available...")
            self.collection = self.chroma_client.get_collection(
                name="all_mpnet_base_v2",
                embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(
                    model_name="sentence-transformers/all-mpnet-base-v2"
                )
            )

        self.bm25 = None
        self.chunks = []

    def build_bm25(self, chunks: List[Chunk]):
        """Build BM25 index from chunks"""
        self.chunks = chunks
        tokenized_chunks = []
        for chunk in chunks:
            tokens = re.findall(r'\b\w+\b', chunk.text.lower())
            tokenized_chunks.append(tokens)

        self.bm25 = BM25Okapi(tokenized_chunks)

    def retrieve_hybrid(self, query: str, k: int = 5, rrf_k: int = 60) -> List[Dict[str, Any]]:
        """
        Retrieve chunks using Hybrid (BM25 + Dense) RRF fusion.
        """

        dense_res = self.collection.query(
            query_texts=[query],
            n_results=20
        )

        dense_ids = dense_res["ids"][0]
        dense_ranks = {id: rank for rank, id in enumerate(dense_ids)}

        query_tokens = re.findall(r'\b\w+\b', query.lower())
        bm25_scores = self.bm25.get_scores(query_tokens)
        bm25_indices = np.argsort(bm25_scores)[::-1][:20]

        bm25_ids = [self.chunks[idx].chunk_id for idx in bm25_indices]
        bm25_ranks = {id: rank for rank, id in enumerate(bm25_ids)}

        all_ids = set(dense_ids) | set(bm25_ids)
        rrf_scores = {}

        for chunk_id in all_ids:
            score = 0.0
            if chunk_id in dense_ranks:
                score += 1.0 / (rrf_k + dense_ranks[chunk_id])
            if chunk_id in bm25_ranks:
                score += 1.0 / (rrf_k + bm25_ranks[chunk_id])
            rrf_scores[chunk_id] = score

        sorted_ids = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:k]

        results = []

        chunk_map = {c.chunk_id: c for c in self.chunks}

        for chunk_id, rrf_score in sorted_ids:
            chunk = chunk_map.get(chunk_id)
            if not chunk:

                print(f"Warning: chunk_id {chunk_id} not found in current chunks map.")
                continue

            results.append({
                "text": chunk.text,
                "chunk_id": chunk.chunk_id,
                "chapter": chunk.chapter,
                "content_type": chunk.content_type,
                "rrf_score": rrf_score,
                "source": "hybrid"
            })

        return results

if __name__ == "__main__":

    print("HybridRetriever module loaded.")

