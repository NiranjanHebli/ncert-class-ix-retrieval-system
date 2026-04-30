import os
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging

from langchain_core.documents import Document
# FIX: Corrected package path from langchain_classic
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
# FIX: Chroma has moved to its own integration package
from langchain_chroma import Chroma 
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder
import cohere

from dotenv import load_dotenv
load_dotenv()  

# ---------------------------------------------------------------------------
# Project layout
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Minimal Chunk dataclass
# ---------------------------------------------------------------------------
try:
    from improved_chunking import Chunk  # type: ignore
except ImportError:
    from dataclasses import dataclass

    @dataclass
    class Chunk:
        text: str
        content_type: str
        chapter: str
        chunk_id: str
        token_count: int
        semantic_boundary: str = "paragraph"

# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------
class EnhancedLangChainEnsemble:
    """
    Hybrid BM25 + dense retriever with Cohere reranking and a 
    cross-encoder fallback.
    """

    COHERE_RERANK_MODEL = "rerank-english-v3.0"

    def __init__(self, cohere_api_key: Optional[str] = None):
        self.chunks: List[Chunk] = []
        self._is_built: bool = False

        self.cohere_api_key = cohere_api_key or os.getenv("COHERE_API_KEY")
        self.cross_encoder: Optional[CrossEncoder] = None
        self.cohere_client: Optional[cohere.Client] = None

        self.latency_metrics: Dict[str, List[float]] = {
            "base_retrieval": [],
            "reranking": [],
            "total": [],
        }

        self._init_cross_encoder()

        if self.cohere_api_key:
            try:
                # FIX: Updated for Cohere v5+ SDK
                self.cohere_client = cohere.Client(api_key=self.cohere_api_key)
                logger.info("Cohere client initialised successfully")
            except Exception as exc:
                logger.warning("Failed to initialise Cohere client: %s", exc)
        else:
            logger.warning("No Cohere API key – cross-encoder fallback will be used")

    def _init_cross_encoder(self) -> None:
        try:
            self.cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
            logger.info("Cross-encoder fallback model loaded")
        except Exception as exc:
            logger.error("Failed to load cross-encoder: %s", exc)

    def build_from_chunks(self, chunks: List[Chunk]) -> None:
        self.chunks = chunks
        self._build_langchain_ensemble()
        self._is_built = True

    def _build_langchain_ensemble(self) -> None:
        documents: List[Document] = [
            Document(
                page_content=chunk.text,
                metadata={
                    "chunk_id": chunk.chunk_id,
                    "chapter": chunk.chapter,
                    "content_type": chunk.content_type,
                },
            )
            for chunk in self.chunks
        ]

        # --- BM25 ---
        self.bm25_retriever = BM25Retriever.from_documents(documents)
        self.bm25_retriever.k = 20

        # --- Dense (Chroma) ---
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        chroma_path = str(PROJECT_ROOT / "data" / "chroma_langchain")

        # FIX: Updated Chroma initialization for langchain-chroma
        self.vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=chroma_path,
            collection_name="ensemble_chunks",
        )

        self.dense_retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 20},
        )

        # --- Ensemble ---
        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[self.bm25_retriever, self.dense_retriever],
            weights=[0.5, 0.5],
        )

        self.multi_query_retriever = self.ensemble_retriever
        logger.info("Enhanced LangChain EnsembleRetriever built successfully")

    def _cohere_rerank(
        self, query: str, documents: List[Document], top_k: int = 5
    ) -> tuple[List[Document], List[float]]:
        if not self.cohere_client:
            raise RuntimeError("Cohere client not initialised")

        docs_text = [doc.page_content for doc in documents]

        # FIX: Updated for latest Cohere SDK syntax
        response = self.cohere_client.rerank(
            model=self.COHERE_RERANK_MODEL,
            query=query,
            documents=docs_text,
            top_n=top_k,
        )

        reranked_docs: List[Document] = []
        scores: List[float] = []
        for result in response.results:
            reranked_docs.append(documents[result.index])
            scores.append(float(result.relevance_score))

        return reranked_docs, scores

    def _cross_encoder_rerank(
        self, query: str, documents: List[Document], top_k: int = 5
    ) -> tuple[List[Document], List[float]]:
        seen: set[str] = set()
        unique_docs: List[Document] = []
        for doc in documents:
            if doc.page_content not in seen:
                seen.add(doc.page_content)
                unique_docs.append(doc)

        if not self.cross_encoder:
            logger.warning("Cross-encoder unavailable – returning original order")
            scores = [1.0 / (i + 1) for i in range(len(unique_docs[:top_k]))]
            return unique_docs[:top_k], scores

        pairs = [(query, doc.page_content) for doc in unique_docs]
        raw_scores: List[float] = self.cross_encoder.predict(pairs).tolist()

        doc_score_pairs = sorted(
            zip(unique_docs, raw_scores), key=lambda x: x[1], reverse=True
        )
        top_docs = [d for d, _ in doc_score_pairs[:top_k]]
        top_scores = [s for _, s in doc_score_pairs[:top_k]]
        return top_docs, top_scores

    def _multi_query_retrieve(self, query: str, k: int = 20) -> List[Document]:
        try:
            raw_docs = self.multi_query_retriever.invoke(query)
        except Exception as exc:
            logger.warning("Multi-query retrieval failed, using ensemble: %s", exc)
            raw_docs = self.ensemble_retriever.invoke(query)

        seen: set[str] = set()
        unique_docs: List[Document] = []
        for doc in raw_docs:
            if doc.page_content not in seen:
                seen.add(doc.page_content)
                unique_docs.append(doc)

        return unique_docs[:k]

    def retrieve(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        if not self._is_built:
            raise RuntimeError("Retrievers are not built yet. Call build_from_chunks() first.")

        start_time = time.time()

        t0 = time.time()
        candidates = self._multi_query_retrieve(query, k=20)
        retrieval_time = time.time() - t0
        self.latency_metrics["base_retrieval"].append(retrieval_time)

        reranker_used: str
        t1 = time.time()
        try:
            if not self.cohere_client:
                raise RuntimeError("Cohere client not available")
            reranked_docs, scores = self._cohere_rerank(query, candidates, top_k=k)
            reranker_used = "cohere"
        except Exception as exc:
            logger.warning("Cohere reranking failed, using cross-encoder: %s", exc)
            reranked_docs, scores = self._cross_encoder_rerank(query, candidates, top_k=k)
            reranker_used = "cross_encoder"

        rerank_time = time.time() - t1
        self.latency_metrics["reranking"].append(rerank_time)
        total_time = time.time() - start_time
        self.latency_metrics["total"].append(total_time)

        results: List[Dict[str, Any]] = []
        for doc, score in zip(reranked_docs, scores):
            results.append({
                "text": doc.page_content,
                "chunk_id": doc.metadata.get("chunk_id", "unknown"),
                "chapter": doc.metadata.get("chapter", "Unknown"),
                "content_type": doc.metadata.get("content_type", "text"),
                "score": score,
                "reranker": reranker_used,
                "source": "enhanced_langchain_ensemble",
                "total_latency_ms": round(total_time * 1000, 2),
            })

        return results

    def get_latency_report(self) -> Dict[str, Dict[str, float]]:
        report: Dict[str, Dict[str, float]] = {}
        for name, times in self.latency_metrics.items():
            if times:
                ms = [t * 1000 for t in times]
                report[name] = {
                    "avg_ms": round(sum(ms) / len(ms), 2),
                    "count": len(ms),
                }
        return report

def test_enhanced_ensemble() -> None:
    """Quick integration test – requires wk10_chunks.json to exist."""
    chunks_file = PROJECT_ROOT / "data" / "improved_vector_db" / "wk10_chunks.json"

    # FIX: friendly error instead of a raw FileNotFoundError crash
    if not chunks_file.exists():
        raise FileNotFoundError(
            f"Test data not found at {chunks_file}. "
            "Generate it first by running improved_chunking.py."
        )

    with chunks_file.open(encoding="utf-8") as fh:
        chunks_data: List[Dict[str, Any]] = json.load(fh)

    chunks = [
        Chunk(
            text=item["text"],
            content_type=item["content_type"],
            chapter=item["chapter"],
            chunk_id=item["chunk_id"],
            token_count=len(item["text"].split()),
            semantic_boundary="paragraph",
        )
        for item in chunks_data
    ]

    ensemble = EnhancedLangChainEnsemble()
    ensemble.build_from_chunks(chunks)

    test_queries = [
        "What are the three states of matter?",
        "State the universal law of gravitation.",
        "Explain the process of photosynthesis.",
    ]

    print("\n=== Enhanced Ensemble Retrieval Test ===")
    for query in test_queries:
        print(f"\nQuery: {query}")
        results = ensemble.retrieve(query, k=3)
        for i, result in enumerate(results, 1):
            print(
                f"  {i}. [{result['reranker']}] score={result['score']:.4f}  "
                f"{result['chapter']} — {result['text'][:70]}…"
            )
            print(f"     Latency: {result['total_latency_ms']:.1f} ms")

    print("\n=== Latency Report ===")
    for metric, stats in ensemble.get_latency_report().items():
        print(
            f"{metric}: {stats['avg_ms']:.1f} ms avg  "
            f"(n={stats['count']})"
        )


if __name__ == "__main__":
    test_enhanced_ensemble()