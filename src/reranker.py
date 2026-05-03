from sentence_transformers import CrossEncoder
from typing import List, Dict, Any

class LocalReranker:
    """
    Implements a local Cross-Encoder reranker as a fallback for production-grade RAG.
    Uses ms-marco-MiniLM-L-6-v2 for a good balance of speed and accuracy.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        print(f"Loading Cross-Encoder model: {model_name}...")
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, chunks: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Reranks the retrieved chunks based on the query.
        """
        if not chunks:
            return []

        pairs = [[query, chunk["text"]] for chunk in chunks]

        scores = self.model.predict(pairs)

        for i, score in enumerate(scores):
            chunks[i]["rerank_score"] = float(score)

        reranked_chunks = sorted(chunks, key=lambda x: x["rerank_score"], reverse=True)

        return reranked_chunks[:top_k]

if __name__ == "__main__":

    reranker = LocalReranker()
    test_query = "What is gravity?"
    test_chunks = [
        {"text": "Gravity is a force of attraction.", "id": "1"},
        {"text": "The moon orbits the earth.", "id": "2"},
        {"text": "Newton's second law is F=ma.", "id": "3"}
    ]
    results = reranker.rerank(test_query, test_chunks)
    for r in results:
        print(f"Score: {r['rerank_score']:.4f} | Text: {r['text']}")

