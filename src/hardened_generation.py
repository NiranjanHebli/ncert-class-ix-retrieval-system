import os
from pathlib import Path
from typing import List, Dict, Any
from groq import Groq
from dotenv import load_dotenv
from hybrid_retrieval import HybridRetriever
from improved_chunking import ImprovedChunker

PROJECT_ROOT = Path(__file__).parent.parent
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

HARDENED_PROMPT = """
You are a strict study assistant for PariShiksha.
Your goal is to provide accurate answers based ONLY on the provided context.

RULES:
1. Use ONLY the provided context to answer the question.
2. If the answer is not in the context, or if the context is insufficient, you MUST respond with exactly:
   "This question is outside the provided NCERT content."
3. Do NOT use any external knowledge, even if you know the answer.
4. For every claim you make, you MUST cite the source using the format [chunk_id] at the end of the sentence.
5. If the question is partially answerable, only answer the part supported by the context and state that the rest is unavailable.

Context:
{context}

Question: {question}

Answer (with [chunk_id] citations):
"""

class HardenedGenerator:
    def __init__(self, collection_name: str = "all_mpnet_base_v2"):
        self.client = Groq(api_key=GROQ_API_KEY)
        self.retriever = HybridRetriever(collection_name)
        self.chunker = ImprovedChunker()

        print("Initializing Hybrid Retriever with BM25...")
        chunks = self.chunker.chunk_directory(str(PROJECT_ROOT / "extracted"))
        self.retriever.build_bm25(chunks)

    def ask(self, question: str, k: int = 5) -> Dict[str, Any]:
        """
        Retrieves context using Hybrid search and generates a hardened, cited answer.
        """

        retrieved_chunks = self.retriever.retrieve_hybrid(question, k=k)

        context_blocks = []
        for c in retrieved_chunks:
            block = f"Source: {c['chapter']} | ID: {c['chunk_id']}\nContent: {c['text']}"
            context_blocks.append(block)

        full_context = "\n\n---\n\n".join(context_blocks)

        prompt = HARDENED_PROMPT.format(context=full_context, question=question)

        try:
            completion = self.client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=1024
            )

            return {
                "answer": completion.choices[0].message.content,
                "sources": retrieved_chunks,
                "prompt_used": prompt
            }
        except Exception as e:
            return {"error": str(e)}

if __name__ == "__main__":

    gen = HardenedGenerator()
    q = "What is the universal law of gravitation?"
    print(f"\n[USER]: {q}")
    res = gen.ask(q)
    print(f"[ASSISTANT]: {res['answer']}")

