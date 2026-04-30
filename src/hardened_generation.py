import os
from typing import List, Dict, Any
from google import genai
from google.genai import types
from dotenv import load_dotenv
from hybrid_retrieval import HybridRetriever
from improved_chunking import ImprovedChunker

load_dotenv()
API_KEY = os.getenv("API_KEY")

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
    def __init__(self, collection_name: str = "gemini_embedding_001"):
        self.client = genai.Client(api_key=API_KEY)
        self.retriever = HybridRetriever(collection_name)
        self.chunker = ImprovedChunker()
        
        # Pre-build BM25 for the retriever
        print("Initializing Hybrid Retriever with BM25...")
        chunks = self.chunker.chunk_directory("extracted")
        self.retriever.build_bm25(chunks)
        
    def ask(self, question: str, k: int = 5) -> Dict[str, Any]:
        """
        Retrieves context using Hybrid search and generates a hardened, cited answer.
        """
        # 1. Retrieve hybrid context
        retrieved_chunks = self.retriever.retrieve_hybrid(question, k=k)
        
        # 2. Format context for prompt
        context_blocks = []
        for c in retrieved_chunks:
            block = f"Source: {c['chapter']} | ID: {c['chunk_id']}\nContent: {c['text']}"
            context_blocks.append(block)
        
        full_context = "\n\n---\n\n".join(context_blocks)
        
        # 3. Generate content
        prompt = HARDENED_PROMPT.format(context=full_context, question=question)
        
        try:
            response = self.client.models.generate_content(
                model="gemini-2.0-flash",
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.0, # Strictness
                    max_output_tokens=1024
                )
            )
            
            return {
                "answer": response.text,
                "sources": retrieved_chunks,
                "prompt_used": prompt
            }
        except Exception as e:
            return {"error": str(e)}

if __name__ == "__main__":
    # Test session
    gen = HardenedGenerator()
    q = "What is the universal law of gravitation?"
    print(f"\n[USER]: {q}")
    res = gen.ask(q)
    print(f"[ASSISTANT]: {res['answer']}")
