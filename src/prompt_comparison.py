import os
import sys
import json
from pathlib import Path
from datetime import datetime
from groq import Groq
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
import re
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# S3-02: Permissive prompt
PERMISSIVE_PROMPT = """
Answer the question using the context.

Context:
{context}

Question: {question}
Answer:
"""

# S3-04: Strict prompt with refusal phrase
STRICT_PROMPT = """
You are a study assistant for PariShiksha.
Answer only from the context.
If the answer is not in context, reply exactly: 
"I don't have that in my study materials."

Context:
{context}

Question: {question}
Answer:
"""

# S3-05: Strict prompt with citations
STRICT_PROMPT_CITATIONS = """
You are a study assistant for PariShiksha.
Answer only from the context.
If the answer is not in context, reply exactly: 
"I don't have that in my study materials."
For every factual claim, include citation in format [Source: chunk_id].

Context:
{context}

Question: {question}
Answer:
"""


class PromptComparator:
    def __init__(self, model="llama-3.1-8b-instant"):
        if not GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY not found in .env file.")
        
        self.client = Groq(api_key=GROQ_API_KEY)
        self.model = model
        self.chunks = []
        
        # Load chunks from wk10_chunks.json
        self._load_chunks()
        
        # Initialize BM25
        self._init_bm25()
    
    def _load_chunks(self):
        """Load chunks from wk10_chunks.json"""
        chunks_file = PROJECT_ROOT / "data" / "improved_vector_db" / "wk10_chunks.json"
        
        if not chunks_file.exists():
            raise FileNotFoundError(f"Chunks file not found: {chunks_file}")
        
        print(f"Loading chunks from {chunks_file}")
        with open(chunks_file, 'r', encoding='utf-8') as f:
            self.chunks = json.load(f)
        
        print(f"Loaded {len(self.chunks)} chunks")
    
    def _init_bm25(self):
        """Initialize BM25 retrieval"""
        tokenized_chunks = []
        for chunk in self.chunks:
            tokens = re.findall(r'\b\w+\b', chunk['text'].lower())
            tokenized_chunks.append(tokens)
        
        self.bm25 = BM25Okapi(tokenized_chunks)
        print("BM25 index built successfully")
    
    def _retrieve_bm25(self, query: str, k: int = 3):
        """Retrieve using BM25"""
        query_tokens = re.findall(r'\b\w+\b', query.lower())
        scores = self.bm25.get_scores(query_tokens)
        top_indices = np.argsort(scores)[::-1][:k]
        
        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # Only include matches with score > 0
                chunk = self.chunks[idx]
                results.append(chunk)
        
        return results
    
    def ask(self, question, prompt_type="strict", k=3):
        """
        S3-08: Update ask(question) to return dict: {answer, sources, chunk_ids}
        """
        # Retrieve context
        retrieved_chunks = self._retrieve_bm25(question, k=k)
        context = "\n\n---\n\n".join([chunk['text'] for chunk in retrieved_chunks])
        
        # Select prompt
        if prompt_type == "permissive":
            prompt = PERMISSIVE_PROMPT.format(context=context, question=question)
        elif prompt_type == "strict":
            prompt = STRICT_PROMPT.format(context=context, question=question)
        elif prompt_type == "strict_citations":
            prompt = STRICT_PROMPT_CITATIONS.format(context=context, question=question)
        else:
            raise ValueError(f"Unknown prompt type: {prompt_type}")
        
        # S3-09: Temperature MUST be set to 0 for all evaluation runs
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0
            )
            answer = completion.choices[0].message.content
        except Exception as e:
            answer = f"[ERROR]: {str(e)}"
        
        return {
            "answer": answer.strip(),
            "sources": [c.get("chapter", "Unknown") for c in retrieved_chunks],
            "chunk_ids": [c.get("chunk_id", "Unknown") for c in retrieved_chunks],
            "prompt_used": prompt,
            "retrieved_chunks": retrieved_chunks
        }
    
    def run_comparison(self, questions):
        """
        Run comparison on provided questions
        """
        results = []
        
        for i, question in enumerate(questions, 1):
            print(f"\n--- Question {i} ---")
            print(f"Q: {question}")
            
            # S3-03: Run permissive prompt
            print("Running permissive prompt...")
            permissive_result = self.ask(question, "permissive")
            print(f"Permissive: {permissive_result['answer'][:100]}...")
            
            # S3-06: Re-run with strict prompt
            print("Running strict prompt...")
            strict_result = self.ask(question, "strict_citations")
            print(f"Strict: {strict_result['answer'][:100]}...")
            
            result = {
                "question": question,
                "permissive": permissive_result,
                "strict": strict_result
            }
            results.append(result)
        
        return results
    
    def save_results(self, results, output_path):
        """
        S3-07: Save both permissive and strict responses verbatim in prompt_diff.md
        """
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("# Prompt Comparison Results\n\n")
            f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  \n")
            f.write(f"**Model:** {self.model}  \n")
            f.write(f"**Temperature:** 0.0  \n\n")
            
            for i, result in enumerate(results, 1):
                f.write(f"## Question {i}\n\n")
                f.write(f"**Question:** {result['question']}\n\n")
                
                f.write("### Permissive Prompt Response\n\n")
                f.write("```\n")
                f.write(result['permissive']['answer'])
                f.write("\n```\n\n")
                
                f.write("### Strict Prompt Response\n\n")
                f.write("```\n")
                f.write(result['strict']['answer'])
                f.write("\n```\n\n")
                
                f.write("---\n\n")
        
        print(f"Results saved to: {output_path}")


def main():
    """Main function to run S3 tasks"""
    comparator = PromptComparator()
    
    # Test questions for S3-03 (including 1 out-of-scope)
    test_questions = [
        "What are the three states of matter?",  # in-scope
        "State the universal law of gravitation.",  # in-scope  
        "Who is the current Prime Minister of India?"  # out-of-scope
    ]
    
    print("Running prompt comparison...")
    results = comparator.run_comparison(test_questions)
    
    # S3-07: Save results
    output_path = str(PROJECT_ROOT / "prompt_diff.md")
    comparator.save_results(results, output_path)
    
    print("\n=== Summary ===")
    for i, result in enumerate(results, 1):
        q = result['question']
        perm = result['permissive']['answer']
        strict = result['strict']['answer']
        
        # Check for hallucinations/refusals
        perm_hallucination = "outside" not in perm.lower() and "don't have" not in perm.lower()
        strict_refusal = "don't have that in my study materials" in strict.lower()
        
        print(f"Q{i}: {q[:50]}...")
        print(f"  Permissive hallucination: {perm_hallucination}")
        print(f"  Strict refusal: {strict_refusal}")
        print()


if __name__ == "__main__":
    main()
