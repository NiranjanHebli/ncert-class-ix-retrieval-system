import os
import time
import json
import csv
import numpy as np
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any
from google import genai
from dotenv import load_dotenv
from improved_chunking import ImprovedChunker, Chunk

# Load environment variables
load_dotenv()
API_KEY = os.getenv("API_KEY")

class GeminiEmbeddingFunction(chromadb.EmbeddingFunction):
    def __init__(self, api_key: str, model_name: str = "models/gemini-embedding-001"):
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name

    def __call__(self, input: chromadb.Documents) -> chromadb.Embeddings:
        embeddings = []
        for text in input:
            try:
                if not text.strip():
                    # Handle empty text
                    embeddings.append([0.0] * 3072) 
                    continue
                res = self.client.models.embed_content(
                    model=self.model_name,
                    contents=text
                )
                embeddings.append(res.embeddings[0].values)
            except Exception as e:
                print(f"Error embedding text: {text[:50]}... | Error: {e}")
                embeddings.append([0.0] * 3072)
        return embeddings

class LocalEmbeddingFunction(chromadb.EmbeddingFunction):
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        self.model = SentenceTransformer(model_name)

    def __call__(self, input: chromadb.Documents) -> chromadb.Embeddings:
        return self.model.encode(input).tolist()

class DBBenchmark:
    def __init__(self, chunks: List[Chunk]):
        self.chunks = chunks
        self.chroma_client = chromadb.PersistentClient(path="data/chroma_db")
        
        # Models
        self.local_ef_1 = LocalEmbeddingFunction("BAAI/bge-small-en-v1.5")
        self.local_ef_2 = LocalEmbeddingFunction("sentence-transformers/all-mpnet-base-v2")
        
        self.models = {
            "bge_small_en_v1.5": self.local_ef_1,
            "all_mpnet_base_v2": self.local_ef_2
        }
        
        self.collections = {}
        
    def setup_collections(self):
        """Create collections for each model"""
        for name, ef in self.models.items():
            try:
                self.chroma_client.delete_collection(name)
            except:
                pass
            
            collection = self.chroma_client.create_collection(
                name=name,
                embedding_function=ef
            )
            self.collections[name] = collection
            
            # Index chunks
            ids = [c.chunk_id for c in self.chunks]
            documents = [c.text for c in self.chunks]
            metadatas = [{
                "content_type": c.content_type,
                "chapter": c.chapter,
                "token_count": c.token_count,
                "semantic_boundary": c.semantic_boundary
            } for c in self.chunks]
            
            print(f"Indexing {len(ids)} chunks for {name}...")
            # Batch indexing to avoid API limits/timeouts
            batch_size = 50
            for i in range(0, len(ids), batch_size):
                print(f"  Batch {i//batch_size + 1}/{(len(ids)-1)//batch_size + 1}")
                collection.add(
                    ids=ids[i:i+batch_size],
                    documents=documents[i:i+batch_size],
                    metadatas=metadatas[i:i+batch_size]
                )

    def run_benchmark(self, queries: List[str]):
        results = []
        
        for model_name, collection in self.collections.items():
            print(f"\nBenchmarking {model_name}...")
            
            latencies = []
            
            for idx, q in enumerate(queries):
                start_time = time.perf_counter()
                res = collection.query(
                    query_texts=[q],
                    n_results=5
                )
                end_time = time.perf_counter()
                latency = (end_time - start_time) * 1000 # ms
                latencies.append(latency)
                
                # Judging recall@5
                retrieved_context = "\n---\n".join(res["documents"][0])
                
                # Recall@5 (Placeholder for manual judgement per PDF)
                recall_hit = 0.0 # To be verified manually
                
                # Log snippet for manual review
                if idx == 0: # Only for first query to see if it works
                    print(f"  [Sample Query]: {q}")
                    print(f"  [Top Result]: {res['documents'][0][0][:100]}...")
                
                results.append({
                    "model": model_name,
                    "query": q,
                    "latency_ms": latency,
                    "recall_hit": recall_hit
                })
                
            # Print stats for model
            model_latencies = [r["latency_ms"] for r in results if r["model"] == model_name]
            model_recall = [r["recall_hit"] for r in results if r["model"] == model_name]
            
            print(f"  Avg Latency: {np.mean(model_latencies):.2f}ms")
            print(f"  p50 Latency: {np.percentile(model_latencies, 50):.2f}ms")
            print(f"  p95 Latency: {np.percentile(model_latencies, 95):.2f}ms")
            print(f"  Recall@5: {np.mean(model_recall)*100:.1f}%")
            
        return results

    def save_results(self, results: List[Dict[str, Any]]):
        csv_path = "data/db_benchmark.csv"
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=["model", "query", "latency_ms", "recall_hit"])
            writer.writeheader()
            writer.writerows(results)
        print(f"\nResults saved to {csv_path}")

def main():
    # Load chunks
    chunker = ImprovedChunker()
    extracted_dir = "extracted"
    print(f"Loading and chunking files from {extracted_dir}...")
    chunks = chunker.chunk_directory(extracted_dir)
    print(f"Total chunks created: {len(chunks)}")
    
    # Sample for faster benchmark (Stage 2 proof of concept)
    chunks = chunks[:200]
    print(f"Sampled to {len(chunks)} chunks for benchmark.")
    
    # Load queries
    with open("data/eval_questions.json", "r") as f:
        categories = json.load(f)
    
    queries = []
    for cat in categories:
        queries.extend(cat["questions"])
    
    # Run benchmark
    benchmark = DBBenchmark(chunks)
    benchmark.setup_collections()
    results = benchmark.run_benchmark(queries)
    benchmark.save_results(results)

if __name__ == "__main__":
    main()
