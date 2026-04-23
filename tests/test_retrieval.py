import os
import sys

# Add src to sys.path to allow importing vec_retrieval
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from vec_retrieval import VectorDatabase

def test_retrieval():
    db = VectorDatabase(use_embeddings=False)
    # Correct root calculation for tests/test_retrieval.py (2 levels up)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    extracted_dir = os.path.join(project_root, "extracted")
    
    print(f"Scanning directory: {extracted_dir}")
    files_found = 0
    for root, dirs, files in os.walk(extracted_dir):
        for filename in files:
            if filename.endswith(".txt"):
                db.build_chunk_store_from_file(os.path.join(root, filename))
                files_found += 1
    
    print(f"\nTotal files loaded: {files_found}")
    print(f"Total chunks in DB: {len(db.chunks)}")
    
    query = "first law of motion"
    print(f"\nQuery: {query}")
    results = db.retrieve_bm25(query, k=5)
    
    print(f"Found {len(results)} results:")
    for i, res in enumerate(results):
        print(f"\n--- Result {i+1} ---")
        print(f"Source: {res.get('chapter', 'Unknown')}")
        print(f"Text Snippet: {res['text'][:300]}...")

if __name__ == "__main__":
    test_retrieval()
