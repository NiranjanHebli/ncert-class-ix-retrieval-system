#!/usr/bin/env python3
"""Vector Database Retrieval Demo."""

from vector_database import VectorDatabase


def calculate_relevance_score(results, expected_type):
    """Calculate relevance score for retrieval results."""
    if not results:
        return 0.0
    
    relevant_count = sum(1 for r in results if r['content_type'] == expected_type)
    return relevant_count / len(results)


def calculate_precision_at_k(results, expected_type, k):
    """Calculate precision@k for top k results."""
    if not results or k <= 0:
        return 0.0
    
    top_k = results[:k]
    relevant_count = sum(1 for r in top_k if r['content_type'] == expected_type)
    return relevant_count / min(k, len(top_k))


def compare_retrieval_methods(bm25_results, faiss_results, expected_type):
    """Compare BM25 vs FAISS retrieval quality."""
    comparison = {
        'bm25': {
            'relevance_score': calculate_relevance_score(bm25_results, expected_type),
            'precision_at_1': calculate_precision_at_k(bm25_results, expected_type, 1),
            'precision_at_3': calculate_precision_at_k(bm25_results, expected_type, 3),
            'total_results': len(bm25_results)
        },
        'faiss': {
            'relevance_score': calculate_relevance_score(faiss_results, expected_type),
            'precision_at_1': calculate_precision_at_k(faiss_results, expected_type, 1),
            'precision_at_3': calculate_precision_at_k(faiss_results, expected_type, 3),
            'total_results': len(faiss_results)
        }
    }
    
    # Determine winner for each metric
    comparison['winner'] = {
        'relevance_score': 'bm25' if comparison['bm25']['relevance_score'] > comparison['faiss']['relevance_score'] else 'faiss',
        'precision_at_1': 'bm25' if comparison['bm25']['precision_at_1'] > comparison['faiss']['precision_at_1'] else 'faiss',
        'precision_at_3': 'bm25' if comparison['bm25']['precision_at_3'] > comparison['faiss']['precision_at_3'] else 'faiss'
    }
    
    return comparison


def main():
    """Run vector database retrieval demo."""
    print("=" * 80)
    print("VECTOR DATABASE RETRIEVAL DEMO")
    print("=" * 80)
    
    with open('sample_science_text.txt', 'r') as f:
        txt = f.read()
    
    print(f"Loaded text with {len(txt)} characters")
    print()
    
    print("Initializing Vector Database...")
    db = VectorDatabase()
    
    print("Building chunk store with FAISS and BM25...")
    chunks = db.build_chunk_store(txt)
    
    print("\n" + "=" * 80)
    print("CHUNK ANALYSIS")
    print("=" * 80)
    
    types = {}
    chaps = {}
    
    for chunk in chunks:
        ct = chunk['content_type']
        ch = chunk['chapter']
        
        types[ct] = types.get(ct, 0) + 1
        chaps[ch] = chaps.get(ch, 0) + 1
    
    print("Content Type Distribution:")
    for ct, cnt in types.items():
        print(f"  {ct}: {cnt} chunks")
    
    print("\nChapter Distribution:")
    for ch, cnt in chaps.items():
        print(f"  {ch}: {cnt} chunks")
    
    print("\n" + "=" * 80)
    print("RETRIEVAL TESTING")
    print("=" * 80)
    
    tests = [
        {
            "question": "What are the different types of chemical reactions?",
            "expected_type": "chemical_reactions",
            "description": "Should find content about combination, decomposition, displacement reactions"
        },
        {
            "question": "How do acids and bases differ in properties?",
            "expected_type": "acids_bases", 
            "description": "Should find content about pH, taste, litmus tests"
        },
        {
            "question": "What makes metals good conductors of electricity?",
            "expected_type": "metals_nonmetals",
            "description": "Should find content about metallic properties, conductivity"
        }
    ]
    
    all_comparisons = []
    
    for i, test in enumerate(tests, 1):
        print(f"\nTest {i}: {test['question']}")
        print(f"Expected: {test['expected_type']}")
        print(f"Description: {test['description']}")
        print("-" * 60)
        
        print("BM25 Results:")
        bm25_res = db.retrieve_bm25(test['question'], k=3)
        for j, res in enumerate(bm25_res, 1):
            rel = "RELEVANT" if res['content_type'] == test['expected_type'] else "LESS RELEVANT"
            print(f"  {j}. [{rel}] Score: {res['bm25_score']:.3f}")
            print(f"     Type: {res['content_type']}")
            print(f"     Text: {res['text'][:100]}...")
        
        print()
        
        print("FAISS Results:")
        faiss_res = db.retrieve_faiss(test['question'], k=3)
        for j, res in enumerate(faiss_res, 1):
            rel = "RELEVANT" if res['content_type'] == test['expected_type'] else "LESS RELEVANT"
            print(f"  {j}. [{rel}] Similarity: {res['similarity_score']:.3f}")
            print(f"     Type: {res['content_type']}")
            print(f"     Text: {res['text'][:100]}...")
        
        comparison = compare_retrieval_methods(bm25_res, faiss_res, test['expected_type'])
        all_comparisons.append(comparison)
        
        print(f"\nCOMPARISON ANALYSIS:")
        print(f"  Relevance Score: BM25 {comparison['bm25']['relevance_score']:.2f} vs FAISS {comparison['faiss']['relevance_score']:.2f}")
        print(f"  Precision@1: BM25 {comparison['bm25']['precision_at_1']:.2f} vs FAISS {comparison['faiss']['precision_at_1']:.2f}")
        print(f"  Precision@3: BM25 {comparison['bm25']['precision_at_3']:.2f} vs FAISS {comparison['faiss']['precision_at_3']:.2f}")
        print(f"  Winner: Relevance={comparison['winner']['relevance_score'].upper()}, P@1={comparison['winner']['precision_at_1'].upper()}, P@3={comparison['winner']['precision_at_3'].upper()}")
        
        print("\n" + "=" * 60)
    
    print("\n" + "=" * 80)
    print("OVERALL PERFORMANCE ANALYSIS")
    print("=" * 80)
    
    avg_bm25_relevance = sum(c['bm25']['relevance_score'] for c in all_comparisons) / len(all_comparisons)
    avg_faiss_relevance = sum(c['faiss']['relevance_score'] for c in all_comparisons) / len(all_comparisons)
    avg_bm25_p1 = sum(c['bm25']['precision_at_1'] for c in all_comparisons) / len(all_comparisons)
    avg_faiss_p1 = sum(c['faiss']['precision_at_1'] for c in all_comparisons) / len(all_comparisons)
    avg_bm25_p3 = sum(c['bm25']['precision_at_3'] for c in all_comparisons) / len(all_comparisons)
    avg_faiss_p3 = sum(c['faiss']['precision_at_3'] for c in all_comparisons) / len(all_comparisons)
    
    bm25_wins = sum(1 for c in all_comparisons if c['winner']['relevance_score'] == 'bm25')
    faiss_wins = sum(1 for c in all_comparisons if c['winner']['relevance_score'] == 'faiss')
    
    print(f"Average Relevance Score: BM25 {avg_bm25_relevance:.2f} vs FAISS {avg_faiss_relevance:.2f}")
    print(f"Average Precision@1: BM25 {avg_bm25_p1:.2f} vs FAISS {avg_faiss_p1:.2f}")
    print(f"Average Precision@3: BM25 {avg_bm25_p3:.2f} vs FAISS {avg_faiss_p3:.2f}")
    print(f"Overall Winner: {'BM25' if bm25_wins > faiss_wins else 'FAISS' if faiss_wins > bm25_wins else 'TIE'}")
    print(f"Win Count: BM25 {bm25_wins} vs FAISS {faiss_wins}")
    
    print("\n" + "=" * 80)
    print("SYSTEM SUMMARY")
    print("=" * 80)
    print(f"Total chunks processed: {len(chunks)}")
    print(f"FAISS index dimension: {db.embeds.shape[1]}")
    print(f"BM25 corpus size: {len(db.chunks)}")
    print(f"Embedding model: BERT-BASE-UNCASED")
    print(f"Chunk sizes: BERT (500 tokens, 50 overlap), T5 (300 tokens, 50 overlap)")
    print("\nVector database successfully implemented with:")
    print("  - Text chunking with BERT and T5 tokenizers")
    print("  - FAISS vector similarity search")
    print("  - BM25 keyword-based retrieval")
    print("  - Metadata-enriched chunk storage")
    print("  - Content classification system")
    print("  - Retrieval quality comparison metrics")

if __name__ == "__main__":
    main()
