import os
import json
import numpy as np
from pathlib import Path
from vec_retrieval import VectorDatabase

# Prevent OpenMP runtime initialization error for FAISS
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def calculate_semantic_similarity(chunk_text, key_concepts, db):
    if not key_concepts:
        return 0.0
    
    concept_query = " ".join(key_concepts)
    chunk_embed = db.get_bert_embeddings([chunk_text])
    concept_embed = db.get_bert_embeddings([concept_query])
    
    chunk_vec = chunk_embed[0]
    concept_vec = concept_embed[0]
    
    chunk_norm = np.linalg.norm(chunk_vec)
    concept_norm = np.linalg.norm(concept_vec)
    
    if chunk_norm == 0 or concept_norm == 0:
        return 0.0
    
    similarity = np.dot(chunk_vec, concept_vec) / (chunk_norm * concept_norm)
    return float(similarity)


def calculate_concept_relevance_score(results, key_concepts, db):
    if not results or not key_concepts:
        return 0.0
    
    similarities = []
    for result in results:
        sim = calculate_semantic_similarity(result['text'], key_concepts, db)
        similarities.append(sim)
    
    return np.mean(similarities)


def calculate_concept_precision_at_k(results, key_concepts, db, k, threshold=0.6):
    if not results or k <= 0 or not key_concepts:
        return 0.0
    
    top_k = results[:k]
    relevant_count = 0
    
    for result in top_k:
        sim = calculate_semantic_similarity(result['text'], key_concepts, db)
        if sim >= threshold:
            relevant_count += 1
    
    return relevant_count / len(top_k)


def compare_retrieval_methods(bm25_results, faiss_results, key_concepts, db):
    comparison = {
        'bm25': {
            'concept_relevance_score': calculate_concept_relevance_score(bm25_results, key_concepts, db),
            'concept_precision_at_5': calculate_concept_precision_at_k(bm25_results, key_concepts, db, 5),
            'concept_precision_at_3': calculate_concept_precision_at_k(bm25_results, key_concepts, db, 3),
            'total_results': len(bm25_results)
        },
        'faiss': {
            'concept_relevance_score': calculate_concept_relevance_score(faiss_results, key_concepts, db),
            'concept_precision_at_5': calculate_concept_precision_at_k(faiss_results, key_concepts, db, 5),
            'concept_precision_at_3': calculate_concept_precision_at_k(faiss_results, key_concepts, db, 3),
            'total_results': len(faiss_results)
        }
    }
    
    def winner(metric):
        b = comparison['bm25'][metric]
        f = comparison['faiss'][metric]
        return 'bm25' if b > f else 'faiss' if b < f else 'tie'

    comparison['winner'] = {
        'concept_relevance_score': winner('concept_relevance_score'),
        'concept_precision_at_5': winner('concept_precision_at_5'),
        'concept_precision_at_3': winner('concept_precision_at_3'),
    }
    
    return comparison


def load_iesc102_questions_and_concepts(data_dir: Path = None):
    if data_dir is None:
        data_dir = Path('data')
    
    file_path = data_dir / 'iesc102_questions_and_concepts.json'
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data['questions_and_concepts']
    except FileNotFoundError:
        print(f"Warning: IESC102 questions and concepts file not found at {file_path}, using default questions")
        return None


def main(data_dir: Path = None, docs_dir: Path = None):

    project_root = Path(__file__).parent.parent
    
    if data_dir is None:
        data_dir = project_root / 'data'
    if docs_dir is None:
        docs_dir = project_root / 'extracted'
    
    questions_data = load_iesc102_questions_and_concepts(data_dir)
    
    if questions_data is None:
        questions_data = []
    else:
        questions_file = data_dir / 'iesc102_questions_and_concepts.json'
        try:
            with open(questions_file, 'r') as f:
                raw = json.load(f)
            doc_metadata = raw
        except FileNotFoundError:
            pass

    doc_file = docs_dir / 'iesc102.txt'
    with open(doc_file, 'r') as f:
        txt = f.read()
    
    print(f"Loaded document text with {len(txt)} characters")
    print()
    
    print("Initializing Vector Database...")
    db = VectorDatabase()
    
    print("Building chunk store from scratch...")
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
    
    print("\n" + "=" * 80)
    print("RETRIEVAL TESTING")
    print("=" * 80)
    
    tests = []
    for qc in questions_data:
        key_concepts = qc.get('key_concepts', [])
        if key_concepts:
            tests.append({
                "question": qc['question'],
                "question_id": qc.get('question_id', 0),
                "description": f"Question {qc.get('question_id', 0)}: {', '.join(key_concepts[:3])}...",
                "key_concepts": key_concepts
            })
    
    all_comparisons = []
    cumulative_bm25 = 0.0
    cumulative_faiss = 0.0
    
    for i, test in enumerate(tests, 1):
        print(f"\nTest {i}: {test['question']}")
        print(f"Question ID: {test['question_id']}")
        print(f"Description: {test['description']}")
        print("-" * 60)
        
        print("BM25 Results:")
        bm25_res = db.retrieve_bm25_with_scores(test['question'], k=3)
        for j, res in enumerate(bm25_res, 1):
            sim = calculate_semantic_similarity(res['text'], test['key_concepts'], db)
            rel = "RELEVANT" if sim >= 0.6 else "LESS RELEVANT"
            print(f"  {j}. [{rel}] BM25 Score: {res['bm25_score']:.3f}")
            print(f"     Type: {res['content_type']}")
            print(f"     Text: {res['text'][:100]}...")
        
        print()
        
        print("FAISS Results:")
        faiss_res = db.retrieve_faiss(test['question'], k=3)
        for j, res in enumerate(faiss_res, 1):
            sim = calculate_semantic_similarity(res['text'], test['key_concepts'], db)
            rel = "RELEVANT" if sim >= 0.6 else "LESS RELEVANT"
            print(f"  {j}. [{rel}] FAISS Similarity: {sim:.3f}")
            print(f"     Type: {res['content_type']}")
            print(f"     Text: {res['text'][:100]}...")
        
        key_concepts = test.get('key_concepts', [])
        comparison = compare_retrieval_methods(bm25_res, faiss_res, key_concepts, db)
        all_comparisons.append(comparison)
        cumulative_bm25 += comparison['bm25']['concept_relevance_score']
        cumulative_faiss += comparison['faiss']['concept_relevance_score']
        
        print(f"\nCOMPARISON ANALYSIS:")
        print(f"  Concept Relevance Score: BM25 {comparison['bm25']['concept_relevance_score']:.3f} vs FAISS {comparison['faiss']['concept_relevance_score']:.3f}")
        print(f"  Concept Precision@5: BM25 {comparison['bm25']['concept_precision_at_5']:.3f} vs FAISS {comparison['faiss']['concept_precision_at_5']:.3f}")
        print(f"  Concept Precision@3: BM25 {comparison['bm25']['concept_precision_at_3']:.3f} vs FAISS {comparison['faiss']['concept_precision_at_3']:.3f}")
        print(f"  Winner: ConceptRelevance={comparison['winner']['concept_relevance_score'].upper()}, ConceptP@5={comparison['winner']['concept_precision_at_5'].upper()}, ConceptP@3={comparison['winner']['concept_precision_at_3'].upper()}")
        
        print("\n" + "=" * 60)
    
    print("\n" + "=" * 80)
    print("OVERALL PERFORMANCE ANALYSIS")
    print("=" * 80)
    
    avg_bm25_relevance = sum(c['bm25']['concept_relevance_score'] for c in all_comparisons) / len(all_comparisons)
    avg_faiss_relevance = sum(c['faiss']['concept_relevance_score'] for c in all_comparisons) / len(all_comparisons)
    avg_bm25_p5 = sum(c['bm25']['concept_precision_at_5'] for c in all_comparisons) / len(all_comparisons)
    avg_faiss_p5 = sum(c['faiss']['concept_precision_at_5'] for c in all_comparisons) / len(all_comparisons)
    avg_bm25_p3 = sum(c['bm25']['concept_precision_at_3'] for c in all_comparisons) / len(all_comparisons)
    avg_faiss_p3 = sum(c['faiss']['concept_precision_at_3'] for c in all_comparisons) / len(all_comparisons)
    
    metrics = ['concept_relevance_score', 'concept_precision_at_5', 'concept_precision_at_3']
    bm25_wins = sum(1 for c in all_comparisons if any(c['winner'][m] == 'bm25' for m in metrics))
    faiss_wins = sum(1 for c in all_comparisons if any(c['winner'][m] == 'faiss' for m in metrics))
    
    print(f"Average Concept Relevance Score: BM25 {avg_bm25_relevance:.3f} vs FAISS {avg_faiss_relevance:.3f}")
    print(f"Average Concept Precision@5: BM25 {avg_bm25_p5:.3f} vs FAISS {avg_faiss_p5:.3f}")
    print(f"Average Concept Precision@3: BM25 {avg_bm25_p3:.3f} vs FAISS {avg_faiss_p3:.3f}")
    print(f"Overall Winner: {'BM25' if bm25_wins > faiss_wins else 'FAISS' if faiss_wins > bm25_wins else 'TIE'}")
    print(f"Win Count: BM25 {bm25_wins} vs FAISS {faiss_wins}")
    print(f"\nCumulative Similarity Score: BM25 {cumulative_bm25:.3f} vs FAISS {cumulative_faiss:.3f}")
    print(f"Cumulative Winner: {'BM25' if cumulative_bm25 > cumulative_faiss else 'FAISS' if cumulative_faiss > cumulative_bm25 else 'TIE'}")
    db.flush_vector_db()

if __name__ == "__main__":
    main()