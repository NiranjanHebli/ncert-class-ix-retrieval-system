import os
import json
import csv
from datetime import datetime
from hardened_generation import HardenedGenerator

def run_v2_evaluation():
    """Runs the Stage 2 evaluation using the HardenedGenerator."""
    print("\n--- Stage 2: Hardened RAG Evaluation ---")
    
    # Initialize generator
    gen = HardenedGenerator()
    
    # Load questions
    with open("data/eval_questions.json", "r") as f:
        categories = json.load(f)
    
    results = []
    
    for cat in categories:
        cat_name = cat["category"]
        q_type = cat["type"]
        print(f"\nEvaluating Category: {cat_name}...")
        
        for q in cat["questions"]:
            print(f"  Question: {q[:60]}...")
            res = gen.ask(q)
            
            if "error" in res:
                print(f"    [ERROR]: {res['error']}")
                continue
                
            answer = res["answer"]
            
            # Heuristic axes (to be refined in Stage 4/5)
            # Grounding: Does it have [chunk_id] citations?
            grounded = "yes" if "[" in answer and "]" in answer and any(c.chunk_id in answer for c in res["sources"]) else "no"
            
            # Refusal: Did it use the strict phrase for OOS?
            is_refusal = "outside the provided NCERT content" in answer.lower()
            
            # Correctness: Manual/Heuristic check
            # For OOS questions, a refusal is correct.
            if q_type == "out_of_scope":
                correctness = "yes" if is_refusal else "no"
            else:
                correctness = "yes" if not is_refusal and len(answer) > 30 else ("no" if is_refusal else "partial")
            
            results.append({
                "question": q,
                "type": q_type,
                "category": cat_name,
                "answer": answer,
                "correctness": correctness,
                "grounded": grounded,
                "refusal_appropriate": "yes" if (q_type == "out_of_scope" and is_refusal) or (q_type != "out_of_scope" and not is_refusal) else "no",
                "sources": [s["chunk_id"] for s in res["sources"]]
            })

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = f"data/eval_v2_scored_{timestamp}.csv"
    
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["question", "type", "category", "answer", "correctness", "grounded", "refusal_appropriate", "sources"])
        writer.writeheader()
        writer.writerows(results)
    
    print(f"\nEvaluation complete! Results saved to {csv_path}")
    
    # Simple summary
    total = len(results)
    correct = sum(1 for r in results if r["correctness"] == "yes")
    grounded_count = sum(1 for r in results if r["grounded"] == "yes")
    print(f"Overall Accuracy: {correct/total*100:.1f}%")
    print(f"Grounding Rate: {grounded_count/total*100:.1f}%")

if __name__ == "__main__":
    run_v2_evaluation()
