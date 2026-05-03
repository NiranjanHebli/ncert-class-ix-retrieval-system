import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import json
import csv
from pathlib import Path
from datetime import datetime
from hardened_generation import HardenedGenerator

PROJECT_ROOT = Path(__file__).parent.parent

def run_v2_evaluation():
    """Runs the evaluation using the HardenedGenerator."""
    print("\n--- Hardened RAG Evaluation ---")

    gen = HardenedGenerator()

    with open("data/eval_questions.json", "r") as f:
        categories = json.load(f)

    results = []

    for cat in categories:
        cat_name = cat["category"]
        q_type = cat["type"]
        print(f"\nEvaluating Category: {cat_name}")

        for q in cat["questions"]:
            print(f"\n  Question: {q}")
            res = gen.ask(q)

            if "error" in res:
                print(f"    [ERROR]: {res['error']}")
                continue

            answer = res["answer"]

            is_refusal = "outside the provided ncert content" in answer.lower()

            if is_refusal:
                grounded = "yes"
                grounded_reason = "(Checked via presence of proper refusal string; correct refusals are grounded behavior)"
            else:
                grounded = "yes" if any(c["chunk_id"] in answer for c in res["sources"]) else "no"
                grounded_reason = "(Checked via presence of valid [chunk_id] citation)"

            if q_type == "out_of_scope":
                correctness = "yes" if is_refusal else "no"
                correct_reason = "(Checked via presence of correct refusal string for out-of-scope query)"
            else:
                correctness = "yes" if not is_refusal and len(answer) > 30 else ("no" if is_refusal else "partial")
                correct_reason = "(Checked via heuristic: valid non-refusal answer > 30 chars)"

            print(f"  Answer: {answer}")
            print(f"  Correctness: {correctness.upper()} {correct_reason}")
            print(f"  Grounded: {grounded.upper()} {grounded_reason}")

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

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = "data/eval_scored"
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, f"eval_scored_{timestamp}.csv")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["question", "type", "category", "answer", "correctness", "grounded", "refusal_appropriate", "sources"])
        writer.writeheader()
        writer.writerows(results)

    print(f"\nEvaluation complete! Results saved to {csv_path}")

    total = len(results)
    correct = sum(1 for r in results if r["correctness"] == "yes")
    grounded_count = sum(1 for r in results if r["grounded"] == "yes")
    print(f"Overall Accuracy: {correct/total*100:.1f}%")
    print(f"Grounding Rate: {grounded_count/total*100:.1f}%")

if __name__ == "__main__":
    run_v2_evaluation()

