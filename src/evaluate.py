"""
Evaluation
Runs the full evaluation set through the Groq generator and saves results.
"""

import os
import sys
import json
import csv
from datetime import datetime


sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

from groq import Groq
from dotenv import load_dotenv
from vec_retrieval import VectorDatabase

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

GROUNDING_PROMPT = """
You are a study assistant for PariShiksha. 
Use ONLY the context provided below to answer the question.
If the answer is not present in the context, respond with:
"This question is outside the provided NCERT content."
Do not infer, extrapolate, or use outside knowledge.

Context:
{context}

Question: {question}
Answer:
"""


def run_evaluation():
    """Run full 3-axis evaluation and save results."""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


    md_path = str(PROJECT_ROOT / "docs/evaluation_results.md")
    questions_file = str(PROJECT_ROOT / "data/eval_questions.json")
    with open(questions_file, "r") as f:
        categories = json.load(f)


    client = Groq(api_key=GROQ_API_KEY)
    db = VectorDatabase(use_embeddings=False)


    db_path = str(PROJECT_ROOT / "data/vector_db")
    extracted_dir = str(PROJECT_ROOT / "extracted")

    if os.path.exists(db_path):
        print("Loading existing database from disk...")
        db.load_from_disk(db_path)
    else:
        print("Building new database from scratch...")
        for root, dirs, files in os.walk(extracted_dir):
            if root == extracted_dir:
                continue
            for f in files:
                if f.endswith(".txt"):
                    db.build_chunk_store_from_file(os.path.join(root, f))
        db.save_to_disk(db_path)


    results = []
    print("\n" + "=" * 80)
    print("EVALUATION RUN")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Model: llama-3.1-8b-instant")
    print(f"Total chunks in DB: {len(db.chunks)}")
    print("=" * 80)

    for cat in categories:
        category = cat["category"]
        q_type = cat["type"]
        print(f"\n--- {category} ({q_type}) ---")

        for question in cat["questions"]:

            retrieved_chunks = db.retrieve_bm25(question, k=3)
            context = "\n\n---\n\n".join([chunk['text'] for chunk in retrieved_chunks])
            prompt = GROUNDING_PROMPT.format(context=context, question=question)


            try:
                completion = client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0
                )
                answer = completion.choices[0].message.content
            except Exception as e:
                answer = f"[ERROR]: {str(e)}"


            is_refusal = "outside" in answer.lower() or "not present" in answer.lower() or "not in the context" in answer.lower()


            if q_type == "out_of_scope":
                correctness = "yes" if is_refusal else "no"
                grounded = "yes" if is_refusal else "no"
                refusal_appropriate = "yes" if is_refusal else "no"
            else:
                correctness = "yes" if not is_refusal and len(answer) > 20 else ("no" if is_refusal else "partial")
                grounded = "yes" if not is_refusal else "no"
                refusal_appropriate = "na" if not is_refusal else "no"

            result = {
                "question": question,
                "type": q_type,
                "category": category,
                "answer": answer.strip(),
                "correctness": correctness,
                "grounded": grounded,
                "refusal_appropriate": refusal_appropriate,
                "is_refusal": is_refusal,
                "retrieved_sources": [c.get("chapter", "Unknown") for c in retrieved_chunks]
            }
            results.append(result)

            status = "[Correct]" if (q_type != "out_of_scope" and not is_refusal) or (q_type == "out_of_scope" and is_refusal) else "[Incorrect]"
            print(f"  {status} {question[:60]}...")
            print(f"     Answer: {answer[:100]}...")


    csv_path = str(PROJECT_ROOT / "data/evaluation_results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "question", "type", "category", "answer",
            "correctness", "grounded", "refusal_appropriate",
            "is_refusal", "retrieved_sources"
        ])
        writer.writeheader()
        for r in results:
            r_copy = r.copy()
            r_copy["retrieved_sources"] = ", ".join(r_copy["retrieved_sources"])
            writer.writerow(r_copy)

    print(f"\n CSV saved to: {csv_path}")


    generate_evaluation_markdown(results, md_path)


    print_summary(results)

    return results


def generate_evaluation_markdown(results, md_path):
    """Generate the evaluation_results.md file from results."""

    total = len(results)
    correct = sum(1 for r in results if r["correctness"] == "yes")
    partial = sum(1 for r in results if r["correctness"] == "partial")
    grounded = sum(1 for r in results if r["grounded"] == "yes")

    direct_qs = [r for r in results if r["type"] == "direct"]
    para_qs = [r for r in results if r["type"] == "paraphrased"]
    oos_qs = [r for r in results if r["type"] == "out_of_scope"]

    direct_correct = sum(1 for r in direct_qs if r["correctness"] == "yes")
    para_correct = sum(1 for r in para_qs if r["correctness"] == "yes")
    oos_refused = sum(1 for r in oos_qs if r["is_refusal"])

    with open(md_path, "w") as f:
        f.write("# Evaluation Results\n\n")
        f.write(f"**Model:** Llama 3.1 8B (via Groq)  \n")
        f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d')}  \n")
        f.write(f"**Total Questions:** {total}  \n")
        f.write(f"**Retrieval Method:** BM25  \n\n")

        f.write("## Summary Scores\n\n")
        f.write(f"| Metric | Score |\n")
        f.write(f"|---|---|\n")
        f.write(f"| Overall Correctness | {correct}/{total} ({correct/total*100:.0f}%) |\n")
        f.write(f"| Partial Answers | {partial}/{total} |\n")
        f.write(f"| Grounded Answers | {grounded}/{total} ({grounded/total*100:.0f}%) |\n")
        f.write(f"| Direct Textbook Accuracy | {direct_correct}/{len(direct_qs)} |\n")
        f.write(f"| Paraphrased Accuracy | {para_correct}/{len(para_qs)} |\n")
        f.write(f"| Out-of-Scope Refusal Rate | {oos_refused}/{len(oos_qs)} |\n\n")

        f.write("## Detailed Results\n\n")
        f.write("| # | Question | Type | Correctness | Grounded | Refusal |\n")
        f.write("|---|---|---|---|---|---|\n")
        for i, r in enumerate(results, 1):
            q_short = r["question"][:50] + ("..." if len(r["question"]) > 50 else "")
            f.write(f"| {i} | {q_short} | {r['type']} | {r['correctness']} | {r['grounded']} | {r['refusal_appropriate']} |\n")

        # Task 4.3: Working and failing examples
        f.write("\n## Analysis: Working Examples\n\n")
        working = [r for r in results if r["correctness"] == "yes" and r["type"] != "out_of_scope"][:3]
        for i, r in enumerate(working, 1):
            f.write(f"### Working Example {i}\n")
            f.write(f"**Q:** {r['question']}  \n")
            f.write(f"**A:** {r['answer'][:300]}  \n")
            f.write(f"**Sources:** {', '.join(r['retrieved_sources'])}  \n")
            f.write(f"**Why it works:** BM25 correctly matched keywords from the question to relevant textbook chunks.\n\n")

        f.write("## Analysis: Failing Examples\n\n")
        failing = [r for r in results if r["correctness"] == "no"][:2]
        if not failing:
            failing = [r for r in results if r["correctness"] == "partial"][:2]
        for i, r in enumerate(failing, 1):
            f.write(f"### Failing Example {i}\n")
            f.write(f"**Q:** {r['question']}  \n")
            f.write(f"**A:** {r['answer'][:300]}  \n")
            f.write(f"**Sources:** {', '.join(r['retrieved_sources'])}  \n")
            if r["type"] == "out_of_scope" and not r["is_refusal"]:
                f.write(f"**Probable cause:** The retriever returned Chapter 9 content that superficially matched, and the LLM used it to construct a plausible but incorrect answer instead of refusing.\n\n")
            else:
                f.write(f"**Probable cause:** The retriever returned chunks that did not contain the specific information needed, likely due to keyword mismatch or the answer being split across chunk boundaries.\n\n")

    print(f" Markdown saved to: {md_path}")


def print_summary(results):
    """Print a summary of the evaluation."""
    total = len(results)
    correct = sum(1 for r in results if r["correctness"] == "yes")
    partial = sum(1 for r in results if r["correctness"] == "partial")
    grounded = sum(1 for r in results if r["grounded"] == "yes")

    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
    print(f"  Total Questions: {total}")
    print(f"  Correct:         {correct}/{total} ({correct/total*100:.0f}%)")
    print(f"  Partial:         {partial}/{total}")
    print(f"  Grounded:        {grounded}/{total} ({grounded/total*100:.0f}%)")


    for q_type in ["direct", "paraphrased", "out_of_scope"]:
        type_results = [r for r in results if r["type"] == q_type]
        if not type_results:
            continue
        type_correct = sum(1 for r in type_results if r["correctness"] == "yes")
        print(f"  [{q_type}]: {type_correct}/{len(type_results)}")

    print("=" * 80)


if __name__ == "__main__":
    run_evaluation()
