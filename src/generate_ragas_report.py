import os
import ast
import json
import glob
import argparse
import warnings
import pandas as pd
from datasets import Dataset
from dotenv import load_dotenv

warnings.filterwarnings("ignore", category=DeprecationWarning)

from openai import OpenAI
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from ragas.llms import llm_factory
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

# ── CLI: optional --sample N to evaluate only N questions ─────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--sample", type=int, default=None,
                    help="Evaluate only the first N rows (useful when rate-limited)")
args = parser.parse_args()

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── LLM: Using OpenAI-compatible factory for Groq ─────────────────────────────
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("GROQ_API_KEY not found in environment variables.")

groq_client = OpenAI(
    api_key=groq_api_key,
    base_url="https://api.groq.com/openai/v1"
)

# Create the evaluator LLM using the factory
ragas_llm = llm_factory(
    model="llama-3.1-8b-instant",
    provider="openai",
    client=groq_client
)

# Use Langchain wrapper for embeddings
lc_embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)
ragas_embeddings = LangchainEmbeddingsWrapper(lc_embeddings)

# ── Load the LATEST eval_scored CSV, fall back to evaluation_results.csv ──────
scored_dir = os.path.join(PROJECT_ROOT, "data", "eval_scored")
scored_files = sorted(glob.glob(os.path.join(scored_dir, "eval_scored_*.csv")))
if scored_files:
    latest_csv = scored_files[-1]
    print(f"Loading evaluation results from: {latest_csv}")
else:
    latest_csv = os.path.join(PROJECT_ROOT, "data", "evaluation_results.csv")
    print(f"No eval_scored files found. Falling back to: {latest_csv}")
df = pd.read_csv(latest_csv)

# Drop duplicates if any
df = df.drop_duplicates(subset=["question"])

if args.sample:
    df = df.head(args.sample)
    print(f"Sample mode: evaluating first {len(df)} unique rows.")

# ── Load chunk map for context lookup ─────────────────────────────────────────
with open(os.path.join(PROJECT_ROOT, "data", "wk10_chunks.json"), "r") as f:
    chunks_list = json.load(f)
chunk_map = {c["chunk_id"]: c["text"] for c in chunks_list}

# ── Ground-truth map from eval_questions.json ──────────────────────────────────
with open(os.path.join(PROJECT_ROOT, "data", "eval_questions.json"), "r") as f:
    categories = json.load(f)

ground_truth_map = {}
for cat in categories:
    for q in cat["questions"]:
        if cat["type"] == "out_of_scope":
            ground_truth_map[q] = "This question is outside the provided NCERT content."
        else:
            ground_truth_map[q] = None  # filled from answer column below

# ── Build RAGAS dataset ────────────────────────────────────────────────────────
data = {"question": [], "answer": [], "contexts": [], "ground_truth": []}

for _, row in df.iterrows():
    q = row["question"]
    a = row["answer"]

    raw_sources = row.get("sources", "[]")
    try:
        source_ids = ast.literal_eval(str(raw_sources))
    except (ValueError, SyntaxError):
        source_ids = []

    ctx = [chunk_map[sid] for sid in source_ids if sid in chunk_map]
    if not ctx:
        ctx = ["No context available."]

    gt = ground_truth_map.get(q) or a

    data["question"].append(q)
    data["answer"].append(a)
    data["contexts"].append(ctx)
    data["ground_truth"].append(gt)

dataset = Dataset.from_dict(data)
dataset = dataset.rename_column("question", "user_input")
dataset = dataset.rename_column("answer", "response")
dataset = dataset.rename_column("contexts", "retrieved_contexts")
dataset = dataset.rename_column("ground_truth", "reference")

print(f"Running RAGAS evaluation on {len(dataset)} questions...")

# Assign the LLM and Embeddings to the metrics
faithfulness.llm = ragas_llm
answer_relevancy.llm = ragas_llm
answer_relevancy.embeddings = ragas_embeddings
context_precision.llm = ragas_llm
context_recall.llm = ragas_llm

result = evaluate(
    dataset=dataset,
    metrics=[
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    ],
    llm=ragas_llm,
    embeddings=ragas_embeddings,
)

result_df = result.to_pandas()
report_path = os.path.join(PROJECT_ROOT, "data", "ragas_report.csv")
result_df.to_csv(report_path, index=False)

print(f"\nRAGAS report saved to: {report_path}")
print(result_df[["user_input", "faithfulness", "answer_relevancy",
                  "context_precision", "context_recall"]].to_string(index=False))