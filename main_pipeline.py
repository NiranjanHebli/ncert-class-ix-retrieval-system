"""
PariShiksha: NCERT Science QA Retrieval System - Main Pipeline
This script runs the end-to-end pipeline: extraction, tokenizer evaluation, and retrieval demo.
"""

import os
import subprocess
import sys

def run_command(command):
    print(f"\nExecuting: {' '.join(command)}")
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {e}")
        sys.exit(1)

def main():
    print("=" * 60)
    print("PARISHIKSHA END-TO-END PIPELINE")
    print("=" * 60)

    # 1. Corpus Extraction & Chunking
    print("\n[STEP 1] Extracting Corpus & Chunking...")
    run_command(["python3", "src/scripts/extract.py"])

    # 2. Tokenizer Evaluation
    print("\n[STEP 2] Running Tokenizer Evaluation...")
    # Use Chapter 1 as a representative sample
    sample_file = "extracted/iesc101.txt"
    if os.path.exists(sample_file):
        run_command(["python3", "src/tokenizer_evaluation.py", sample_file])
    else:
        print(f"Warning: Sample file {sample_file} not found. Skipping tokenizer evaluation.")

    # 3. Retrieval Demo
    print("\n[STEP 3] Running Retrieval Demo...")
    run_command(["python3", "src/retrieval.py"])

    # 4. Grounded Generation (Stage 3)
    print("\n[STEP 4] Running Grounded Generation Test (Groq)...")
    run_command(["python3", "src/groq_generation.py"])

    # 5. Evaluation (Stage 4)
    print("\n[STEP 5] Running Stage 4 Evaluation...")
    run_command(["python3", "src/evaluate.py"])

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 60)

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--evaluate":
        run_command(["python3", "src/evaluate.py"])
    else:
        main()
