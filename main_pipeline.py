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
    run_command(["python3", "src/retrieval_demo.py"])

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 60)

if __name__ == "__main__":
    main()
