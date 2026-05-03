# PariShiksha NCERT Science QA Retrieval System

![Python](https://img.shields.io/badge/python-3.13-3670A0?logo=python&logoColor=ffdd54)
![PyTorch](https://img.shields.io/badge/PyTorch-2.11.0-%23EE4C2C.svg?logo=PyTorch&logoColor=white)
![Transformers](https://img.shields.io/badge/Transformers-5.5.4-yellow)
![Gemini](https://img.shields.io/badge/Gemini-2.5%20Flash-8E75B2?logo=google%20gemini&logoColor=white)
![Groq](https://img.shields.io/badge/Groq-Llama%203.1-f55036)
![FAISS](https://img.shields.io/badge/FAISS-1.13.2-blue)
![BM25](https://img.shields.io/badge/Rank--BM25-0.2.2-green)
![Pandas](https://img.shields.io/badge/pandas-3.0.2-%23150458.svg?logo=pandas&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.10.8-%23ffffff.svg?logo=Matplotlib&logoColor=black)

## Problem Statement

PariShiksha is an ed-tech non-profit rolling out AI-powered tutoring centres in Tier-2 and Tier-3 Indian cities. Students in these centres need a study assistant that can answer questions from the NCERT (Class 9 Science) textbooks accurately and reliably. The assistant must:

- **Answer only from the textbook** — hallucinated or fabricated answers erode tutor and parent trust
- **Refuse gracefully** when asked questions outside the NCERT syllabus
- **Handle messy real-world input** — PDF extraction artifacts, code-switched queries, paraphrased questions

This project builds a Retrieval-Augmented Generation (RAG) pipeline that extracts, chunks, and indexes NCERT Science content, retrieves relevant passages using a **Hybrid Retrieval approach (BM25 + Dense FAISS/ChromaDB)**, and generates grounded answers using LLMs. Due to persistent rate-limiting and quota issues with the Gemini API (specifically `gemini-2.0-flash`), we have transitioned the evaluation and generation logic to use the **Groq API (Llama 3.1 8B)**. This ensures stable and fast generation during evaluation and interactive queries with strict grounding prompts. The system is evaluated on 20 questions across 3 categories (direct, paraphrased, out-of-scope) using a 3-axis scoring framework (correctness, groundedness, refusal appropriateness).


## Demo Video

Watch the end-to-end demonstration of the PariShiksha NCERT Science QA system:

- [PariShiksha Demo Video (Loom)](https://www.loom.com/share/92827fee7ea24c75b39f69aad02e708b)


## Requirements Doc (For V2)

We made use of the following document to build the V2 of our RAG pipeline.

You can check the URL below :-

- [Requirement Doc V2](https://drive.google.com/file/d/1CNPMb2BSiHHLYUGMq3VfxeHymO7JIcqE/view?usp=sharing)

## Environment Setup

The project uses Python 3.10+.

1. Set up the virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install pymupdf pdfplumber transformers tokenizers torch rank_bm25 google-genai groq sentence-transformers
```

or

```bash
pip install -r requirements.txt
```

3. Ensure you have a `.env` file with your Gemini free-tier and Groq API keys:

```bash
cp .env.example .env
```

```env
API_KEY=your_gemini_api_key
GROQ_API_KEY=your_groq_api_key
OCR_SPACE_API_KEY=your_ocr_space_api_key
```

## Data Source

NCERT science textbook source:
[https://ncert.nic.in/textbook.php?iesc1=0-11](https://ncert.nic.in/textbook.php?iesc1=0-11)

Download Chapter files as: `iesc1XX.pdf` (e.g., `iesc102.pdf` = Chapter 2: Motion) and place them in the `iesc1dd/` directory in the root.

## Running the Study Assistant v2.0 Pipeline

Execute the scripts in the following order to run the complete V2.0 pipeline:

### 1. Data Extraction & Structure-Aware Chunking
Extracts raw text from NCERT PDFs and applies structure-aware chunking (preserving tables and examples).

```bash
python src/improved_chunking.py
```

### 2. Database Initialization & Benchmarking
Indexes the generated chunks into a persistent **ChromaDB** vector store using dense embeddings (`all-mpnet-base-v2` / `bge-small`).

```bash
python src/db_benchmark.py
```

### 3. Testing Hybrid Retrieval
Initializes the BM25 (keyword) and ChromaDB (semantic) indices, combining scores via Reciprocal Rank Fusion (RRF).

```bash
python src/hybrid_retrieval.py
```

### 4. Interactive Q&A (Grounded Generation)
Connects the Hybrid Retriever to the **Groq API** (`llama-3.1`), using strict grounding prompts to enforce citations and hard-refuse out-of-scope queries.

```bash
python src/hardened_generation.py
```

### 5. Running the V2.0 Evaluation
Runs the 20-question evaluation set through the hardened pipeline, autoscoring responses on correctness, groundedness, and refusal appropriateness.

```bash
python src/evaluate_v2.py
```
Outputs: `data/eval_scored.csv`

### 6. Generating the RAGAS Metrics Report (Stretch Goal)
Takes the evaluation scores and queries the Groq API as an LLM-judge to calculate advanced RAGAS metrics (Faithfulness, Answer Relevancy, Context Precision, Context Recall).

```bash
python src/generate_ragas_report.py
```
Outputs: `data/ragas_report.csv`

To save tokens (evaluates first 3 questions):
```bash 
python src/generate_ragas_report.py --sample 3
```

### Interpreting RAGAS Results
When running the report, you may see a warning: `LLM returned 1 generations instead of requested 3`. This is **expected** behavior. Groq's API currently supports `n=1`, so the RAGAS judge performs single-shot evaluation instead of its default triple-check mode.

*   **Faithfulness (~1.0):** High score means the model is NOT hallucinating (it only uses the provided text).
*   **Context Precision (~0.2-0.5):** This is often lower in v2.0 because we retrieve 5 chunks to ensure coverage, but the answer is usually contained in just 1. This "noise" is normal for a robust retriever.
*   **Context Recall (~1.0):** High score means the retriever found all the facts needed to answer.



## High-Level Design

![High-Level Design](./docs/diagrams/HLD.png)

View Complete Diagram :- [HLD](./docs/diagrams/HLD.png)

The system follows a production-grade RAG pipeline. PDFs are extracted into Markdown and chunked using structure-aware logic, indexed via **BM25 and ChromaDB**, and retrieved using **Hybrid RRF with Cross-Encoder reranking**. The response is generated through a hardened grounding prompt that forces the LLM to either answer with citations or refuse.


| Layer      | Technology     | Purpose                                 |
| ---------- | -------------- | --------------------------------------- |
| Extraction | pymupdf4llm    | PDF → structured markdown              |
| Chunking   | Structure-Aware| Preserves tables and worked examples    |
| Retrieval  | Hybrid (RRF)   | BM25 + ChromaDB (Dense)                 |
| Generation | Groq: Llama-3.1| Grounded answer with hard refusal       |
| Evaluation | RAGAS          | Automated metrics via LLM-as-Judge      |

## Project Documentation

| Document                                         | Description                                        |
| ------------------------------------------------ | -------------------------------------------------- |
| [notebook.ipynb](notebook.ipynb)                    | End-to-end pipeline demonstration (v2.0)          |
| [reflection.md](docs/reflection.md)                 | Project Reflection (Updated for v2.0)             |
| [failure_memo.md](docs/failure_memo.md)             | Top 3 production failure modes analysis            |
| [ragas_report_memo.md](docs/ragas_report_memo.md)   | Explanation of RAGAS metrics and scores            |
| [chunking_strategy.md](docs/chunking_strategy.md)   | Chunking size, overlap, and strategy justification |
| [db_comparison.md](docs/db_comparison.md)           | Comparison of BGE-Small vs MPNet-Base embeddings   |
| [retrieval_misses.md](docs/retrieval_misses.md)     | Diagnosis of wrong retrievals                      |
| [prompt_diff.md](docs/prompt_diff.md)               | Verbatim permissive vs strict prompt responses     |
| [fix_memo.md](docs/fix_memo.md)                     | Rationale and results of the Hybrid Retrieval fix  |
| [chunking_compare.md](docs/chunking_compare.md)     | Comparison of Regex vs Heading-anchored chunking   |

## Diagnostic Tools

A dedicated retrieval test script is available in the tests directory:

```bash
python tests/test_retrieval.py
```

## Contributors 

* [Bryson Gracias](https://github.com/MrGladiator14)
* [Niranjan Hebli](https://github.com/NiranjanHebli)
