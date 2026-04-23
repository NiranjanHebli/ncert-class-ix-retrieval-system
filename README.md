# PariShiksha NCERT Science QA Retrieval System

## Environment Setup
The project uses Python 3.10+.

1. Set up the virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:
```bash
pip install pymupdf pdfplumber transformers tokenizers torch rank_bm25 google-generativeai sentence-transformers
```

or 

```bash 
pip install -r requirements.txt
```

3. Ensure you have a `.env` file with your Gemini free-tier API key:

```bash
cp .env.example .env
```

```env
API_KEY=your_gemini_api_key
OCR_SPACE_API_KEY=your_ocr_space_api_key
```

## Data Source

NCERT science textbook source:
[https://ncert.nic.in/textbook.php?iesc1=0-11](https://ncert.nic.in/textbook.php?iesc1=0-11)

Download Chapter files as: `iesc1XX.pdf` (e.g., `iesc102.pdf` = Chapter 2: Motion) and place them in the `iesc1dd/` directory.

## How to Extract PDF Content

### 1. Extraction using pymupdf4llm (Recommended)
To extract text using `pymupdf4llm` (which preserves formulas and equations) and automatically split them:

```bash
python src/scripts/extract.py
```

### 2. OCR Extraction
Extracting PDF using OCR Space API (image-based OCR):

```bash
python src/scripts/old_extract.py
```

### 3. Splitting the text files into sections (Optional)
If you already have text files and only need to re-run the splitting logic:

```bash
python src/scripts/split_sections.py
```

## Running the Retrieval System

### 1. Vector Database & Retrieval Demo
To run the primary retrieval demo which tests semantic (FAISS) and keyword (BM25) search:

```bash
cd src
python retrieval_demo.py
```

### 2. Tokenizer & Chunking Evaluation
To evaluate different tokenizers and chunking configurations:

```bash
cd src
python tokenizer_evaluation.py
```

Results and comparison tables will be saved in the `data/` directory.
