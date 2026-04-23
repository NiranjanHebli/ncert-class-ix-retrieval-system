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
python scripts/extract.py
```

This is the recommended approach as it handles mathematical notation much better and runs locally without API limits.

### 2.  OCR Extraction
Extracting PDF usin OCR Space API (image-based OCR):

```bash
python scripts/old_extract.py
```

*Note: This requires an `OCR_SPACE_API_KEY` in your `.env` file.*

### 3. Splitting the text files into sections (Optional)
If you already have text files and only need to re-run the splitting logic:

```bash
python scripts/split_sections.py
```

The organized files will be stored in subdirectories within `docs/`, named after each PDF (e.g., `docs/iesc101/iesc101_section_1.1.txt`).
