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
