# Data Organization & RAG Benefits

##   Extracted Content Structure
The extraction pipeline organizes NCERT content into specific subdirectories within the `extracted/` folder to optimize retrieval performance.

### 1. Concepts (`/concepts`)
- **Content**: Pure theoretical explanations, definitions, and conceptual prose.
- **Filter**: Specifically excludes numerical examples and test questions.
- **Use Case**: Best for answering "What is..." or "Explain the process of..." queries.

### 2. Worked Examples (`/worked_examples`)
- **Content**: Step-by-step problems and their solutions identified by keywords like "Example" and "Solution".
- **Use Case**: Ideal for "How do I solve..." or "Show an example of..." queries.

### 3. Exercises (`/exercises`)
- **Content**: End-of-chapter practice questions and assessment problems.
- **Use Case**: Useful for generating similar practice tests or checking textbook questions.

### 4. Sanitized Paragraphs (`/paragraphs`)
- **Content**: A normalized version of the entire textbook, split strictly into blocks by double newlines.
- **Cleaning**: Leading/trailing whitespace is stripped, and redundant empty lines are collapsed.
- **Use Case**: The primary source for general vector retrieval and structural chunking.

---

##  How this improves RAG
Organizing content into semantic categories provides three major technical advantages for Retrieval-Augmented Generation:

### 1. Noise Reduction (Signal-to-Noise Ratio)
By separating theoretical `concepts` from practice `exercises`, the retriever avoids fetching irrelevant question prompts when the user is looking for a factual explanation. This ensures the LLM receives higher-quality context.

### 2. Intent-Aware Retrieval
The system can implement "routing" logic. If a query is identified as a problem-solving request, the retriever can prioritize the `worked_examples` folder, drastically increasing the precision of the retrieved context.

### 3. Better Grounding & Reduced Hallucination
The `paragraphs` folder provides "sanitized" context. By feeding the LLM clean, coherent blocks of text rather than messy fragments with irregular spacing, the model can better "ground" its answers in the provided text, significantly reducing the likelihood of hallucination.
