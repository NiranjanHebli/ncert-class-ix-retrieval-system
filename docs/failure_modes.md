# Failure Mode Catalog (Week 10)

Based on Stage 4/5 analysis of the Study Assistant v2.0, here are the documented failure modes and their diagnoses.

## 1. Synonym/Acronym Mismatch
- **Diagnosis:** Dense retrieval (semantic) handles synonyms well, but BM25 (lexical) fails when keywords don't match exactly.
- **Example:** User asks for "g value" but text says "acceleration due to gravity".
- **Fix:** Use MultiQuery to expand "g" to "acceleration due to gravity".

## 2. Lost Cross-Reference
- **Diagnosis:** The answer depends on information from another page or chapter that isn't retrieved in the top-k.
- **Example:** "Compare plant cells (Chapter 5) to animal cells (Chapter 5)" but only plant cell chunks are retrieved.
- **Fix:** Increase `k` for initial retrieval or use Query Decomposition (Multi-hop).

## 3. Multi-hop Reasoning
- **Diagnosis:** The question requires combining two facts from different parts of the textbook.
- **Example:** "How does the structure of a cell (Ch 5) relate to the work done by muscles (Ch 10)?"
- **Fix:** MultiQuery/Decomposition to retrieve from both chapters.

## 4. Mixed Structure (Tables/Formulas)
- **Diagnosis:** Chunking breaks a table or formula, losing the relationship between rows/columns.
- **Example:** A table of densities where the substance is in one chunk and the value in another.
- **Fix:** Structure-aware chunking (Markdown table preservation).

## 5. Ambiguous Query
- **Diagnosis:** The user's query is too vague, leading to retrieval of noisy/irrelevant chunks.
- **Example:** "Explain the process." (Which process?)
- **Fix:** Strict grounding prompt to refuse or ask for clarification (though refusal is the Wk10 goal).
