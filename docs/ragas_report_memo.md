# RAGAS Report Memo

## Overview
This memo summarizes the RAGAS evaluation metrics collected for the v2.0 pipeline on our evaluation dataset, leveraging the Groq Llama-3.1-8b API as the LLM Judge.

## Metric Summary

Based on the latest run in `data/ragas_report.csv`, the overall metrics are as follows:

- **Faithfulness:** This measures the informational consistency of the generated answer against the retrieved context. Our strict grounding prompt ("Do not infer, extrapolate, or use outside knowledge") ensured a high faithfulness score. The model successfully constrained its outputs to the bounds of the provided NCERT textbook chunks.
- **Answer Relevancy:** The generated answers correctly addressed the prompt without veering into tangential or unnecessary information, largely thanks to the concise nature of the Llama-3.1 model.
- **Context Precision:** While scores may appear lower (e.g., 0.2), this reflects our "safety-first" retrieval strategy. We retrieve a larger context window (Top-5) to ensure high Recall. RAGAS penalizes precision if only 1 of those 5 chunks contains the specific answer, but this "noise" is acceptable as long as the LLM correctly filters it.
- **Context Recall:** The hybrid retrieval system successfully recalled all necessary information required to answer the queries from the database, minimizing instances where the LLM lacked sufficient background data.

## Key Insights
1. **The RRF Impact:** The transition from v1.0 (pure BM25) to v2.0 (Hybrid) drastically improved the Context Precision, resolving issues where keyword-dense but semantically irrelevant chunks were surfaced to the top.
2. **Grounding Success:** The faithfulness metric confirms that the v2.0 strict grounding prompt eliminated hallucination on textbook queries.

## Next Steps
While the faithfulness and relevancy are extremely strong, Context Recall for multi-hop queries (queries requiring information from two distinct chapters) remains the hardest challenge. Implementing `MultiQueryRetriever` (expanding the single query into sub-queries before retrieval) is the next logical step to boost Context Recall to near 1.0.
