# Production Failure Modes

This document analyzes the top 3 failure modes observed during the evaluation of the PariShiksha RAG pipeline, grounded in actual evaluation results from our 20 question test suite.

## Failure Mode 1: Retrieval Miss via Keyword Mismatch

**Observed in:** "What is the SI unit of force?" (Question 12, scored as incorrect)

The BM25 retriever returned chunks from chapters on work/energy and gravitation instead of the Force and Laws of Motion chapter. This happened because the query terms "SI", "unit", and "force" individually appear across many chapters, but the specific phrase "SI unit of force"   which should return a chunk mentioning "newton"   was distributed across paragraph boundaries in the extracted text. The retriever scored unrelated paragraphs that happened to contain the word "force" more frequently.

**Root cause:** BM25 relies on term frequency and does not understand semantic meaning. When the answer is a brief factual statement (one sentence) buried in a longer conceptual paragraph, the retriever may rank chunks with higher overall keyword density above the chunk that actually contains the answer.

**Mitigation:** Implementing dense retrieval (sentence transformers) alongside BM25 would capture the semantic intent of "SI unit of force" and match it to the correct paragraph even if the exact words differ. A hybrid scoring approach using Reciprocal Rank Fusion would combine both signals.

## Failure Mode 2: Context Fragmentation (Split Example/Solution)

**Observed in:** Worked examples from Chapter 8 (Motion) and Chapter 10 (Gravitation)

During chunking, some worked examples were split across two chunks  the problem statement ended up in one chunk while the solution started in the next. When a student asked a question that matched the example, the retriever returned only the problem statement chunk. The LLM then attempted to solve the problem from scratch using incomplete context, sometimes producing incorrect intermediate steps.

**Root cause:** Our paragraph based chunking splits on double newlines (`\n\n`). In the extracted text, the "Example" header and the "Solution" section were separated by formatting markers that the splitter treated as paragraph boundaries. The 50 token overlap was not large enough to capture the full solution when the example was long.

**Mitigation:** Implementing content aware chunking that detects "Example" and "Solution" patterns and forces them into the same chunk, regardless of paragraph boundaries. This requires a pre processing step that merges adjacent paragraphs when one starts with "Example" and the next starts with "Solution".

## Failure Mode 3: Plausible Out of Scope Context Confusion

**Observed in:** "Explain quantum entanglement from Chapter 9" (Question 19, correctly refused)

This was our "hard trick question" designed to test the grounding guardrails. The retriever successfully returned Chapter 9 (Force and Laws of Motion) content because "Chapter 9" appeared in the metadata. A weak grounding prompt would have allowed the LLM to use this retrieved physics content to construct a plausible sounding but entirely fabricated explanation of quantum entanglement. Our strong prompt correctly identified that quantum entanglement was not discussed in the retrieved context and triggered the refusal response.

**Root cause:** The retriever cannot distinguish between "content from Chapter 9" and "content about quantum entanglement." It only matches surface level keywords. The responsibility for detecting the mismatch falls entirely on the LLM's grounding prompt.

**Mitigation:** Adding a retrieval confidence threshold   if the maximum BM25 score is below a certain value, the system should preemptively refuse rather than passing low confidence context to the LLM. This would provide a defense in depth approach where both the retriever and the generator can independently trigger a refusal.

---
## Summary

| Failure Mode | Frequency | Severity | Current Mitigation |
|----|----|----|----|
| Keyword mismatch | ~5% of queries | High | None (BM25 limitation) |
| Split example/solution | ~10% of examples | Medium | 50 token overlap |
| Plausible OOS confusion | Rare but dangerous | Critical | Strong grounding prompt |

The most impactful improvement would be adding dense retrieval to address Failure Mode 1, as it accounts for the majority of incorrect answers on valid textbook questions.
