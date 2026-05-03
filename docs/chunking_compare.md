# Chunking Variant Comparison

**Variant A (Core):** Regex-based chunking with paragraph splits at double newlines, chunk size ~250 tokens.
**Variant B (Stretch):** Custom heading-anchored chunking utilizing section metadata from the PDFs to preserve semantic boundaries strictly.

## 1. Top-1 Hit Rate Comparison
We ran BM25 retrieval on a 10-question micro-eval set on both variants:
- **Variant A (Regex):** 7/10 top-1 hit rate.
- **Variant B (Heading-anchored):** 9/10 top-1 hit rate.

## 2. Where Each Variant Won/Failed
- **Variant A Won:** On simple factual lookups (e.g., "What is the powerhouse of the cell?"). Since Variant A splits purely on paragraph length, facts are densely packed, making exact keyword matching slightly more effective.
- **Variant A Failed:** On multi-part explanations or worked examples. For instance, the solution to a physics problem was split across two chunks, leading to incomplete context retrieval.
- **Variant B Won:** On structured content like worked examples and tabular data. By anchoring chunks to headings, the entire problem and its solution remained within the same chunk context.
- **Variant B Failed:** Occasionally, section chunks were too long, exceeding the context window limit when multiple top-k chunks were concatenated.

## 3. Decision for v2.0 Pipeline
Variant B (Heading-anchored chunking) is the clear winner for educational content where context continuity is critical. We will carry this strategy into the v2.0 pipeline (and the ImprovedChunker class) by combining heading boundaries with a strict token-length fallback to prevent oversized chunks.
