# Chunking Strategy & Tokenizer Evaluation

##  Evaluation Summary
After evaluating three major tokenizer families (GPT-2 BPE, BERT WordPiece, and T5 SentencePiece) across 12 NCERT Science textbook chapters, we have determined the optimal chunking configuration for this corpus.

### Top Performing Configuration
- **Overall Best Tokenizer**: `bert-base-uncased` (WordPiece)
- **Optimal Chunk Size**: 500 characters
- **Optimal Overlap**: 50 characters
- **Average Compression Ratio**: ~3.7 - 3.8
- **Average Tokens per Chunk**: ~115 - 125

##  Key Insights from Evaluation

### 1. Tokenizer Performance
The **BERT WordPiece** tokenizer consistently outperformed GPT-2 and T5 on this scientific corpus. It achieved a higher compression ratio (~3.8) while maintaining better token consistency (lower standard deviation in tokens per chunk), which is critical for stable retrieval scores.

### 2. Optimal Chunk Size (500 characters)
A chunk size of **500 characters** was found to be the "sweet spot."
- **Reasoning**: It provides enough context for complete scientific concepts (e.g., definitions of inertia or chemical equations) without exceeding the 512-token limit of modern transformer models. Larger chunks (800+) often resulted in lower vocab utilization and less granular retrieval.

### 3. Context Retention (50 char overlap)
An overlap of **50 characters** (~10% of chunk size) provided the highest composite scores.
- **Reasoning**: This overlap ensures that scientific definitions or examples that span the end of one chunk are "carried over" into the next, preventing semantic fragmentation.

##  Implementation Details

### Content Classification
To further improve retrieval grounding, we complement the fixed-size chunking with semantic classification:
1.  **Concepts**: Descriptive prose and scientific principles.
2.  **Worked Examples**: Labeled with "Example" and "Solution" keywords to ensure problem-solution pairs stay reachable.
3.  **Exercises**: End-of-chapter questions for specific testing.

### Scoring Methodology
Our evaluation script used a composite score based on:
- **Compression Ratio (50%)**: Efficient representation of the scientific text.
- **Token Consistency (50%)**: Uniformity in chunk lengths to ensure predictable model behavior.
- **Context Bonus**: A 20% score multiplier for configurations with 10-30% overlap.

---
*Summary based on evaluation results from `src/tokenizer_evaluation.py` run on 2026-04-23.*

## References 

[Evaluation Results](../data/tokenizer_evaluation.log)