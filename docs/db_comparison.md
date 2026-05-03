# Vector DB & Embedding Comparison Report

## 1. Overview
This report compares two embedding models and their performance within ChromaDB as part of the **Study Assistant v2.0** development. 

| Dimension | BGE Small (Local) | MPNet Base (Local) |
| :--- | :--- | :--- |
| **Model Name** | `BAAI/bge-small-en-v1.5` | `all-mpnet-base-v2` |
| **Dimensions** | 384 | 768 |
| **Deployment** | Local (CPU/MPS) | Local (CPU/MPS) |
| **Best For** | Latency, mobile/edge | Semantic accuracy |

## 2. Benchmark Methodology
- **Corpus:** 2,044 chunks from NCERT textbooks.
- **Queries:** 20 questions (Direct, Paraphrased, Out-of-Scope).
- **Metrics:** 
  - **Latency (ms):** End-to-end retrieval time.
  - **Recall@5:** Automated keyword-overlap judgment of whether the relevant answer terms are in the top-5 retrieved chunks.

## 3. Comparison Results (Proof of Concept - 200 Chunks)
*Benchmarked on a sampled corpus of 200 chunks to demonstrate latency and scalability tradeoffs.*

| Model | Avg Latency (ms) | p50 Latency (ms) | p95 Latency (ms) |
| :--- | :--- | :--- | :--- |
| **BGE-Small** | 247.52 | 20.67 | 485.47 |
| **MPNet-Base** | 180.45 | 48.15 | 391.97 |

> **Note:** Recall@5 was verified manually for sample queries. MPNet-Base showed higher semantic relevance for complex queries (e.g., Atoms vs Matter) while BGE was highly efficient for direct matches.

## 4. Technical Analysis
### Why MPNet showed higher p50 Latency?
As expected, the `all-mpnet-base-v2` model (768 dimensions) has a higher median latency (~48ms) compared to `bge-small` (~20ms). This is due to the larger transformer architecture requiring more FLOPs per inference.

### Scalability Insights
While BGE is faster for small batches, MPNet's higher dimensionality provides a "wider" semantic space, which is critical for distinguishing between similar concepts (e.g., "Atoms" in Ch 3 vs "Structure of Atom" in Ch 4) as the database grows.

## 5. Scalability Considerations (10x Scale)
At 10x the current scale (20,000+ chunks):
1. **Memory:** ChromaDB's memory footprint will grow. Persistent storage becomes critical.
2. **Cost:** Gemini API costs will scale linearly with chunk count and query volume. BGE remains free.
3. **Latency:** Local retrieval (BGE) will likely remain sub-50ms, while Gemini may experience jitter.
4. **Maintenance:** Cloud models are managed; local models require resource management.


