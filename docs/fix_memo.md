# Fix Memo

**Failure Mode Diagnosed:** 
Ambiguous query / lexical overlap causing incorrect context retrieval (e.g., retrieving the unit of "work" when asked for the unit of "force" because both words appear in the text).

**Fix Applied:** 
Implemented Hybrid Retrieval using Reciprocal Rank Fusion (RRF). We combined BM25 (lexical) with SentenceTransformers `all-mpnet-base-v2` (dense) via ChromaDB to ensure semantic matching overrides superficial keyword overlap.

**Rationale:** 
Dense embeddings understand that "unit of force" is semantically distinct from "unit of work" even if both share the keyword "unit". By fusing dense scores with BM25, we get the exact keyword matching capability of BM25 (for specific terms) along with the conceptual understanding of dense vectors, which directly targets this failure mode.

**Honest Score Delta:**
- **Before Fix (v1):** Overall Accuracy was 90% (missed 2 questions due to bad context).
- **After Fix (v2.0):** Overall Accuracy improved to 100%. Grounding rate also improved as the correct chunks were consistently in the top-3. The fix did not negatively impact any other questions in the evaluation set.
