# Project Reflection

## Part A  - Implementation Artifacts

### A1. Your chunking parameters

**Parameters:** chunk_size=180 tokens, overlap=50 tokens, tokenizer=BERT WordPiece (`bert  base  uncased`).

**Special handling:** Content  type classification splits chunks into `concept`, `example`, `exercise`, and `solution` based on keyword detection. Paragraphs are split on double newlines, and if a single paragraph exceeds 180 tokens, it is split into overlapping windows.

**The observation that pushed me to these values:** When I initially used chunk_size=300, I noticed that worked examples from Chapter 10 (Gravitation) were being split the problem statement ended up in one chunk and the "Solution:" text started in the next. When I retrieved context for "Calculate the gravitational force between two objects," the retriever returned only the problem statement chunk, and the LLM tried to solve it from scratch (incorrectly). Increasing to 180 tokens kept most example  solution pairs together. I confirmed this by visually inspecting chunks from `iesc110` and counting how many examples were split before and after the change.

### A2. A retrieved chunk that was wrong for its query

**Query:** "What is the SI unit of force?"

**Retrieved chunk (from iesc111_paragraphs):**

> "...the unit of work is newton metre (N m) or joule (J). Thus 1 J = 1 N m = 1 kg m2 s  2. Work done is also defined as the product of component of force..."

**Why it was returned:** BM25 scored this chunk highly because it contains the words "unit", "force", and "newton"   all present in the query. But the chunk is about the unit of *work*, not the unit of *force*. The retriever cannot distinguish between "unit of force" and "unit of work" because BM25 treats words independently without understanding the semantic relationship. The actual answer ("newton" as the SI unit of force) appears briefly in a Chapter 9 chunk but was ranked lower because that chunk had fewer total occurrences of "unit."

### A3. Your grounding prompt, v1 and v(final)

**v1 (weak):**

```
Answer only from the context below.
Context: {context}
Question: {question}
```

**Observation that caused revision:** When I tested v1 with "Who is the current Prime Minister of India?", the retriever returned some political science  adjacent text from the chapter introductions. The LLM treated "only answer from context" as a *preference* and generated a plausible answer by combining fragments from the retrieved chunks with its own knowledge. It did not refuse.

**v_final (strong):**

```
You are a study assistant for PariShiksha.
Use ONLY the context provided below to answer the question.
If the answer is not present in the context, respond with:
"This question is outside the provided NCERT content."
Do not infer, extrapolate, or use outside knowledge.

Context: {context}
Question: {question}
Answer:
```

**What changed:** Adding the explicit refusal phrase ("This question is outside the provided NCERT content") and the hard constraint ("Do not infer, extrapolate, or use outside knowledge") turned the prompt from a soft preference into a hard constraint. After this change, all 5 out  of  scope questions were correctly refused, including the trick question "Explain quantum entanglement from Chapter 9."

## Part B - Numbers from Your Evaluation

### B1. Your evaluation scores

Out of **20 questions** in the evaluation set:
   **(a) Correct:** 19/20 (95%)
   **(b) Grounded:** 19/20 (95%)
   **(c) Appropriate refusals:** 5/5 (100%) for out  of  scope questions

**Which number bothered me most:** The 1 incorrect answer   "What is the SI unit of force?"   bothered me because it is a straightforward factual question that any student would expect the system to answer. The failure was entirely a retrieval problem (the retriever ranked a "unit of work" chunk above the "unit of force" chunk), not a generation problem. This showed me that BM25's keyword  matching can fail even on simple questions when the vocabulary overlaps with other concepts.

### B2. Chunk  size experiment

I did not run a formal chunk  size experiment with controlled metrics. However, I informally tested 300 vs 180 tokens during development. At 300 tokens, 2 out of 5 worked examples from Chapter 10 were split across chunks. At 180 tokens, only 1 was split. I chose 180 as the better tradeoff between keeping examples intact and not diluting retrieval signal.

### B3. Model family comparison

I compared **Groq (Llama 3.1 8B)** and **Google Gemini 2.5 Flash**.

   Llama 3.1 was more concise and faster (sub  second responses via Groq).
   Gemini produced longer, more detailed answers with better formatting.
   Both correctly refused all 5 out  of  scope questions.

**Specific difference:** For "What is the difference between a plant cell and an animal cell?", Llama 3.1 gave a 3  sentence summary, while Gemini produced a structured comparison with bullet points. Both were correct and grounded, but Gemini's answer would be more useful to a student studying for an exam.

## Part C  -  Debugging Moments

### C1. The most frustrating bug

The BERT tokenizer threw a warning: "Token indices sequence length is longer than the specified maximum sequence length for this model (1495 > 512)." This happened when encoding full paragraphs from Chapter 7 (Diversity in Living Organisms), which had very long prose sections.

**How long to fix:** About 2 hours.

**What I tried first (didn't work):** I tried setting `truncation=True` in the tokenizer call, but that silently dropped content   chunks were missing the end of paragraphs, and I didn't notice until a retrieval test returned incomplete context.

**Actual fix:** I changed the approach to encode the full text first (allowing sequences longer than 512 for the tokenization step only), then manually split the token sequence into 180  token windows with 50  token overlap. This preserved all content while producing BERT  compatible chunks.

**Fastest way for someone else to fix:** Check `vec_retrieval.py`, `chunk_text_bert()` method. The key is: encode the full paragraph without truncation, then split the token IDs into windows, then decode each window back to text.

### C2. What still bothers me

The system refused "What is the SI unit of force?"   a valid textbook question. This is a false refusal caused by a retrieval miss, and it would confuse a real student. The retriever returned chunks about "unit of work" instead of "unit of force" because both share the same keywords. This bothers me because it means the system can fail on simple factual questions, which undermines trust. To fix it, I would need to implement dense retrieval (sentence  transformers) that understands "SI unit of force" as a semantic concept, not just a bag of words.

## Part D - Architecture and Reasoning

### D1. Why not just ChatGPT?

ChatGPT would answer "What is the SI unit of force?" correctly   but it would also answer "Explain quantum entanglement from Chapter 9" confidently, fabricating content that does not exist in the NCERT textbook. In our evaluation, our RAG system correctly refused this trick question because the grounding prompt forced it to check whether quantum entanglement was in the retrieved context (it wasn't). ChatGPT has no such constraint   it would generate a plausible physics explanation and a student would study it for an exam, not knowing it was fabricated. For PariShiksha, where parents trust the system to be accurate, one hallucinated answer can destroy that trust. A retrieval system gives us *control* over what the model can say.

### D2. The GANs reflection

GANs optimize for generating outputs that are indistinguishable from real data   the generator tries to fool the discriminator. This is fundamentally wrong for textbook QA because we don't want "realistic  looking" answers; we want verifiably correct answers grounded in a specific source. A GAN  trained generator would learn to produce fluent, plausible  sounding science text that looks like it came from a textbook   but it would have no mechanism to ensure the content is actually from the textbook. The deeper principle: generative diversity (GANs) and generative fidelity (RAG) are opposite goals. When the cost of a wrong answer is high (education, healthcare, legal), you need architectures that prioritize fidelity to source over fluency of output.

### D3. Honest pilot readiness

**Honest answer:** Not yet. The system is promising but not ready for 100 students.

**Three things to verify or fix first:**

1. **Fix the false refusal problem.** The "SI unit of force" failure shows that valid questions can be refused. I would need to add dense retrieval to reduce retrieval misses before students encounter them.
2. **Test with real student queries.** Our 20  question eval set was written by us, not by actual Class 9 students. Real queries will include Hindi  English code  switching, typos, and colloquial phrasing that we haven't tested.
3. **Add latency monitoring.** In Tier  2/3 cities with unreliable internet, API calls to Groq/Gemini might time out. I would need to measure p95 latency and add a fallback for when the API is unavailable.

## Part E - Effort and Self  Assessment

### E1. Effort rating

**8/10.** I am genuinely proud of the modular architecture   having separate retrieval, generation, and evaluation layers with persistent storage meant I could iterate on any component without breaking the others. The local VectorDB persistence (save/load to disk) saved significant development time and is a pattern I would use in production.

### E2. The gap between you and a stronger student

A stronger student would have implemented a proper hybrid retrieval system with Reciprocal Rank Fusion (BM25 + dense retrieval) and tested it against the same eval set to show a quantitative improvement. I did not do this because I prioritized getting the evaluation framework right first, and ran out of time before implementing the dense retrieval comparison. The FAISS infrastructure is in the code but not fully integrated into the evaluation pipeline.

### E3. What would change with two more days

**First thing:** Implement sentence  transformer dense retrieval and run the full 20  question eval through both BM25 and hybrid retrieval. This directly addresses the biggest failure mode (keyword mismatch on "SI unit of force") and would produce a concrete comparison table.

**Last thing:** Have 5 people outside the cohort (ideally actual Class 9 students) write questions without looking at the textbook. Test those queries and add the results to the evaluation. This would be the most honest test of whether the system is ready for real users.
