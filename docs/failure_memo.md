# Failure Memo (Study Assistant v2.0)

Based on the evaluation of the v2.0 pipeline, the system now successfully grounds generation and correctly refuses out-of-scope questions. However, the top three failure modes that the v2.0 system *still* exhibits are detailed below, along with the smallest planned fixes for Wk11.

## 1. Plausibly Answerable Out-of-Scope Hallucinations
* **Where it fails:** When a question is technically outside the syllabus but mentions entities present in the textbook (e.g., "Calculate the value of g on the surface of the Moon").
* **Why it fails:** The model retrieved chunks about weight on the Moon vs Earth. Instead of strictly refusing as instructed for a calculation not in context, it attempted to derive a solution using partial formulas. This violates the "hard refusal" constraint but happens because the model finds "highly relevant" context.
* **Smallest fix (Wk11):** Implement a dedicated "Calculation Guardrail" prompt that explicitly tells the model to refuse any math problem where all variables/constants are not explicitly provided in the retrieved chunks.

## 2. Table Column Ambiguity
* **Where it fails:** Queries that demand correlating values across complex or borderless tables. For example, "What is the specific heat capacity of X compared to Y?" when extracted from a dense summary table.
* **Why it fails:** While `OpenDataLoader-PDF` and structure-aware chunking improved the preservation of simple worked examples, highly dense tabular data sometimes still flattens into unstructured sequences when fed into the LLM context window. The retriever finds the correct chunk, but the generator gets confused by the flattened schema (Speculation: The LLM prompt lacks instructions on how to parse markdown pipes `|` reliably for dense cross-referencing).
* **Smallest fix (Wk11):** Pass the extracted table as a structured JSON object or inject a schema definition prompt into the generator specifically when the retrieved chunk is of `content_type=table`.

## 3. Over-Refusal of Paraphrased Content
* **Where it fails:** Some valid textbook concepts are refused when phrased as a narrative or complex process (e.g., "Explain how a drop of ink spreads in water").
* **Why it fails:** The hybrid retriever found chunks about diffusion, but the model likely didn't see an explicit "drop of ink" example in those specific 5 chunks. The strict prompt then forced a refusal because of the "Do not infer" rule.
* **Smallest fix (Wk11):** Loosen the refusal threshold slightly by providing "Example-based Grounding" where the model is shown how to bridge a paraphrased user query to a textbook concept without "inventing" new facts.
