# Evaluation Results

**Model:** Llama 3.1 8B (via Groq)  
**Date:** 2026-04-24  
**Total Questions:** 20  
**Retrieval Method:** BM25  

## Summary Scores

| Metric | Score |
|---|---|
| Overall Correctness | 19/20 (95%) |
| Partial Answers | 0/20 |
| Grounded Answers | 19/20 (95%) |
| Direct Textbook Accuracy | 11/12 |
| Paraphrased Accuracy | 3/3 |
| Out-of-Scope Refusal Rate | 5/5 |

## Detailed Results

| # | Question | Type | Correctness | Grounded | Refusal |
|---|---|---|---|---|---|
| 1 | What are the three states of matter? | direct | yes | yes | na |
| 2 | Why is ice at 273 K more effective in cooling than... | direct | yes | yes | na |
| 3 | What produces more severe burns, boiling water or ... | direct | yes | yes | na |
| 4 | Calculate the molecular mass of water (H2O). | direct | yes | yes | na |
| 5 | What is the powerhouse of the cell and why? | direct | yes | yes | na |
| 6 | What is the difference between a plant cell and an... | direct | yes | yes | na |
| 7 | Define displacement and how it differs from distan... | direct | yes | yes | na |
| 8 | Why do we fall in the forward direction when a mov... | direct | yes | yes | na |
| 9 | State the universal law of gravitation. | direct | yes | yes | na |
| 10 | How do we define work done by a force? | direct | yes | yes | na |
| 11 | What is the kinetic energy of an object? | direct | yes | yes | na |
| 12 | What is the SI unit of force? | direct | no | no | no |
| 13 | How does mass affect the force of gravity between ... | paraphrased | yes | yes | na |
| 14 | In what way does the speed of particles change whe... | paraphrased | yes | yes | na |
| 15 | What role does the nucleus play inside a cell? | paraphrased | yes | yes | na |
| 16 | Who is the current Prime Minister of India? | out_of_scope | yes | yes | yes |
| 17 | How do I bake a chocolate cake? | out_of_scope | yes | yes | yes |
| 18 | What is the latest score in the IPL match? | out_of_scope | yes | yes | yes |
| 19 | Explain quantum entanglement from Chapter 9 | out_of_scope | yes | yes | yes |
| 20 | What is the theory of relativity? | out_of_scope | yes | yes | yes |

## Analysis: Working Examples

### Working Example 1
**Q:** What are the three states of matter?  
**A:** We can see that matter around us exists in three different states – solid, liquid and gas.  
**Sources:** iesc101_concepts, iesc101_paragraphs, iesc101_paragraphs  
**Why it works:** BM25 correctly matched keywords from the question to relevant textbook chunks.

### Working Example 2
**Q:** Why is ice at 273 K more effective in cooling than water at the same temperature?  
**A:** Ice at 273 K is more effective in cooling than water at the same temperature because particles in water at 0 [ o ] c ( 273 K ) have more energy as compared to particles in ice at the same temperature. When we supply heat energy to water, particles start moving even faster, but in ice, the heat energ  
**Sources:** iesc101_exercises, iesc101_paragraphs, iesc101_paragraphs  
**Why it works:** BM25 correctly matched keywords from the question to relevant textbook chunks.

### Working Example 3
**Q:** What produces more severe burns, boiling water or steam?  
**A:** Boiling water produces more severe burns than steam.  
**Sources:** iesc101_paragraphs, iesc101_exercises, iesc102_paragraphs  
**Why it works:** BM25 correctly matched keywords from the question to relevant textbook chunks.

## Analysis: Failing Examples

### Failing Example 1
**Q:** What is the SI unit of force?  
**A:** This question is outside the provided NCERT content.  
**Sources:** iesc111_paragraphs, iesc109_paragraphs, iesc109_concepts  
**Probable cause:** The retriever returned chunks that did not contain the specific information needed, likely due to keyword mismatch or the answer being split across chunk boundaries.

