# Evaluation Results

**Model:** Llama 3.1 8B (via Groq)  
**Date:** 2026-05-03  
**Total Questions:** 22  
**Retrieval Method:** BM25  

## Summary Scores

| Metric | Score |
|---|---|
| Overall Correctness | 17/22 (77%) |
| Partial Answers | 0/22 |
| Grounded Answers | 17/22 (77%) |
| Direct Textbook Accuracy | 10/12 |
| Paraphrased Accuracy | 2/5 |
| Out-of-Scope Refusal Rate | 5/5 |

## Detailed Results

| # | Question | Type | Correctness | Grounded | Refusal |
|---|---|---|---|---|---|
| 1 | What are the three states of matter? | direct | yes | yes | na |
| 2 | Why is ice at 273 K more effective in cooling than... | direct | no | no | no |
| 3 | What produces more severe burns, boiling water or ... | direct | yes | yes | na |
| 4 | Calculate the molecular mass of water (H2O). | direct | yes | yes | na |
| 5 | What is the powerhouse of the cell and why? | direct | yes | yes | na |
| 6 | What is the difference between a plant cell and an... | direct | yes | yes | na |
| 7 | Define displacement and how it differs from distan... | direct | yes | yes | na |
| 8 | Why do we fall in the forward direction when a mov... | direct | yes | yes | na |
| 9 | State the universal law of gravitation. | direct | no | no | no |
| 10 | How do we define work done by a force? | direct | yes | yes | na |
| 11 | What is the kinetic energy of an object? | direct | yes | yes | na |
| 12 | What is the SI unit of force? | direct | yes | yes | na |
| 13 | How does mass affect the force of gravity between ... | paraphrased | yes | yes | na |
| 14 | In what way does the speed of particles change whe... | paraphrased | no | no | no |
| 15 | What role does the nucleus play inside a cell? | paraphrased | yes | yes | na |
| 16 | Describe the visual evidence that suggests particl... | paraphrased | no | no | no |
| 17 | Compare the structural features of a factory to th... | paraphrased | no | no | no |
| 18 | Who is the current Prime Minister of India? | out_of_scope | yes | yes | yes |
| 19 | How do I bake a chocolate cake? | out_of_scope | yes | yes | yes |
| 20 | What is the latest score in the IPL match? | out_of_scope | yes | yes | yes |
| 21 | Explain quantum entanglement from Chapter 9 | out_of_scope | yes | yes | yes |
| 22 | What is the theory of relativity? | out_of_scope | yes | yes | yes |

## Analysis: Working Examples

### Working Example 1
**Q:** What are the three states of matter?  
**A:** The three states of matter are:

1. Solid
2. Liquid
3. Gas  
**Sources:** iesc101_paragraphs, iesc101_paragraphs, iesc101_paragraphs  
**Why it works:** BM25 correctly matched keywords from the question to relevant textbook chunks.

### Working Example 2
**Q:** What produces more severe burns, boiling water or steam?  
**A:** Steam produces more severe burns than boiling water.  
**Sources:** iesc101_paragraphs, iesc101_paragraphs, iesc101_paragraphs  
**Why it works:** BM25 correctly matched keywords from the question to relevant textbook chunks.

### Working Example 3
**Q:** Calculate the molecular mass of water (H2O).  
**A:** Atomic mass of hydrogen = 1u
oxygen = 16 u

So the molecular mass of water, which contains two atoms of hydrogen and one atom of oxygen is = 2 ´ 1 + 1´16 = 18 u  
**Sources:** iesc103_paragraphs, iesc103_paragraphs, iesc103_paragraphs  
**Why it works:** BM25 correctly matched keywords from the question to relevant textbook chunks.

## Analysis: Failing Examples

### Failing Example 1
**Q:** Why is ice at 273 K more effective in cooling than water at the same temperature?  
**A:** This question is outside the provided NCERT content.  
**Sources:** iesc101_paragraphs, iesc101_paragraphs, iesc101_paragraphs  
**Probable cause:** The retriever returned chunks that did not contain the specific information needed, likely due to keyword mismatch or the answer being split across chunk boundaries.

### Failing Example 2
**Q:** State the universal law of gravitation.  
**A:** The universal law of gravitation is not explicitly stated in the provided context. However, it is mentioned that the universal law of gravitation successfully explained several phenomena, but the actual law is not provided.

This question is outside the provided NCERT content.  
**Sources:** iesc109_paragraphs, iesc109_paragraphs, iesc109_paragraphs  
**Probable cause:** The retriever returned chunks that did not contain the specific information needed, likely due to keyword mismatch or the answer being split across chunk boundaries.

