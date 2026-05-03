# Retrieval Misses Diagnosis

1. **Question:** What is the SI unit of force?
   - **Top-1 Chunk ID:** `iesc111_concept_3`
   - **Wrong Chunk Content:** "...the unit of work is newton metre (N m) or joule (J). Thus 1 J = 1 N m = 1 kg m2 s -2. Work done is also defined as the product of component of force..."
   - **Diagnosis:** Bad retrieval ranking (lexical overlap). BM25 scored this highly because it contains "unit", "force", and "newton", but the chunk actually discusses the unit of work, not the unit of force.

2. **Question:** Calculate the molecular mass of water (H2O).
   - **Top-1 Chunk ID:** `iesc103_concept_8`
   - **Wrong Chunk Content:** "The mass of one mole of a substance in grams is called its molar mass. The molar mass of water is 18 g."
   - **Diagnosis:** Embedding limitation. The dense retrieval found the molar mass of water, but the question explicitly asked to "calculate" the molecular mass, which requires a chunk showing the steps (H = 1u, O = 16u, 2*1 + 16 = 18u).

3. **Question:** What produces more severe burns, boiling water or steam?
   - **Top-1 Chunk ID:** `iesc101_concept_12`
   - **Wrong Chunk Content:** "Water boils at 100°C (373 K). Steam is water in the gaseous state."
   - **Diagnosis:** Chunking miss. The explanation about latent heat of vaporization is in a different chunk or split from the fact about steam and boiling water.
