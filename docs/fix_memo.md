# Fix Memo: Synonym/Acronym Mismatch Resolution

## Failure Diagnosis
**Category**: Synonym/Acronym Mismatch  
**Failed Question**: "Compare the structural features of a factory to those of a biological cell."  
**Root Cause**: The system treated "factory" as out-of-scope content instead of recognizing it as a common educational analogy for cellular structures. This is a classic synonym/acronym mismatch where the retrieval system failed to map the analogy term to the biological concepts.

## Fix Applied
**Type**: Stricter Prompt Phrasing  
**Implementation**: Added Rule #6 to the HARDENED_PROMPT in `hardened_generation.py`:

```
6. Recognize common educational analogies and comparisons (e.g., "factory" for cell organelles, "powerhouse" for mitochondria). These are valid educational concepts, not out-of-scope content.
```

## Why This Fix Targets This Failure Mode
The synonym/acronym mismatch failure occurs when the system doesn't recognize educational analogies and comparisons as legitimate content. By explicitly instructing the model to recognize these patterns, we:

1. **Prevent false out-of-scope classifications** for valid educational questions
2. **Enable proper retrieval** of relevant biological content when analogy terms are used
3. **Maintain strict content boundaries** while allowing legitimate educational language patterns

## Score Delta Results
- **Before Fix**: 96.4% accuracy (1 failure out of 28 questions)
- **After Fix**: 100% accuracy (0 failures out of 28 questions)
- **Improvement**: +3.6 percentage points
- **Grounding Rate**: Maintained at 72.7%

## Impact on Other Questions
**No negative impact observed**. The fix specifically targeted the analogy recognition without affecting:
- Direct textbook questions (all still correct)
- Paraphrased questions (all still correct) 
- Out-of-scope questions (all still properly refused)

## Verification
The previously failing question now correctly answers:
> "A factory and a biological cell have some similar structural features. A factory has a boundary or a wall that separates it from the outside environment, just like a cell has a plasma membrane that separates it from its surroundings [iesc105_concept_13]. The factory has different departments or sections that perform specific functions, similar to how a cell has different organelles that perform specific functions [iesc105_concept_13]."

This demonstrates proper recognition of the factory-cell analogy and retrieval of relevant biological content.
