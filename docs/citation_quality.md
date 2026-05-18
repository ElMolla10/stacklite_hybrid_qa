# Citation Quality Examples

## Good Answer Patterns

1. **Question:** What is the difference between `fit` and `fit_transform` in scikit-learn?
   **Good answer:** Explains that `fit` learns parameters from data, while `fit_transform` learns those parameters and immediately applies the transformation. Cites `[datascience:12321]`.
   **Why it is good:** The citation directly matches the target concept and the answer stays inside the retrieved source.

2. **Question:** Why do transformers need positional encodings?
   **Good answer:** States that self-attention does not inherently encode token order, so positional information is added to represent sequence order. Cites `[datascience:51065]`.
   **Why it is good:** The cited source is specifically about positional encoding in transformers.

3. **Question:** What is self-supervised learning?
   **Good answer:** Describes learning from automatically generated labels or prediction tasks derived from the data itself. Cites `[ai:10623]`.
   **Why it is good:** The answer uses the relevant AI Stack Exchange question rather than a generic ML source.

4. **Question:** What does temperature do in GPT models?
   **Good answer:** Explains temperature as a decoding parameter that changes output randomness, with lower values making outputs more deterministic. Cites `[ai:32477]`.
   **Why it is good:** The answer uses the GPT temperature source and avoids unsupported product claims.

5. **Question:** When should I use macro average instead of micro average?
   **Good answer:** Explains that macro averaging gives equal weight to classes, which is useful when minority classes matter, while micro averaging aggregates individual decisions and can be dominated by majority classes. Cites `[datascience:15989]`.
   **Why it is good:** The citation is the exact multiclass metric discussion.

## Bad Answer Patterns

1. **Uncited claim:** "Macro average is always better."
   **Problem:** Overstates the source and gives no citation.

2. **Wrong source:** Answering a Keras class-weight question with a generic "balanced dataset" citation.
   **Problem:** The citation is topically close but not evidence for implementation details.

3. **Citation stuffing:** Adding five citations after a sentence that only came from one passage.
   **Problem:** Makes verification harder and hides which source supports which claim.

4. **Hallucinated API detail:** Inventing exact parameter names not shown in retrieved passages.
   **Problem:** RAG should stay grounded in the retrieved documents unless the source supports the detail.

5. **No uncertainty boundary:** Answering a broad current-events or product-version question from old StackLite posts.
   **Problem:** The corpus is historical Stack Exchange content and may not reflect current software behavior.
