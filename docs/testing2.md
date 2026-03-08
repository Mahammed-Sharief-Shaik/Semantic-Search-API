### Semantic Cache Threshold Selection

A similarity threshold of **0.70** was chosen after empirical evaluation.

During testing, semantically related queries such as:

* "computer graphics rendering"
* "3D rendering techniques"

produced cosine similarity values around **0.68**.

Although these queries belong to the same domain, they retrieved slightly different document rankings. Reusing cached results in such cases could reduce retrieval accuracy.

Therefore, a threshold of **0.70** was selected to ensure that cached results are reused only when queries are strongly semantically aligned.

This conservative threshold improves the reliability of the semantic cache while still allowing reuse for near-identical queries.
