## Semantic Cache Similarity Threshold

The semantic cache determines whether a new query can reuse the result of a previously processed query.

Instead of relying on exact string matching, the system compares query embeddings using cosine similarity.

### Threshold Selection

The similarity threshold determines when two queries are considered semantically equivalent.

If the similarity between the new query embedding and a cached query embedding exceeds the threshold, the cached result is returned.

Choosing the threshold involves balancing two competing factors:

* **Precision** — ensuring cached results remain relevant
* **Cache hit rate** — maximizing reuse of previous computations

If the threshold is too low, unrelated queries may be incorrectly treated as similar, leading to incorrect cache hits.

If the threshold is too high, semantically equivalent queries may fail to reuse cached results, reducing the effectiveness of the cache.

Empirical studies of sentence embedding similarity show that paraphrased sentences typically have cosine similarity between **0.80 and 0.90**.

Based on this observation, a threshold of **0.85** was selected. This value allows the cache to capture semantically similar queries while minimizing the risk of returning unrelated results.
w