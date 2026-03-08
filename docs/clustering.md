## Cluster Count Selection Strategy

The 20 Newsgroups dataset contains 20 labeled categories. However, these labels do not necessarily represent the true semantic structure of the corpus.

Many categories overlap conceptually. For example:

* `talk.politics.guns`, `talk.politics.misc`, and `talk.politics.mideast` all belong to the broader theme of politics.
* `rec.sport.baseball` and `rec.sport.hockey` are both sports-related discussions.

As a result, forcing the clustering algorithm to produce exactly 20 clusters would artificially impose the dataset labels rather than discovering the natural semantic structure.

Instead, an unsupervised approach was used to determine the optimal number of clusters.

### Cluster Selection Method

Multiple Gaussian Mixture Models were trained with different numbers of clusters:

```
5, 8, 10, 12, 15, 20
```

Each model was evaluated using the **Bayesian Information Criterion (BIC)**.

BIC balances two factors:

* goodness of fit
* model complexity

The model with the lowest BIC score was selected as the optimal clustering configuration.

### Choice of Clustering Algorithm

A **Gaussian Mixture Model (GMM)** was used because it produces **soft cluster assignments**.

Instead of assigning each document to a single cluster, GMM produces a probability distribution across clusters.

Example:

```
Document A
Cluster 3 → 0.72
Cluster 7 → 0.18
Cluster 1 → 0.10
```

This allows documents that discuss multiple topics to belong to multiple clusters with different degrees, which better reflects the structure of real-world discussions.

This approach aligns with the assignment requirement that documents may belong to multiple semantic topics simultaneously.
