## Embedding Model and Vector Search Design Decisions

### Choice of Embedding Model

For semantic representation of documents, the **SentenceTransformers model `all-MiniLM-L6-v2`** was selected.

This model was chosen because it provides an effective balance between semantic quality, computational efficiency, and memory usage.

Key characteristics of the model:

* Produces **384-dimensional embeddings**
* Lightweight (~90MB model size)
* Optimized for **sentence similarity tasks**
* Fast inference on CPU

The model is trained on large-scale sentence similarity datasets, which enables it to map semantically related sentences close together in vector space. This makes it particularly suitable for **semantic search and clustering applications**.

---

### Embedding Dimensionality

The selected model produces **384-dimensional vectors**.

Higher-dimensional embeddings (e.g., 768 or 1024) can capture additional information but significantly increase:

* memory usage
* vector index size
* similarity search latency

For medium-sized corpora such as the 20 Newsgroups dataset (~20k documents), 384 dimensions provide a strong balance between semantic expressiveness and computational efficiency.

---

### Choice of Vector Search Engine

To perform similarity search over document embeddings, **FAISS (Facebook AI Similarity Search)** was used.

FAISS is a highly optimized library designed for efficient nearest-neighbor search over high-dimensional vectors.

The reasons for selecting FAISS include:

* Extremely fast similarity search
* Lightweight local deployment
* No external infrastructure required
* Widely used in modern AI retrieval systems

---

### Index Selection

The FAISS index type **`IndexFlatL2`** was selected.

This index performs **exact nearest-neighbor search** using Euclidean distance.

Since the dataset size is relatively small (~20,000 documents), exact search is computationally efficient while ensuring high retrieval accuracy.

For very large datasets (millions of documents), approximate search indices such as IVF or HNSW may be preferred to reduce latency and memory usage.

For the scope of this assignment, `IndexFlatL2` provides the most straightforward and reliable solution.
