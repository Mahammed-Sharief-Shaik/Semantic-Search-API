# Semantic Search System with Fuzzy Clustering and Semantic Caching

## Overview

This project implements a **semantic search engine** built on top of the **20 Newsgroups dataset**.
Instead of traditional keyword search, the system uses **sentence embeddings and vector similarity** to retrieve semantically relevant documents.

The system also includes:

* **Vector similarity search using FAISS**
* **Fuzzy clustering using Gaussian Mixture Models**
* **Semantic caching to accelerate repeated or similar queries**
* **FastAPI service layer for API-based interaction**

The goal of the project is to demonstrate how modern **AI-powered search systems** work internally using embeddings, clustering, and intelligent caching.

---

# System Architecture

The system follows a modular pipeline architecture.

```
User Query
   ↓
Query Embedding (Sentence Transformer)
   ↓
Cluster Prediction (Gaussian Mixture Model)
   ↓
Semantic Cache Lookup
   ↓
Cache Hit → Return Cached Results
   ↓
Cache Miss
   ↓
FAISS Vector Similarity Search
   ↓
Return Top Results
   ↓
Store Results in Cache
```

This architecture improves both **search accuracy and response performance**.

---

# Key Components

## 1. Dataset Processing

The system uses the **20 Newsgroups dataset**, which contains approximately **20,000 Usenet posts** categorized into **20 discussion groups**.

Examples of categories include:

* `comp.graphics`
* `sci.space`
* `rec.sport.baseball`
* `talk.politics.guns`
* `sci.electronics`

Each document is stored as a text file inside category folders.

### Preprocessing Steps

To improve semantic search quality, the following preprocessing steps are applied:

* Remove message headers such as routing paths and message IDs
* Remove quoted replies (`> previous messages`)
* Remove email signatures and ASCII art
* Retain meaningful message content and subject information
* Normalize whitespace and remove noise

The cleaned documents are then used for embedding generation.

---

# 2. Sentence Embeddings

The system converts documents into **dense vector representations** using the model:

```
sentence-transformers/all-MiniLM-L6-v2
```

This model generates **384-dimensional embeddings** that capture the semantic meaning of text.

Example:

```
"space shuttle mission"
      ↓
[0.12, -0.44, 0.81, ... 384 dimensions]
```

To improve similarity computation, embeddings are generated with:

```
normalize_embeddings=True
```

This ensures vectors lie on a **unit hypersphere**, allowing cosine similarity to be computed efficiently.

---

# 3. Vector Search using FAISS

The embeddings are indexed using **FAISS (Facebook AI Similarity Search)**.

FAISS enables **efficient nearest neighbor search** over large vector collections.

The project uses:

```
IndexFlatIP
```

Because with normalized vectors:

```
Inner Product ≈ Cosine Similarity
```

This allows accurate semantic similarity retrieval.

During a query:

1. Query text is converted into an embedding
2. FAISS retrieves the **top-k most similar documents**

---

# 4. Fuzzy Clustering (Gaussian Mixture Model)

To capture thematic structure in the dataset, the system performs **fuzzy clustering** using a **Gaussian Mixture Model (GMM)**.

Unlike hard clustering (e.g., K-Means), GMM provides **probabilistic cluster membership**.

Example:

```
Document → [Cluster 1: 0.65, Cluster 3: 0.25, Cluster 5: 0.10]
```

This allows documents to belong partially to multiple topics.

The system determines the **optimal number of clusters** using **BIC (Bayesian Information Criterion)**.

---

# 5. Semantic Cache

A semantic cache is implemented to avoid recomputing results for **similar queries**.

Instead of exact matching, the system compares **query embeddings**.

### Cache Logic

1. A new query is embedded.
2. Cosine similarity is computed against cached queries.
3. If similarity ≥ threshold (0.70), the cached result is reused.

Example:

```
Query 1: "computer graphics rendering"
Query 2: "3D rendering techniques"

Similarity: 0.685
```

Depending on the threshold, the cache may reuse the previous result.

This reduces:

* repeated FAISS searches
* redundant embedding computations
* system latency

---

# API Layer

The system exposes functionality through a **FastAPI service**.

FastAPI provides:

* automatic API documentation
* asynchronous request handling
* JSON-based interaction

---

# API Endpoints

## 1. Query Endpoint

```
POST /query
```

Runs semantic search with caching.

### Request

```json
{
  "query": "computer graphics rendering"
}
```

### Response

```json
{
  "cache_hit": false,
  "results": [
    {
      "label": "comp.graphics",
      "document": "Computer Graphics studies at the Technion..."
    }
  ]
}
```

If a similar query exists in cache:

```json
{
  "cache_hit": true,
  "matched_query": "computer graphics rendering",
  "similarity": 0.86,
  "results": [...]
}
```

---

## 2. Cache Statistics

```
GET /cache/stats
```

Returns:

* number of cached queries
* cache hit count
* cache miss count
* cache hit rate

Example response:

```json
{
  "total_entries": 10,
  "hit_count": 4,
  "miss_count": 6,
  "hit_rate": 0.40
}
```

---

## 3. Clear Cache

```
DELETE /cache
```

Clears all cached query results.

---

# Running the Project Locally

## 1. Clone the Repository

```
git clone https://github.com/Mahammed-Sharief-Shaik/Semantic-Search-API
cd Semantic-Search-API
```

---

## 2. Install Dependencies

```
pip install -r requirements.txt
```

---

## 3. Run the API Server

```
uvicorn app.api:app --reload
```

---

## 4. Access API Documentation

Open:

```
http://127.0.0.1:8000/docs
```

FastAPI automatically provides an interactive Swagger interface.

---

# Project Structure

```
semantic-search-system/
│
├── app/
│   ├── api.py
│   ├── dataset_loader.py
│   ├── embeddings.py
│   ├── clustering.py
│   ├── semantic_cache.py
│
├── data/
│   ├── mini_newsgroups.tar.gz
│   └── embeddings.npy
│
├── requirements.txt
├── Procfile
├── runtime.txt
├── README.md
└── .gitignore
```

---

# Performance Optimizations

The project includes several optimizations:

### Cached Embeddings

Generated embeddings are saved to:

```
data/embeddings.npy
```

This avoids recomputing embeddings on every server restart.

---

### Semantic Cache

Reduces repeated FAISS searches for similar queries.

---

### Threadpool Execution

Heavy operations run inside a **threadpool** to avoid blocking the FastAPI event loop.

---

# Example Queries

The system works best with semantic queries such as:

```
space shuttle mission
computer graphics rendering
baseball team statistics
gun control laws
medical treatment research
electronics circuit design
```

---

# Deployment

The API can be deployed using platforms such as:

* Render
* Railway
* Fly.io

Example start command:

```
uvicorn app.api:app --host 0.0.0.0 --port $PORT
```

---

# Future Improvements

Potential extensions include:

* Query expansion using LLMs
* Hybrid search (keyword + semantic)
* Advanced vector indexes (HNSW, IVF)
* Distributed vector storage
* Learning-to-rank for improved result ordering


