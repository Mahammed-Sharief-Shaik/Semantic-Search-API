from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
import os
import numpy as np
from app.dataset_loader import extract_dataset, load_documents
from app.embeddings import EmbeddingEngine
from app.clustering import FuzzyClusterer
from app.semantic_cache import SemanticCache
from contextlib import asynccontextmanager
import threading

engine = None
clusterer = None
cache = None


@asynccontextmanager
async def lifespan(app: FastAPI):

    print("Starting API...")

    threading.Thread(target=initialize_system).start()

    yield

    print("Shutting down API...")


app = FastAPI(
    title="Semantic Search API",
    lifespan=lifespan
)

# -------------------------
# Request Model
# -------------------------

class QueryRequest(BaseModel):
    query: str

def initialize_system():

    global engine, clusterer, cache

    try:

        print("Initializing system...")

        dataset_folder = extract_dataset("data/mini_newsgroups.tar.gz")
        documents, labels = load_documents(dataset_folder)

        engine = EmbeddingEngine()

        if os.path.exists("data/embeddings.npy"):
            print("Loading cached embeddings...")
            embeddings = np.load("data/embeddings.npy")
        else:
            print("Generating embeddings...")
            embeddings = engine.generate_embeddings(documents)
            os.makedirs("data", exist_ok=True)
            np.save("data/embeddings.npy", embeddings)

        engine.build_faiss_index(embeddings, documents, labels)

        clusterer = FuzzyClusterer()
        clusterer.find_optimal_clusters(embeddings)

        cache = SemanticCache(threshold=0.70)

        print("System ready.")

    except Exception as e:
        print("SYSTEM INITIALIZATION FAILED:", e)

# @app.on_event("startup")
# def startup():
#     # threading.Thread(target=initialize_system).start()
#     print("API started successfully")



# -------------------------
# Query Endpoint
# -------------------------
def process_query(query):


    # if engine is None:
    #     return {"message": "System still initializing. Please try again shortly."}
    if engine is None:
        initialize_system()

    # query_embedding = engine.model.encode([query])[0]
    query_embedding = engine.model.encode(
        [query],
        normalize_embeddings=True
    )[0]

    cluster_id, _ = clusterer.predict_cluster(query_embedding)

    cache_result = cache.lookup(query_embedding, cluster_id)

    if cache_result["cache_hit"]:

        return {
            "cache_hit": True,
            "matched_query": cache_result["matched_query"],
            "similarity": cache_result["similarity"],
            "results": cache_result["result"]
        }

    results = engine.search(query)

    cache.store(query, query_embedding, results, cluster_id)

    return {
        "cache_hit": False,
        "results": results
    }

@app.post("/query")
async def query_system(request: QueryRequest):

    return await run_in_threadpool(process_query, request.query)

# -------------------------
# Cache Stats Endpoint
# -------------------------

@app.get("/cache/stats")
def cache_stats():

    return cache.stats()


# -------------------------
# Clear Cache Endpoint
# -------------------------

@app.delete("/cache")
def clear_cache():

    cache.clear()

    return {"message": "Cache cleared"}