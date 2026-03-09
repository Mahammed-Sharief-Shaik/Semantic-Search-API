# from dataset_loader import extract_dataset, load_documents

# dataset_folder = extract_dataset("mini_newsgroups.tar.gz")

# documents, labels = load_documents(dataset_folder)

# print("Documents loaded:", len(documents))
# print("First document preview:")
# print(documents[0])


# from dataset_loader import extract_dataset, load_documents

# dataset_folder = extract_dataset("mini_newsgroups.tar.gz")

# documents, labels = load_documents(dataset_folder)

# print("Documents loaded:", len(documents))
# print()
# print("Cleaned document preview:")
# # print(documents[0][:500])
# print(documents[20])

from dataset_loader import extract_dataset, load_documents
from embeddings import EmbeddingEngine
from clustering import FuzzyClusterer
from semantic_cache import SemanticCache

def main():

    print("Loading dataset...")

    dataset_folder = extract_dataset("20_newsgroups.tar.gz")

    documents, labels = load_documents(dataset_folder)

    print("Documents loaded:", len(documents))


    # -------------------------
    # Create embedding engine
    # -------------------------

    engine = EmbeddingEngine()


    # -------------------------
    # Generate embeddings
    # -------------------------

    embeddings = engine.generate_embeddings(documents)

    print("Embeddings shape:", embeddings.shape)
    # -------------------------
    # Build vector index
    # -------------------------

    # engine.build_faiss_index(embeddings, documents)
    engine.build_faiss_index(embeddings, documents, labels)

    # -------------------------
    # Test search
    # -------------------------

    # Initialize cache
    cache = SemanticCache(threshold=0.70)

    # Initialize clustering
    clusterer = FuzzyClusterer()
    clusterer.find_optimal_clusters(embeddings)

    cluster_probs = clusterer.get_cluster_probabilities(embeddings)

    print("\nSystem Ready!\n")

    while True:

        query = input("Enter query (or type 'exit'): ")

        if query.lower() == "exit":
            break

        # ---------------------------
        # Generate query embedding
        # ---------------------------

        # query_embedding = engine.model.encode([query])[0]
        query_embedding = engine.model.encode(
            [query],
            normalize_embeddings=True
        )[0]

        # ---------------------------
        # Predict cluster
        # ---------------------------

        cluster_id, probs = clusterer.predict_cluster(query_embedding)

        # ---------------------------
        # Check semantic cache
        # ---------------------------

        cache_result = cache.lookup(query_embedding, cluster_id)
        results = []

        if cache_result["cache_hit"]:

            print("\nCACHE HIT")
            print("Matched query:", cache_result["matched_query"])
            print("Similarity:", cache_result["similarity"])

            results = cache_result["result"]

        else:

            print("\nCACHE MISS → running semantic search")

            results = engine.search(query)

            cache.store(query, query_embedding, results, cluster_id)

        # ---------------------------
        # Display results
        # ---------------------------

        print("\nTop Results:\n")

        for i, result in enumerate(results):

            print(f"Result {i+1}")
            print("Category:", result["label"])
            print(result["document"][:250])
            print("-" * 50)

        print("\nCache Stats:", cache.stats())




if __name__ == "__main__":
    main()


