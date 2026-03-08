from sentence_transformers import SentenceTransformer
import numpy as np
import faiss


class EmbeddingEngine:

    def __init__(self, model_name="all-MiniLM-L6-v2"):
        """
        Initialize the embedding model.
        """

        print("Loading embedding model...")
        self.model = SentenceTransformer(model_name)

        self.index = None
        self.documents = None

    def generate_embeddings(self, documents):
        """
        Convert documents into embedding vectors.
        """

        print("Generating embeddings...")

        embeddings = self.model.encode(
            documents,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        return embeddings
    
    def build_faiss_index(self, embeddings, documents, labels):

        dimension = embeddings.shape[1]

        print("Embedding dimension:", dimension)

        self.index = faiss.IndexFlatL2(dimension)

        self.index.add(embeddings)

        self.documents = documents
        self.labels = labels

        print("FAISS index built with", len(documents), "documents.")

    # def build_faiss_index(self, embeddings, documents):
    #     """
    #     Build FAISS vector index for similarity search.
    #     """

    #     dimension = embeddings.shape[1]

    #     print("Embedding dimension:", dimension)

    #     self.index = faiss.IndexFlatL2(dimension)

    #     self.index.add(embeddings)

    #     self.documents = documents

    #     print("FAISS index built with", len(documents), "documents.")

    # def search(self, query, k=5):
    #     """
    #     Perform semantic search.
    #     """

    #     query_embedding = self.model.encode([query])

    #     distances, indices = self.index.search(query_embedding, k)

    #     results = []

    #     for idx in indices[0]:
    #         results.append(self.documents[idx])

    #     return results
    def search(self, query, k=5):

        query_embedding = self.model.encode([query])

        distances, indices = self.index.search(query_embedding, k)

        results = []

        for idx in indices[0]:

            result = {
                "document": self.documents[idx],
                "label": self.labels[idx]
            }

            results.append(result)

        return results