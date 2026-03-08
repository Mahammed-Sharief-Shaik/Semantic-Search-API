import numpy as np
from sklearn.mixture import GaussianMixture


class FuzzyClusterer:

    def __init__(self):
        self.model = None
        self.n_clusters = None


    def find_optimal_clusters(self, embeddings, cluster_range=[5, 8, 10, 12, 15, 20]):
        """
        Determine the optimal number of clusters using BIC.
        """

        print("\nFinding optimal number of clusters using BIC...\n")

        best_bic = float("inf")
        best_model = None
        best_k = None

        for k in cluster_range:

            print(f"Testing {k} clusters...")

            gmm = GaussianMixture(
                n_components=k,
                covariance_type="full",
                random_state=42
            )

            gmm.fit(embeddings)

            bic = gmm.bic(embeddings)

            print(f"BIC score: {bic}")

            if bic < best_bic:

                best_bic = bic
                best_model = gmm
                best_k = k


        self.model = best_model
        self.n_clusters = best_k

        print("\nOptimal cluster count:", best_k)

        return best_model


    def get_cluster_probabilities(self, embeddings):
        """
        Returns probability distribution over clusters for each document.
        """

        if self.model is None:
            raise ValueError("Model not trained yet")

        probabilities = self.model.predict_proba(embeddings)

        return probabilities
    
    def predict_cluster(self, embedding):
        """
        Predict dominant cluster for a given embedding.
        """

        probs = self.model.predict_proba(embedding.reshape(1, -1))

        cluster_id = probs.argmax()

        return cluster_id, probs