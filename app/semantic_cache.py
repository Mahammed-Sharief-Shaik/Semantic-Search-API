import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class SemanticCache:

    def __init__(self, threshold=0.85):

        self.threshold = threshold

        self.cache = []

        self.hit_count = 0
        self.miss_count = 0


    def lookup(self, query_embedding, cluster_id):
        """
        Check if a similar query exists in cache.
        """

        best_similarity = 0
        best_entry = {}

        for entry in self.cache:

            # Optional optimization: check same cluster
            # if entry["cluster"] != cluster_id:
            #     continue

            similarity = cosine_similarity(
                query_embedding.reshape(1, -1),
                entry["embedding"].reshape(1, -1)
            )[0][0]

            print(f"Comparing with '{entry['query']}' → similarity: {similarity:.3f}")

            if similarity > best_similarity:
                best_similarity = similarity
                best_entry = entry

        if best_similarity >= self.threshold:

            self.hit_count += 1

            return {
                "cache_hit": True,
                "similarity": float(best_similarity),
                "result": best_entry["result"],
                "matched_query": best_entry["query"]
            }

        self.miss_count += 1

        return {"cache_hit": False}


    def store(self, query, query_embedding, result, cluster_id):
        """
        Store query result in cache.
        """

        entry = {
            "query": query,
            "embedding": query_embedding,
            "result": result,
            "cluster": cluster_id
        }

        self.cache.append(entry)


    def stats(self):

        total = self.hit_count + self.miss_count

        hit_rate = self.hit_count / total if total > 0 else 0

        return {
            "total_entries": int(len(self.cache)),
            "hit_count": int(self.hit_count),
            "miss_count": int(self.miss_count),
            "hit_rate": float(hit_rate)
        }


    def clear(self):

        self.cache = []

        self.hit_count = 0
        self.miss_count = 0