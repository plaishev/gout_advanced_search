import numpy as np
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cdist
from scipy.spatial.distance import cdist


model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)  # loading model to encode query

embeddings = np.load(
    "./embeddings/embeddings.npy"
)  # rows are reviews, cols are feature coordinates
place_ids = np.load("./embeddings/place_ids.npy")  # ids of each review

query = "vibrant expensive cocktail bar"


query_encoded = np.expand_dims(model.encode(query), 0)

distances = cdist(query_encoded, embeddings, metric="euclidean")

most_similar_idx = np.argmin(distances)
top_n_idx = np.squeeze(np.argpartition(distances, 10))[:10]
unique_vals_and_occurrences = np.unique(place_ids[top_n_idx], return_counts=True)

top_places_dict = {
    k: v for k, v in zip(unique_vals_and_occurrences[0], unique_vals_and_occurrences[1])
}

print({k: v for k, v in sorted(top_places_dict.items(), key=lambda item: -item[1])})
