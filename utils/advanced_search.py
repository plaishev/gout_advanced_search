from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import seuclidean
from torch.utils.data import DataLoader
from torch_dataset import ReviewsDataset

dataset = ReviewsDataset("./data/data_translated_combined/all_reviews.json")
sentences_dataloader = DataLoader(dataset, batch_size=256, shuffle=True)
reviews = next(iter(sentences_dataloader))

model = SentenceTransformer("sentence-transformers/msmarco-distilbert-base-v4")

sentences = reviews.get("review")

embeddings = {
    "encoding": model.encode(sentences),
    "place_id": reviews.get("place_id"),
    "place_name": reviews.get("place_name"),
}


def find(query: int) -> list[int]:
    def getdist(emb):
        return emb.get("distance")

    distances = []
    for i in range(len(embeddings.get("place_id"))):
        distances.append(
            {
                "distance": seuclidean(
                    model.encode(query),
                    embeddings.get("encoding")[i],
                    [1] * len(embeddings.get("encoding")[i]),
                ),
                "place": embeddings.get("place_name")[i],
            }
        )

    distances.sort(key=getdist)

    places = []
    for item in distances:
        places.append(item.get("place"))
    distances_num = []
    for item in distances:
        distances_num.append(item.get("distance"))
    scores = []
    completed_places = []
    for place in places:
        if place in completed_places:
            continue
        else:
            completed_places.append(place)
            indices = [i for i, x in enumerate(places) if x == place]
            score = 0
            for index in indices:
                score += 1 / distances_num[index]
        scores.append({"place": place, "score": score})

    def getscore(item):
        return item.get("score")

    scores.sort(key=getscore)
    return scores


print(find("sashimi"))
