import os
import json


def read_json_to_dict(path: str) -> dict:
    with open(path, encoding="utf-8") as json_file:
        res_dict = json.load(json_file)
    return res_dict


list_of_files = os.listdir("./data/data_translated")
placeid = 0
all_reviews = []
review_id = 0


for file in list_of_files:
    placeid += 1
    reviews = read_json_to_dict(os.path.join("./data/data_translated", file))
    for review in reviews:
        review_id += 1
        review.update(
            {"place_id": placeid, "review_id": review_id, "place_name": file[:-5]}
        )
        all_reviews.append(review)

json_file = json.dumps(all_reviews, ensure_ascii=False).encode("utf-8")
open(os.path.join("./data/data_translated_combined", "all_reviews.json"), "w").write(
    json_file.decode()
)
