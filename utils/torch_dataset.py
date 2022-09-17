from numpy import place
from torch.utils.data import Dataset
import json


def read_json_to_dict(path: str) -> dict:
    with open(path, encoding="utf-8") as json_file:
        res_dict = json.load(json_file)
    return res_dict


class ReviewsDataset(Dataset):
    def __init__(self, path_to_reviews, transform):
        self.reviews = path_to_reviews
        self.transform = transform

    def __len__(self):
        return 64424

    def __getitem__(self, reviewid):
        all_reviews = read_json_to_dict(f"{self.reviews}")[reviewid]
        return all_reviews
