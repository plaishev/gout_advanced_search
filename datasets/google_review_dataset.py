from torch.utils.data import Dataset
import json


def read_json_to_dict(path: str) -> dict:
    with open(path, encoding="utf-8") as json_file:
        res_dict = json.load(json_file)
    return res_dict


class GoogleReviewsDataset(Dataset):
    def __init__(self, path_to_reviews, transform=None):
        self.reviews = path_to_reviews
        self.all_reviews = read_json_to_dict(f"{self.reviews}")
        self.transform = transform

    def __len__(self):
        return len(self.all_reviews)

    def __getitem__(self, reviewid):
        curr_review = self.all_reviews[reviewid]
        return curr_review['review'], curr_review['place_id'], curr_review['place_name']
