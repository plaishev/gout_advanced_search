from googletrans import Translator
from utils import read_json_to_dict
import os
import sys
import json

translator = Translator()

datapath = "./data/restaurants_data_v2"
files = os.listdir(datapath)


for file in files:
    review_dict_list = read_json_to_dict(os.path.join(datapath, file))
    for review in review_dict_list:
        review_text = review.get("review")
        if "(Translated by Google)" in review_text:
            trans_by_google_idx = review_text.find("(Translated by Google)")
            cut_review_text = review_text[(trans_by_google_idx + 22) :]
            while "(Original)" in review_text:
                original_idx = review_text.find("(Original)")
                cut_review_text = cut_review_text[:original_idx]
                review["review"] = cut_review_text
        else:
            review_text_translated = translator.translate(review_text).text
            review["review"] = review_text_translated
    json_file = json.dumps(review_dict_list, ensure_ascii=False).encode("utf8")
    open(os.path.join("./data/restaurants_data_trans", file), "w").write(
        json_file.decode()
    )
