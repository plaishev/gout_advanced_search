import json
import pandas
import os


def read_json_to_dict(path: str) -> dict:
    with open(path, encoding='utf-8') as json_file: 
        res_dict = json.load(json_file)
    return res_dict


def save_df_cyrillic(df: pandas.core.frame.DataFrame, path: str) -> None:
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    df.to_csv(path, encoding='utf-8-sig')