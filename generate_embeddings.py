import argparse
import os

from utils.advanced_search_utils import init_datamodule, init_hf_transformer, generate_embeddings_hf
from utils.utils import save_array


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_path", type=str, help="path to data")
    parser.add_argument("--model_name", type=str, help="name of the Hugging Face transformer model")
    parser.add_argument("--destination_path", type=str, help="destination file for generated embeddings")

    return parser.parse_args()


def generate_and_save_embeddings(args):
    google_review_datamodule = init_datamodule(args.data_path)
    hf_model = init_hf_transformer(args.model_name)
    embeddings, place_ids = generate_embeddings_hf(hf_model, google_review_datamodule)
    os.makedirs(args.destination_path, exist_ok=True)
    save_array(embeddings, os.path.join(args.destination_path, 'embeddings.npy'))
    save_array(place_ids, os.path.join(args.destination_path, 'place_ids.npy'))
    # save_array(place_names, os.path.join(args.destination_path, 'place_names.npy'))


if __name__ == '__main__':
    args = parse_arguments()
    generate_and_save_embeddings(args)