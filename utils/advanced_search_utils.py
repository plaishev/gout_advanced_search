import numpy as np

from tqdm import tqdm

from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import seuclidean
from torch.utils.data import DataLoader
from datasets.google_review_dataset import GoogleReviewsDataset


def init_datamodule(dataset_path, batch_size=256):
    dataset = GoogleReviewsDataset(dataset_path)
    sentences_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return sentences_dataloader


def init_hf_transformer(model_name):
    """
    Initialize hugging face transformer
    """
    return SentenceTransformer(model_name)


def generate_embeddings_hf(model, dataloader):
    """
    Generate embeddings using Hugging Face Transformer model
    """
    embeddings = []
    place_ids = []
    # place_names = []

    for _, data in tqdm(enumerate(dataloader)):
        curr_reviews = data[0]
        curr_place_ids = data[1]
        # curr_place_names = data[2]
        curr_embeddings = model.encode(curr_reviews)
        
        embeddings.append(curr_embeddings)
        place_ids.append(curr_place_ids.numpy())
        # place_names.append(curr_place_names.numpy())

    embeddings = np.vstack(embeddings)
    place_ids = np.hstack(place_ids)
    # place_names = np.hstack(place_names)

    return embeddings, place_ids #, place_names
