import json

import numpy as np
from sentence_transformers import SentenceTransformer


def store_embeddings(
    sentences: list[str], emb_model: SentenceTransformer, filename: str
):
    """
    Stores embeddings for the given sentences in a JSON file.

    :param sentences: A list of sentences.
    :param emb_model: The embedding model.
    :param filename: The path to the JSON file.
    """
    if not sentences:
        print("Warning: The sentences list is empty. No embeddings will be stored.")
        return

    try:
        # Convert sentences to embeddings
        embeddings = emb_model.encode(sentences)
    except Exception as e:
        print(f"Error during encoding: {e}")
        return

    # Create a dictionary with sentences as keys and embeddings as values
    embeddings_dict = {}
    for i in range(len(sentences)):
        try:
            embeddings_dict[sentences[i]] = embeddings[i].tolist()
        except Exception as e:
            print(f"Error processing sentence '{sentences[i]}': {e}")

    try:
        # Save the dictionary to a JSON file
        with open(filename, "w") as f:
            json.dump(embeddings_dict, f)
    except OSError as e:
        print(f"Error saving embeddings to file {filename}: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")


def load_embeddings(filename: str):
    """
    Loads embeddings from a JSON file.

    :param filename: The path to the JSON file.
    :return: A dictionary with sentences as keys and embeddings as values.
    """
    with open(filename) as f:
        embeddings_dict = json.load(f)
    embeddings_dict = {k: np.array(v) for k, v in embeddings_dict.items()}
    return embeddings_dict
