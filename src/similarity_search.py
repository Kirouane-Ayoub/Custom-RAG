import numpy as np
from scipy.spatial.distance import cityblock, cosine, euclidean
from sentence_transformers import SentenceTransformer


def find_similar_sentences(input_sentence:str, emb_model:SentenceTransformer, stored_embeddings:dict[str, np.array], metric:str='cosine', threshold:float=0.8, top_k:int=5):
    input_embedding = emb_model.encode([input_sentence])[0]

    # Define a function for each metric
    def calculate_similarity(embedding1, embedding2, metric):
        if metric == 'cosine':
            return 1 - cosine(embedding1, embedding2)  # Cosine similarity
        elif metric == 'euclidean':
            return 1 - (euclidean(embedding1, embedding2) / np.sqrt(len(embedding1)))  # Euclidean similarity
        elif metric == 'manhattan':
            return 1 - (cityblock(embedding1, embedding2) / np.sqrt(len(embedding1)))  # Manhattan similarity
        elif metric == 'dot':
            return np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))  # Dot product
        else:
            raise ValueError(f"Unknown metric: {metric}")

    similarities = {}
    try:
        for sentence, embedding in stored_embeddings.items():
            try:
                similarities[sentence] = calculate_similarity(input_embedding, embedding, metric)
            except Exception as e:
                print(f"Error calculating similarity for sentence '{sentence}': {e}")
                similarities[sentence] = -1  # Assign a default low value in case of an error
    except Exception as e:
        print(f"Error calculating similarities: {e}")
        return []

    # Filter sentences based on similarity threshold
    try:
        filtered_similarities = {sentence: similarity for sentence, similarity in similarities.items() if similarity >= threshold}
    except Exception as e:
        print(f"Error filtering similarities: {e}")
        return []

    # Sort sentences by similarity score in descending order
    try:
        sorted_sentences = sorted(filtered_similarities.items(), key=lambda item: item[1], reverse=True)
    except Exception as e:
        print(f"Error sorting sentences: {e}")
        return []

    # Select the top k sentences
    try:
        top_sentences = [sentence for sentence, _ in sorted_sentences[:top_k]]
    except Exception as e:
        print(f"Error selecting top {top_k} sentences: {e}")
        return []

    return top_sentences