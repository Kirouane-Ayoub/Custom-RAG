import torch
from models import reranker_model, reranker_tokenizer
from torch.nn.functional import softmax


def reranker(pairs: list[list[str]], threshold: float):
    """
    Reranks pairs based on a model's scores and a given probability threshold.

    Parameters:
    - pairs (list of list of str): List of [query, candidate] pairs.
    - threshold (float): Probability threshold for filtering the pairs.

    Returns:
    - filtered_answers (list of str): List of answers from the pairs that pass the threshold.
    - filtered_probabilities (torch.Tensor): Probabilities corresponding to the filtered answers.
    """
    if not pairs:
        print("Warning: The input pairs list is empty.")
        return [], []

    try:
        with torch.no_grad():
            # Tokenize the input pairs
            inputs = reranker_tokenizer(
                pairs,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=512,
            )
    except Exception as e:
        print(f"Error during tokenization: {e}")
        return [], []

    try:
        # Get the scores from the model
        scores = reranker_model(**inputs, return_dict=True).logits.view(-1).float()
    except Exception as e:
        print(f"Error getting scores from the model: {e}")
        return [], []

    try:
        # Apply softmax to get probabilities
        probabilities = softmax(scores, dim=0)
    except Exception as e:
        print(f"Error applying softmax: {e}")
        return [], []

    try:
        # Filter pairs based on the threshold
        high_prob_indices = probabilities > threshold
        filtered_probabilities = probabilities[high_prob_indices]
        # Extract only the answers and their corresponding scores
        filtered_answers = [
            pairs[i][1] for i in range(len(pairs)) if high_prob_indices[i]
        ]
    except Exception as e:
        print(f"Error during filtering or extraction: {e}")
        return [], []

    return filtered_answers, filtered_probabilities
