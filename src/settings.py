import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EMBEDDING_STORE_FILENAME = "embeddings/embeddings.json"
REKANKER_MODEL_NAME = "BAAI/bge-reranker-v2-m3"
EMBEDDING_MODEL_NAME = "Alibaba-NLP/gte-base-en-v1.5"
LLM_MODEL_NAME = "Qwen/Qwen2-1.5B-Instruct"


DATA_FOLDER = "data"
CHUNK_SIZE = 1024
OVERLAP = 50

SIMILARITY_METRIC = "cosine"
RERANKER_THRESHOLD = 0.5
SIMILARITY_THRESHOLD = 0.5
TOP_K = 5
