import settings
from sentence_transformers import SentenceTransformer
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

# Define the embedding model
emb_model = SentenceTransformer(settings.EMBEDDING_MODEL_NAME, trust_remote_code=True)


reranker_tokenizer = AutoTokenizer.from_pretrained(settings.REKANKER_MODEL_NAME)
reranker_model = AutoModelForSequenceClassification.from_pretrained(
    settings.REKANKER_MODEL_NAME
)
reranker_model.eval()


# Now you do not need to add "trust_remote_code=True"
llm_model = AutoModelForCausalLM.from_pretrained(
    settings.LLM_MODEL_NAME, device_map=settings.DEVICE
)
llm_tokenizer = AutoTokenizer.from_pretrained(settings.LLM_MODEL_NAME)
