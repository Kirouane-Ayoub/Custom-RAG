import settings
from embedding import load_embeddings, store_embeddings
from generator import pipe
from models import emb_model
from pdf_reader import get_pdf_text, regex_splitter
from reranker import reranker
from similarity_search import find_similar_sentences

total_text = get_pdf_text(settings.DATA_FOLDER)
chunks = regex_splitter(
    text=total_text, chunk_size=settings.CHUNK_SIZE, overlap=settings.OVERLAP
)

store_embeddings(chunks, emb_model, settings.EMBEDDING_STORE_FILENAME)
stored_embeddings = load_embeddings(settings.EMBEDDING_STORE_FILENAME)


def run(prompt: str):
    """
    Executes the full pipeline: similarity search, reranking, and text generation.

      Parameters:
      - prompt (str): The input query for which the response is generated.
    """
    # Simularity search
    similar_sentences = find_similar_sentences(
        prompt,
        emb_model,
        stored_embeddings,
        metric=settings.SIMILARITY_METRIC,
        threshold=settings.SIMILARITY_THRESHOLD,
        top_k=settings.TOP_K,
    )

    # Create Pairs
    pairs = [[prompt, sentence] for sentence in similar_sentences]

    # Reranker
    filtered_answers, _ = reranker(pairs, settings.RERANKER_THRESHOLD)

    # Generator
    context = "\n =======".join(filtered_answers)

    content = f"""
    Context information is below.\n
    ---------------------\n
    {context}\n"
    ---------------------\n
    Given the context information and not prior knowledge,
    answer the query.\n
    Query: {prompt}\n
    Answer:
    """
    print("Content : \n", content)
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": content},
    ]
    # Run the text generator
    pipe(messages)


while True:
    prompt = input("Enter your question : ")
    run(prompt)
