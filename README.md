# Custom-RAG: Build Your Own Question Answering System

This repository provides a hands-on implementation of a Retrieval Augmented Generation (RAG) system, built from scratch. 
**Learn More:** For a detailed explanation of the concepts and techniques used in this project, please refer to the accompanying blog post: [Building a Simple Question Answering Pipeline from Scratch](https://medium.com/@ayoubkirouane3/building-a-simple-question-answering-pipeline-from-scratch-e2d0da83412f)

## Getting Started

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Kirouane-Ayoub/Custom-RAG.git
   cd Custom-RAG
   ```

2. **Install the required packages:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare your data:**
   - Move your PDF documents into the `data` folder.

4. **Configure the settings:**
   - Open `src/settings.py` and modify the following parameters as needed:
     - `EMBEDDING_STORE_FILENAME`: Name of the file to store embeddings.
     - `REKANKER_MODEL_NAME`: Name of the reranker model.
     - `EMBEDDING_MODEL_NAME`: Name of the embedding model.
     - `LLM_MODEL_NAME`: Name of the Large Language Model (LLM) for answer generation.
     - `SIMILARITY_METRIC`: Similarity metric for search (cosine, Euclidean, Manhattan, dot product). Default is cosine.

5. **Run the pipeline:**
   ```bash
   python src/main.py
   ```

## Key Features and Customization

* **Similarity Metrics:** This pipeline supports various similarity metrics for search, including cosine, Euclidean, Manhattan, and dot product. You can easily switch between these metrics in `settings.py`. For a deeper understanding of these metrics, please refer to the blog post linked above.

* **Efficient LLM:** The default LLM used is `Qwen/Qwen2-1.5B-Instruct`, a fast and capable 1.5B parameter model. You can choose a larger model in `settings.py` if you require greater accuracy or complexity.

* **Chunking Control:** The chunking mechanism plays a crucial role in the effectiveness of the RAG system.  You can control the `CHUNK_SIZE` and `OVERLAP` parameters in `settings.py` or modify the `pdf_reader.py` file to fine-tune the chunking process for optimal performance.
