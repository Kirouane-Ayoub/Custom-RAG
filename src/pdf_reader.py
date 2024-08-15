import os
import re

from PyPDF2 import PdfReader


def regex_splitter(text: str, chunk_size: int, overlap: int):
    """
    split the input text into chunks of the specified size with optional overlap.

    :param text: The input text to be split.
    :param chunk_size: The size of each chunk.
    :param overlap: The amount of overlap between consecutive chunks.
    :return: A list of text chunks.
    """
    # Regular expression to match sentence endings (period, question mark, etc.)
    sentence_endings = re.compile(r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s")

    # Split the text into sentences based on sentence endings
    sentences = sentence_endings.split(text)

    # Initialize variables for chunking
    chunks = []
    current_chunk = ""

    # Iterate through each sentence and create chunks
    for sentence in sentences:
        # If adding the current sentence to the current chunk doesn't exceed the chunk size, add it
        if len(current_chunk) + len(sentence) + 1 <= chunk_size:
            current_chunk += " " + sentence
        # Otherwise, add the current chunk to the list and reset it
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence

    # Add the last chunk if it exists
    if current_chunk:
        chunks.append(current_chunk.strip())

    # If overlap is specified, create overlapping chunks
    if overlap > 0:
        overlapping_chunks = []
        for i in range(len(chunks)):
            # Calculate the start and end indices for the current overlapping chunk
            start = max(0, i * chunk_size - i * overlap)
            end = start + chunk_size
            # Add the overlapping chunk to the list
            overlapping_chunks.append(text[start:end].strip())
        return overlapping_chunks
    else:
        return chunks


def get_pdf_text(pdf_path):
    """
    Extracts text from a PDF file.

    :param pdf_path: The path to the PDF file.
    :return: The extracted text from the PDF.
    """
    total_text = ""
    for filename in os.listdir(pdf_path):
        if filename.lower().endswith(".pdf"):
            pdf_path = os.path.join("data", filename)
            with open(pdf_path, "rb") as file:
                reader = PdfReader(file)
                for page in reader.pages:
                    total_text += page.extract_text() or ""
    return total_text
