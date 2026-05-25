"""# Embedding Module for Modular Traditional RAG"""
"""### Chunking and embedding module for the Modular Traditional RAG system."""

# Import required Libraries
import os
import numpy as np
from typing import List, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from src.data_loader import load_all_documents

# Embedding class to handle document chunking and embedding
class EmbeddingPipeline:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize the embedding pipeline with a specified model and chunking parameters.

        Args:
            model_name (str): The name of the sentence transformer model to use for embeddings.
            chunk_size (int): The size of each text chunk.
            chunk_overlap (int): The number of overlapping characters between chunks.
        """
        self.model_name = model_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.model = SentenceTransformer(model_name)
        print(f"[INFO] Initialized embedding model: {model_name}")

    # Method to split and chunk documents
    def chunk_document(self, documents: List[Any]) -> List[str]:
        """
        Chunk a document into smaller pieces based on the specified chunk size and overlap.

        Args:
            document (str): The input document to be chunked.

        Returns:
            List[str]: A list of chunked text pieces.
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size, 
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            seperators=["\n\n", "\n", " ", ""]
        )
        chunks = text_splitter.split_documents(documents)
        print(f"[INFO] Split {len(documents)} document into {len(chunks)} chunks.")
        return chunks
    
    # Method to generate embeddings for the chunked documents
    def generate_embeddings(self, chunks: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of text chunks.

        Args:
            chunks (List[str]): A list of text chunks to be embedded.
        Returns:
            np.ndarray: An array of embeddings corresponding to the input chunks.
        """
        texts = [chunk.page_content for chunk in chunks]
        print(f"[INFO] Generating embeddings for {len(texts)} chunks using model: {self.model_name}")
        embeddings = self.model.encode(texts, show_progress_bar=True)
        print(f"[INFO] Generated embeddings for {len(chunks)} chunks.")
        print(f"[INFO] Embeddings shape: {embeddings.shape}")
        return embeddings