"""# Embedding Module for Modular Traditional RAG"""
"""### Chunking and embedding module for the Modular Traditional RAG system."""

# Import required Libraries
import os
import numpy as np
from typing import List, Any
#from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from src.document_loader import load_all_documents

# Embedding class to handle document chunking and embedding
import os
import numpy as np
from typing import List, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

class EmbeddingPipeline:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", chunk_size: int = 1000, chunk_overlap: int = 200):
        self.model_name = model_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.model = SentenceTransformer(model_name)
        print(f"[INFO] Initialized embedding model: {model_name}")

    def chunk_document(self, documents: List[Any]) -> List[Any]:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size, 
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = text_splitter.split_documents(documents)
        print(f"[INFO] Split {len(documents)} documents into {len(chunks)} chunks.")
        return chunks
    
    def generate_embeddings(self, chunks: List[Any]) -> np.ndarray:
        texts = [chunk.page_content for chunk in chunks]
        print(f"[INFO] Generating embeddings for {len(texts)} chunks using model: {self.model_name}")
        embeddings = self.model.encode(texts, show_progress_bar=True)
        print(f"[INFO] Generated embeddings for {len(chunks)} chunks.")
        print(f"[INFO] Embeddings shape: {embeddings.shape}")
        return embeddings