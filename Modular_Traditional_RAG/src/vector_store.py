"""# Vector Store Module for Modular Traditional RAG"""
"""### Vector storage and retrieval module for the Modular Traditional RAG system."""

# Import required Libraries
import os
import faiss
import numpy as np
import pickle
from typing import List, Any
from sentence_transformers import SentenceTransformer
from src.embedding import EmbeddingPipeline

# Define a simple vector store class with Faiss