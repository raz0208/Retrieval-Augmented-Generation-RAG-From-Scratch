# Import required libraries
from src.document_loader import load_all_documents
from src.embedding import EmbeddingPipeline

# Implement the main function to run the RAG pipeline application
if __name__ == "__main__":
    print("[INFO] Starting document loading process...")
    documents = load_all_documents("data")
    print(f"[INFO] Finished loading documents. Total documents loaded: {len(documents)}")

    print("\n[INFO] Sample loaded documents:")
    for i, doc in enumerate(documents[:10]):
        print(f"\nDocument {i+1}:")
        print(f"Content: {doc.page_content[:200]}...")  # Print first 200 characters
        print(f"Metadata: {doc.metadata}")
    
    # Implement Chunking and Embedding
    print("\n[INFO] Initializing chunking documents and embedding pipeline...")
    chunks = EmbeddingPipeline().chunk_document(documents)
    print("[INFO] Generating embeddings for chunks...")
    embeddingsVectors = EmbeddingPipeline().generate_embeddings(chunks)
    print(f"[INFO] Generated embedding vectors for {len(chunks)} chunks.")
    print(embeddingsVectors)