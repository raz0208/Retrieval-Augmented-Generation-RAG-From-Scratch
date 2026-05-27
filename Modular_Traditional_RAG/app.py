from src.document_loader import load_all_documents
from src.embedding import EmbeddingPipeline
from src.vector_store import FaissVectorStore

if __name__ == "__main__":
    print("[INFO] Starting document loading process...")
    documents = load_all_documents("data")
    print(f"[INFO] Finished loading documents. Total documents loaded: {len(documents)}")

    if not documents:
        print("[WARNING] No documents found. Please add files to the 'data' directory.")
        exit()

    print("\n[INFO] Sample loaded documents:")
    for i, doc in enumerate(documents[:2]):  # Trimmed to 2 sample prints to save terminal space
        print(f"\nDocument {i+1}:")
        print(f"Content: {doc.page_content[:200]}...")
        print(f"Metadata: {doc.metadata}")
    
    # Method A: Manual Flow (What you started writing)
    print("\n--- Running Manual Chunking & Embedding Flow ---")
    embedding_pipeline = EmbeddingPipeline()
    chunks = embedding_pipeline.chunk_document(documents)
    embeddings_vectors = embedding_pipeline.generate_embeddings(chunks)
    print(f"[INFO] Generated embedding vectors shape: {embeddings_vectors.shape}")

    # Method B: Utilizing your Vector Store Flow (Recommended)
    print("\n--- Running Integrated Vector Store Flow ---")
    vector_store = FaissVectorStore()
    vector_store.build_from_documents(documents)
    
    # Test Querying
    test_query = "What information is inside the loaded documents?"
    results = vector_store.query(test_query, top_k=2)
    
    print("\n[INFO] Query Results:")
    for res in results:
        print(f"Distance: {res['distance']:.4f} | Text Chunk: {res['metadata']['text'][:150]}...")