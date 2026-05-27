# from src.document_loader import load_all_documents
# from src.embedding import EmbeddingPipeline
# from src.vector_store import FaissVectorStore

# if __name__ == "__main__":
#     print("[INFO] Starting document loading process...")
#     documents = load_all_documents("data")
#     print(f"[INFO] Finished loading documents. Total documents loaded: {len(documents)}")

#     if not documents:
#         print("[WARNING] No documents found. Please add files to the 'data' directory.")
#         exit()

#     print("\n[INFO] Sample loaded documents:")
#     for i, doc in enumerate(documents[:2]):  # Trimmed to 2 sample prints to save terminal space
#         print(f"\nDocument {i+1}:")
#         print(f"Content: {doc.page_content[:200]}...")
#         print(f"Metadata: {doc.metadata}")
    
#     # Method A: Manual Flow (What you started writing)
#     print("\n--- Running Manual Chunking & Embedding Flow ---")
#     embedding_pipeline = EmbeddingPipeline()
#     chunks = embedding_pipeline.chunk_document(documents)
#     embeddings_vectors = embedding_pipeline.generate_embeddings(chunks)
#     print(f"[INFO] Generated embedding vectors shape: {embeddings_vectors.shape}")

#     # Method B: Utilizing your Vector Store Flow (Recommended)
#     print("\n--- Running Integrated Vector Store Flow ---")
#     vector_store = FaissVectorStore()
#     vector_store.build_from_documents(documents)
    
#     # Test Querying
#     test_query = "What information is inside the loaded documents?"
#     results = vector_store.query(test_query, top_k=2)
    
#     print("\n[INFO] Query Results:")
#     for res in results:
#         print(f"Distance: {res['distance']:.4f} | Text Chunk: {res['metadata']['text'][:150]}...")

import os
from src.document_loader import load_all_documents
from src.vector_store import FaissVectorStore
from src.search import RAGSearch

def setup_environment():
    """Ensure data directories exist before running the pipeline."""
    if not os.path.exists("data"):
        os.makedirs("data")
        print("[INFO] Created missing 'data/' directory. Please place your documents here.")
    
    if not os.path.exists(".env"):
        print("[WARNING] '.env' file not found! Please create a .env file in the root directory and add your GROQ_API_KEY.")

def main():
    setup_environment()
    
    print("\n==============================================")
    print("🚀 INITIALIZING MODULAR RAG PIPELINE 🚀")
    print("==============================================\n")

    # 1. Check/Load Documents
    print("[INFO] Scanning 'data/' directory for documents...")
    documents = load_all_documents("data")
    
    if not documents:
        print("\n[WARNING] No files found in the 'data/' folder.")
        print("[INFO] If you have already built a FAISS index, the search engine will attempt to load it.")
    else:
        print(f"[INFO] Successfully loaded {len(documents)} source documents.")
        
        # Optional: Print a tiny snippet of the first document to verify structure
        print(f"[DEBUG] Sample Document Metadata: {documents[0].metadata}")

    # 2. Initialize the RAG Search Engine
    # This automatically builds the vector store if it doesn't exist,
    # or loads the existing index from 'faiss_store/' if it does.
    print("\n[INFO] Starting RAG Search Engine (Loading Index & ChatGroq LLM)...")
    try:
        rag_engine = RAGSearch(
            persist_dir="faiss_store",
            embedding_model="all-MiniLM-L6-v2",
            llm_model="llama-3.1-8b-instant"
        )
    except Exception as e:
        print(f"\n[CRITICAL ERROR] Failed to initialize RAG Engine: {e}")
        print("[INFO] Please verify your Groq API key and library installations.")
        return

    print("\n==============================================")
    print("🎉 RAG SYSTEM READY FOR QUERIES 🎉")
    print("==============================================\n")

    # 3. Interactive Query Loop
    while True:
        try:
            query = input("Ask a question about your documents (or type 'exit' to quit): ").strip()
            
            if not query:
                continue
            if query.lower() in ['exit', 'quit', 'q']:
                print("\nShutting down RAG pipeline. Goodbye!")
                break
                
            print(f"\n[PROCESSING] Searching vector store and generating answer...")
            response = rag_engine.search_and_summarize(query, top_k=4)
            
            print("\n📬 [RESPONSE]:")
            print("-" * 50)
            print(response)
            print("-" * 50 + "\n")
            
        except KeyboardInterrupt:
            print("\n\nPipeline interrupted by user. Exiting.")
            break
        except Exception as e:
            print(f"\n[ERROR] An error occurred while processing your query: {e}\n")

if __name__ == "__main__":
    main()