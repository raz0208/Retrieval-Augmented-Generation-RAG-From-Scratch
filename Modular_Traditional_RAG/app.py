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

    # Check/Load Documents
    print("[INFO] Scanning 'data/' directory for documents...")
    documents = load_all_documents("data")
    
    if not documents:
        print("\n[WARNING] No files found in the 'data/' folder.")
        print("[INFO] If you have already built a FAISS index, the search engine will attempt to load it.")
    else:
        print(f"[INFO] Successfully loaded {len(documents)} source documents.")
        
        # Optional: Print a tiny snippet of the first document to verify structure
        print(f"[DEBUG] Sample Document Metadata: {documents[0].metadata}")

    # Initialize the RAG Search Engine
    """
    This automatically builds the vector store if it doesn't exist,
    or loads the existing index from 'faiss_store/' if it does.
    """"
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