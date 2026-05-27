import os
from dotenv import load_dotenv
# FIXED: Corrected folder and file name mismatch (added underscore)
from src.vector_store import FaissVectorStore 
from langchain_groq import ChatGroq

load_dotenv()

class RAGSearch:
    def __init__(self, persist_dir: str = "faiss_store", embedding_model: str = "all-MiniLM-L6-v2", llm_model: str = "llama-3.1-8b-instant"):
        # FIXED: Pass parameters cleanly as keyword arguments
        self.vectorstore = FaissVectorStore(persist_dir=persist_dir, embedding_model=embedding_model)
        
        # Load or build vectorstore
        faiss_path = os.path.join(persist_dir, "faiss.index")
        meta_path = os.path.join(persist_dir, "metadata.pkl")
        
        if not (os.path.exists(faiss_path) and os.path.exists(meta_path)):
            print("[WARNING] Vector store files not found. Attempting to build from scratch...")
            # FIXED: Corrected the path to import from the src directory and match 'document_loader'
            from src.document_loader import load_all_documents
            docs = load_all_documents("data")
            if not docs:
                print("[ERROR] No data found in 'data/' directory. Cannot build vector store.")
            else:
                self.vectorstore.build_from_documents(docs)
        else:
            self.vectorstore.load()
            
        # FIXED: Pulled api key dynamically from .env instead of hardcoding an empty string
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            print("[WARNING] GROQ_API_KEY not found in environment variables. Ensure it is set in your .env file.")
            
        self.llm = ChatGroq(groq_api_key=groq_api_key, model_name=llm_model)
        print(f"[INFO] Groq LLM initialized: {llm_model}")

    def search_and_summarize(self, query: str, top_k: int = 5) -> str:
        results = self.vectorstore.query(query, top_k=top_k)
        
        # Extract content out safely
        texts = [r["metadata"].get("text", "") for r in results if r.get("metadata")]
        context = "\n\n".join(texts)
        
        if not context.strip():
            return "No relevant context found in documents to answer the query."
            
        # Refined RAG prompt template to enforce context usage
        prompt = f"""You are a helpful assistant. Answer the user query using only the provided context below. If you do not know the answer based on the context, say that you don't know.
        Context:
        {context}
        
        Query: {query}
        
        Answer:"""

        # Using LangChain invoke syntax
        response = self.llm.invoke([{"role": "user", "content": prompt}])
        return response.content

# Example usage check
if __name__ == "__main__":
    # Ensure you have GROQ_API_KEY="gsk_..." inside a .env file in your root folder
    rag = RAGSearch()
    user_query = "Summarize the main takeaways from the loaded data."
    print(f"\n[USER]: {user_query}")
    answer = rag.search_and_summarize(user_query, top_k=3)
    print(f"\n[LLM RESPONSE]:\n{answer}")