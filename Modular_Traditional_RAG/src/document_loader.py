# Import necessary libraries for document loading
from pathlib import Path
from typing import List, Any
from langchain_community.document_loaders import TextLoader, PyPDFLoader, CSVLoader
from langchain_community.document_loaders.excel import UnstructuredExcelLoader
from langchain_community.document_loaders import JSONLoader
from langchain_community.document_loaders import Docx2txtLoader

# Function to load all supported documents from a specified directory
def load_all_documents(data_dir: str) -> List[Any]:
    """
    Load all supported files from the specified directory and convert them to langchain documents structure.
    Supported file types include .txt, .pdf, .csv, .xlsx, .json, and .docx.

    Args:
        data_dir (str): The directory containing the documents.
    Returns:
        List[Any]: A list of loaded documents.
    """
    # Use data root folder to load all documents
    data_path = Path(data_dir).resolve()
    print(f"Loading documents from: {data_path}\n")
    documents = []

    # Load .txt files
    txt_files = list(data_path.glob("**/*.txt"))
    print(f"[DEBUG] Found {len(txt_files)} .txt files: {[str(file) for file in txt_files]}")
    for txt_file in txt_files:
        print(f"[DEBUG] Loading .txt file: {txt_file}")
        try:
            loader = TextLoader(str(txt_file))
            loaded = loader.load()
            print(f"[DEBUG] Loaded {len(loaded)} txt documents from {txt_file}")
            documents.extend(loaded)
        except Exception as e:
            print(f"[ERROR] Failed to load .txt file {txt_file}: {e}")
    
    # Load .pdf files
    pdf_files = list(data_path.glob("**/*.pdf"))
    print(f"[DEBUG] Found {len(pdf_files)} .pdf files: {[str(files) for files in pdf_files]}")
    for pdf_file in pdf_files:
        print(f"[DEBUG] Loading .pdf file: {pdf_file}")
        try:
            loader = PyPDFLoader(str(pdf_file))
            loaded = loader.load()
            print(f"[DEBUG] Loaded {len(loaded)} pdf documents from {pdf_file}")
            documents.extend(loaded)
        except Exception as e:
            print(f"[ERROR] Failed to load .pdf file {pdf_file}: {e}")

    # Load .csv files
    csv_files = list(data_path.glob("**/*.csv"))
    print(f"[DEBUG] Found {len(csv_files)} .csv files: {[str(file) for file in csv_files]}")
    for csv_file in csv_files:
        print(f"[DEBUG] Loading .csv file: {csv_file}")
        try:
            loader = CSVLoader(str(csv_file))
            loaded = loader.load()
            print(f"[DEBUG] Loaded {len(loaded)} csv documents from {csv_file}")
            documents.extend(loaded)
        except Exception as e:
            print(f"[ERROR] Failed to load .csv file {csv_file}: {e}")

    # Load .xlsx excel files
    xlsx_files = list(data_path.glob("**/*.xlsx"))
    print(f"[DEBUG] Found {len(xlsx_files)} .xlsx files: {[str(file) for file in xlsx_files]}")
    for xlsx_file in xlsx_files:
        print(f"[DEBUG] Loading .xlsx file: {xlsx_file}")
        try:
            loader = UnstructuredExcelLoader(str(xlsx_file))
            loaded = loader.load()
            print(f"[DEBUG] Loaded {len(loaded)} xlsx documents from {xlsx_file}")
            documents.extend(loaded)
        except Exception as e:
            print(f"[ERROR] Failed to load .xlsx file {xlsx_file}: {e}")

    # Load .json files
    json_files = list(data_path.glob("**/*.json"))
    print(f"[DEBUG] Found {len(json_files)} .json files: {[str(file) for file in json_files]}")
    for json_file in json_files:
        print(f"[DEBUG] Loading .json file: {json_file}")
        try:
            loader = JSONLoader(str(json_file))
            loaded = loader.load()
            print(f"[DEBUG] Loaded {len(loaded)} json documents from {json_file}")
            documents.extend(loaded)
        except Exception as e:
            print(f"[ERROR] Failed to load .json file {json_file}: {e}")

    # Load .docx word files
    docx_files = list(data_path.glob("**/*.docx"))
    print(f"[DEBUG] Found {len(docx_files)} .docx files: {[str(file) for file in docx_files]}")
    for docx_file in docx_files:
        print(f"[DEBUG] Loading .docx file: {docx_file}")
        try:
            loader = Docx2txtLoader(str(docx_file))
            loaded = loader.load()
            print(f"[DEBUG] Loaded {len(loaded)} docx documents from {docx_file}")
            documents.extend(loaded)
        except Exception as e:
            print(f"[ERROR] Failed to load .docx file {docx_file}: {e}")

    print(f"\n[INFO] Total documents loaded: {len(documents)}")
    return documents

    # # Example usage:
    # if __name__ == "__main__":
    #     data_directory = "data"  # Change this to your actual data directory
    #     all_docs = load_all_documents(data_directory)
    #     print(f"Total documents loaded: {len(all_docs)}")