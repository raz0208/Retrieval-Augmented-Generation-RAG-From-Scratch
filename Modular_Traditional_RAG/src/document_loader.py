from pathlib import Path
from typing import List, Any
from langchain_community.document_loaders import TextLoader, PyPDFLoader, CSVLoader
from langchain_community.document_loaders.excel import UnstructuredExcelLoader
from langchain_community.document_loaders import JSONLoader
from langchain_community.document_loaders import Docx2txtLoader

class DocumentLoader:
    def __init__(self, file_path: str):
        self.file_path = file_path

    def load(self) -> List[Any]:
        file_extension = Path(self.file_path).suffix.lower()
        if file_extension == '.txt':
            loader = TextLoader(self.file_path)
        elif file_extension == '.pdf':
            loader = PyPDFLoader(self.file_path)
        elif file_extension == '.csv':
            loader = CSVLoader(self.file_path)
        elif file_extension in ['.xls', '.xlsx']:
            loader = UnstructuredExcelLoader(self.file_path)
        elif file_extension == '.json':
            loader = JSONLoader(self.file_path)
        elif file_extension == '.docx':
            loader = Docx2txtLoader(self.file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")
        
        return loader.load()

def load_document(file_path: str) -> List[Any]:
    loader = DocumentLoader(file_path)
    return loader.load()

# Exampe usage:
if __name__ == "__main__":
    file_path = "example.pdf"  # Replace with your file path
    documents = load_document(file_path)
    for doc in documents:
        print(doc)
