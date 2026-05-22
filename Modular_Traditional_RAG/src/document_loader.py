from pathlib import Path
from typing import List, Any
from langchain_community.document_loaders import TextLoader, PyPDFLoader, CSVLoader
from langchain_community.document_loaders.excel import UnstructuredExcelLoader
from langchain_community.document_loaders import JSONLoader
from langchain_community.document_loaders import Docx2txtLoader


def load_all_documents(data_dir: str) -> List[Any]:
    """
    Load all supported files from the specified directory and convert them to langchain documents structure.

    Args:
        data_dir (str): The directory containing the documents.
    Returns:
        List[Any]: A list of loaded documents.
    """
    