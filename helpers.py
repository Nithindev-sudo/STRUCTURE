import os
import httpx
from pdfminer.high_level import extract_text
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

from config import BASE_URL, EMBEDDING_MODEL


# ---------- Extract text from uploaded file ----------
def get_text_from_file(file_path: str) -> str:
    """
    Extracts text from a PDF file using pdfminer
    """
    text = extract_text(file_path)
    return text


# ---------- Initialize Vector Database ----------
def initialize_vector_db(text: str, persist_directory: str = "chroma_db"):
    """
    Creates embeddings and initializes Chroma vector database
    """

    # Create HTTP client with SSL verify disabled
    http_client = httpx.Client(verify=False)

    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        base_url=BASE_URL,
        api_key=os.getenv("OPENAI_API_KEY"),
        http_client=http_client
    )

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )

    docs = text_splitter.create_documents([text])

    vector_db = Chroma.from_documents(
        docs,
        embedding=embeddings,
        persist_directory=persist_directory
    )

    vector_db.persist()

    return vector_db
