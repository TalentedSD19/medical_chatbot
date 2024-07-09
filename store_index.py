from src.helper import load_pdf,text_splitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

extracted_data = load_pdf("data/")

text_chunks = text_splitter(extracted_data)

embeddings=OpenAIEmbeddings()

persist_directory=r"db"
vectorstore = Chroma.from_documents(
    text_chunks,
    embedding=embeddings,
    persist_directory=persist_directory
)