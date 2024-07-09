from langchain.document_loaders import PyPDFLoader,DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
 

def load_pdf(data):
    loader = DirectoryLoader(data,glob="*pdf",
                    loader_cls=PyPDFLoader)
    docs = loader.load()
    return docs


def text_splitter(extracted_data):
    splitter = RecursiveCharacterTextSplitter(chunk_size=3000,chunk_overlap=500)
    chunks = splitter.split_documents(extracted_data)
    return chunks