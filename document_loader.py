from langchain_community.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient
from config import *

def ingest_document(file_path: str):
    loader = UnstructuredFileLoader(file_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " "]
    )
    chunks = splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL, openai_api_key=OPENAI_API_KEY)

    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    vectorstore = Qdrant(client=client, collection_name=QDRANT_COLLECTION, embeddings=embeddings)

   
    vectorstore.add_documents(chunks)

 
