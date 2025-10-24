from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.chat_models import ChatOpenAI
from langchain_community.document_loaders import UnstructuredFileLoader
from config import *

def summarize_document(file_path: str):
    loader = UnstructuredFileLoader(file_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    texts = splitter.split_documents(documents)

    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.3, openai_api_key=OPENAI_API_KEY)
    chain = load_summarize_chain(llm, chain_type="map_reduce", verbose=True)

    summary = chain.run(texts)
    return summary
