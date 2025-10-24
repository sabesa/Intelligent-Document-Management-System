from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain_community.vectorstores import Qdrant
from langchain.prompts import PromptTemplate
from langchain.embeddings import OpenAIEmbeddings
from qdrant_client import QdrantClient
from config import *

def get_rag_pipeline():
    
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL, openai_api_key=OPENAI_API_KEY)

    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    vectorstore = Qdrant(client=client, collection_name=QDRANT_COLLECTION, embeddings=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    llm = ChatOpenAI(model_name=LLM_MODEL, temperature=0, openai_api_key=OPENAI_API_KEY)

    prompt = PromptTemplate(
        template=(
            '''You are a helpful AI assistant helping consultants answer questions based on document context.

Context Information:
{context}

Chat History:
{chat_history}

Question: {question}

Instructions:
1. Answer based ONLY on the provided context
2. If the context doesn't contain relevant information, say "I don't have enough information to answer this question"
3. Cite specific documents when possible (e.g., "According to [Document Name]...")
4. Be concise and professional
5. If multiple documents provide conflicting information, acknowledge this'''
        ),
        input_variables=["context", "question"]
    )

    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )

    return rag_chain
