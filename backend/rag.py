from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings
)

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

def build_rag_chain(pdf_path: str):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    splits = splitter.split_documents(documents)

    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001"
    )

    vectorstore = FAISS.from_documents(splits,embeddings)
    retriever = vectorstore.as_retriever()

    prompt = ChatPromptTemplate.from_template(
        """
        You are a helpful assistant.
        Answer ONLY using the provided context.
        If the answer is not in the context, say "I don't know".
        You also Extract the details and return from the uploaded file context.
        Context:
        {context}

        Question:
        {question}
        """
    )    

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        tempereture = 0
    )

    return (
        {
            "context":retriever,
            "question":RunnablePassthrough()
        }
        | prompt
        | llm
    )