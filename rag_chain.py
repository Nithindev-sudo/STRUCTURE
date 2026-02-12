import os
import httpx

from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

from config import BASE_URL, DEFAULT_MODEL


# -------- Initialize LLM --------
def initialize_llm(model_name: str = DEFAULT_MODEL):
    """
    Initialize ChatOpenAI with required base_url and SSL disabled
    """

    http_client = httpx.Client(verify=False)

    llm = ChatOpenAI(
        model=model_name,
        base_url=BASE_URL,
        api_key=os.getenv("OPENAI_API_KEY"),
        http_client=http_client,
        temperature=0
    )

    return llm


# -------- Build RAG Chain --------
def create_retrieval_chain(vector_db, model_name: str = DEFAULT_MODEL):
    """
    Creates a RetrievalQA chain using Chroma vector DB
    """

    llm = initialize_llm(model_name)

    retriever = vector_db.as_retriever(search_kwargs={"k": 4})

    prompt_template = """
You are an expert AI assistant helping analyze software requirement documents.

Use the provided context to answer the question.
If the answer is not found in the context, say you don't know.
Be precise and structured in your response.

Context:
{context}

Question:
{question}

Answer:
"""

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )

    return qa_chain
