# app.py

import os
import httpx
import tempfile
import streamlit as st
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from pdfminer.high_level import extract_text


# ---------- Load environment ----------
load_dotenv()

# ---------- Constants ----------
BASE_URL = "https://genailab.tcs.in"
EMBEDDING_MODEL = "azure/genailab-maas-text-embedding-3-large"

AVAILABLE_MODELS = [
    "genailab-maas-gpt-35-turbo",
    "gemini-3-pro-preview",
    "azure/genailab-maas-gpt-4o-mini",
    "azure_ai/genailab-maas-DeepSeek-R1",
    "azure_ai/genailab-maas-Llama-3.2-90B-Vision-Instruct",
    "azure/genailab-maas-gpt-4.1",
    "genailab-maas-DeepSeek-V3-0324",
    "gemini-2.5-pro",
]

DEFAULT_MODEL = "azure_ai/genailab-maas-DeepSeek-R1"


# ---------- Helper Functions ----------
def get_text_from_file(file_path):
    return extract_text(file_path)


def initialize_vector_db(text):

    http_client = httpx.Client(verify=False)

    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        base_url=BASE_URL,
        api_key=os.getenv("OPENAI_API_KEY"),
        http_client=http_client,
    )

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs = splitter.create_documents([text])

    vector_db = Chroma.from_documents(docs, embedding=embeddings)
    return vector_db


def create_chain(vector_db, model_name):

    http_client = httpx.Client(verify=False)

    llm = ChatOpenAI(
        model=model_name,
        base_url=BASE_URL,
        api_key=os.getenv("OPENAI_API_KEY"),
        http_client=http_client,
        temperature=0,
    )

    prompt_template = """
Use the following context to answer the question.
If the answer is not found, say you don't know.

Context:
{context}

Question:
{question}

Answer:
"""

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"],
    )

    retriever = vector_db.as_retriever(search_kwargs={"k": 4})

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True,
    )

    return chain


# ---------- Streamlit UI ----------
st.set_page_config(page_title="RAG Requirement Assistant")

st.sidebar.title("Model Settings")
selected_model = st.sidebar.selectbox(
    "Select LLM Model",
    AVAILABLE_MODELS,
    index=AVAILABLE_MODELS.index(DEFAULT_MODEL),
)

st.title("AI Requirement Assistant")

uploaded_file = st.file_uploader("Upload Requirement PDF", type=["pdf"])

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_query = st.chat_input("Ask something about the document...")


# ---------- Document Processing ----------
if uploaded_file and "vector_db" not in st.session_state:
    with st.spinner("Processing document..."):

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        text = get_text_from_file(tmp_path)
        st.session_state.vector_db = initialize_vector_db(text)


# ---------- Chain Creation / Reinitialization ----------
if "vector_db" in st.session_state:

    if (
        "current_model" not in st.session_state
        or st.session_state.current_model != selected_model
    ):
        st.session_state.chain = create_chain(
            st.session_state.vector_db,
            selected_model,
        )
        st.session_state.current_model = selected_model

    if user_query:
        st.session_state.messages.append({"role": "user", "content": user_query})

        with st.spinner("Generating response..."):
            result = st.session_state.chain({"query": user_query})
            response = result["result"]

        st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun()
