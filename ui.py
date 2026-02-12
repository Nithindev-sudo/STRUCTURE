import streamlit as st
from config import AVAILABLE_MODELS, DEFAULT_MODEL


def render_sidebar():
    """
    Sidebar UI for model selection
    """
    st.sidebar.title("Settings")

    model_names = list(AVAILABLE_MODELS.keys())

    selected_model = st.sidebar.selectbox(
        "Select LLM Model",
        model_names,
        index=model_names.index(DEFAULT_MODEL)
    )

    return AVAILABLE_MODELS[selected_model]


def render_main_ui():
    """
    Main UI layout
    """

    st.title("AI Requirement Assistant")

    # File uploader
    uploaded_file = st.file_uploader(
        "Upload Requirement PDF",
        type=["pdf"]
    )

    # Chat display container
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    user_query = st.chat_input("Ask a question about the document...")

    return uploaded_file, user_query
