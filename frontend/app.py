import streamlit as st
import requests

BACKEND_URL = "http://localhost:8000"

st.set_page_config(page_title="Chat with PDF", layout="centered")
st.title("📄 Chat with PDF (LangChain)")

# Upload PDF
uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file:
    with st.spinner("Uploading & processing PDF..."):
        response = requests.post(
            f"{BACKEND_URL}/upload",
            files={"file": uploaded_file}
        )
    st.success(response.json()["message"])

# Chat UI
st.divider()
question = st.text_input("Ask a question about the PDF")

if st.button("Ask"):
    if question:
        response = requests.post(
            f"{BACKEND_URL}/chat",
            json={"question": question}
        )
        st.markdown("### Answer")
        st.write(response.json()["answer"])
