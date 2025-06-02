import streamlit as st
from langchain_ollama import ChatOllama
import concurrent.futures

def safe_invoke(llm, prompt, timeout=20):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(llm.invoke, prompt)
        try:
            return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            return "⚠️ Model timed out!"

# Initialize the LLM
llm = ChatOllama(model="mistral")

st.title("Minimal Ollama Test")
prompt = st.text_input("Ask something:")

if prompt:
    with st.spinner("Thinking..."):
        response = safe_invoke(llm, prompt)
        st.write(response.content) 