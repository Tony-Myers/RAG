# Bayesian_RAG_streamlit.py
# Bayesian_RAG_streamlit.py
import streamlit as st
from Bayesian_RAG import generate_response

def check_password():
    """
    Simple password protection using Streamlit secrets.
    Ensure you have a .streamlit/secrets.toml file with:
    
    [general]
    password = "your_simple_password"
    """
    if "password_correct" not in st.session_state:
        pwd = st.text_input("Enter password", type="password")
        if pwd == st.secrets["general"]["password"]:
            st.session_state["password_correct"] = True
        else:
            st.error("Password is incorrect")
            st.stop()  # Stop execution if the password is wrong.
    return True

def main():
    if check_password():
        st.title("Bayesian AI Assistant")
        st.markdown(
            """
            This assistant uses a Retrieval-Augmented Generation (RAG) pipeline to answer questions about Bayesian documentation.
            
            **How it works:**
            1. Your question is used to retrieve relevant chunks from stored Bayesian R documentation.
            2. These chunks are sent to the DeepSeek API to generate a context-aware answer.
            3. The answer is displayed below.
            """
        )
        question = st.text_input("Enter your question:", value="How do I specify a multilevel model in brms?")
        if st.button("Get Answer"):
            with st.spinner("Generating answer..."):
                answer = generate_response(question)
            st.subheader("Answer")
            st.write(answer)

if __name__ == "__main__":
    main()
