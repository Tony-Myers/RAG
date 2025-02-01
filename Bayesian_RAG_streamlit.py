import streamlit as st
from Bayesian_RAG import generate_response  # Your RAG pipeline module

def check_password():
    """Simple password protection using Streamlit secrets."""
    if "password_correct" not in st.session_state:
        pwd = st.text_input("Enter password", type="password")
        if pwd == st.secrets["general"]["password"]:
            st.session_state["password_correct"] = True
        else:
            st.error("Password is incorrect")
            st.stop()  # Stop execution until correct password is provided
    return True

def main():
    # Ensure the user is authenticated
    if check_password():
        st.title("Bayesian AI Assistant")
        st.markdown(
            """
            This assistant uses a RAG pipeline to answer questions about Bayesian documentation.
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
