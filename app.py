# Streamlit UI for financial research assistant
import streamlit as st
from tools.vector_search import vector_search
from agents.analyst import Analyst

def main():
    st.title('Multi-Agent Financial Research Assistant')
    query = st.text_input('Enter your financial research question:')
    if query:
        retrieved_chunks = vector_search(query)
        analyst = Analyst()
        answer = analyst.answer_query(query, retrieved_chunks)
        st.markdown("### Answer")
        st.write(answer)
        st.markdown("### Citations")
        for chunk in retrieved_chunks:
            st.write(f"**{chunk['ticker']} {chunk['date']} [{chunk['section']}]**: {chunk['text'][:200]}...")

if __name__ == '__main__':
    main()
