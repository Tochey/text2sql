import streamlit as st
from text2sql import Text2Sql

agent = Text2Sql()

def main():

    st.title("AI ENG Text2SQL")
    user_question = st.text_input("Ask a question:")

    if user_question:
        
        llm_response = agent.rephrase_chain().invoke({
            "question" : user_question
        })

        st.write(llm_response)
           
        
main()