# initialisation of environment variable for Open AI key and imports

import os
os.environ["OPENAI_API_KEY"] = "sk-proj-xMvVuPJN0DyZ65czTmTYT3BlbkFJW2PFU95Sl7L0GasG8CYC"

import streamlit as st
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings  
from langchain.memory import ConversationBufferMemory
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks import StreamlitCallbackHandler
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from htmlTemplates import css
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import os
import time



# Bind the Vector DB, Large Language models and Embedding models all into one container
def get_conversation_chain(vectorstore):
    """
    This is a langchain model where we will be binding the runner to infer data from LLM
    """
    
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

    
    llm = OpenAI(callback_manager = callback_manager, 
                    max_tokens= 200 , api_key="sk-proj-xMvVuPJN0DyZ65czTmTYT3BlbkFJW2PFU95Sl7L0GasG8CYC")


    prompt_template = """You are a personal Risk Copilot Bot assistant for answering any questions about risk and MSA's
    You are given a question and a set of documents.
    If the user's question requires you to provide specific information from the documents, give your answer based only on the examples provided below. DON'T generate an answer that is NOT written in the provided examples.
    If you don't find the answer to the user's question with the examples provided to you below, answer that you didn't find the answer in the documentation and propose him to rephrase his query with more details.
    Use bullet points if you have to make a list, only if necessary. Use 'DOCUMENTS' as a reference point, to understand and give a consciese output in 3 or 5 sentences. 
    QUESTION: {question}
    DOCUMENTS:
    =========
    {context}
    =========
    Finish by proposing your help for anything else.
    """

    rag_prompt_custom = PromptTemplate.from_template(prompt_template)

    
    conversation_chain = RetrievalQA.from_chain_type(
        llm,
        retriever= vectorstore.as_retriever(),
        chain_type_kwargs={"prompt": rag_prompt_custom},
    )
    conversation_chain.callback_manager = callback_manager
    conversation_chain.memory = ConversationBufferMemory()

    return conversation_chain

# an stream lit interface to handle and save our chats
def handle_userinput():

    clear = False

    # Add clear chat button
    if st.button("Clear Chat history"):
        clear = True
        st.session_state.messages = []

    # initialise our stream  lit chat interface
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "How can I help you?"}] 

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Clear the cash memory
    if clear:
        st.session_state.conversation.memory.clear()
        clear = False

    if prompt := st.chat_input():

        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # add user question to chat history
        st.session_state.messages.append( {"role": "user", "content": prompt})


        with st.chat_message("assistant"):
            # set up a call back handler
            st_callback = StreamlitCallbackHandler(st.container())
            message_holder = st.empty()
            full_response = ""

            # streamlit call back manager
            st.session_state.conversation.callback_manager = st_callback
            msg = st.session_state.conversation.run(prompt)
            #st.markdown(msg)
            for chunk in msg.split():
                full_response += chunk + " "
                time.sleep(0.09)

                # add a blinking cursor to simulate typing 
                message_holder.markdown(full_response + "✏️ ")

        # Display the responce
        message_holder.info(full_response)
        
        # add responce to session state
        st.session_state.messages.append({"role": "assistant", "content": full_response}) 


# Function to apply rounded edges using CSS
def add_rounded_edges(image_path="./randstad_featuredimage.png", radius=30):
    st.markdown(
        f'<style>.rounded-img{{border-radius: {radius}px; overflow: hidden;}}</style>',
        unsafe_allow_html=True,)
    st.image(image_path, use_column_width=True, output_format='auto')


def main():

    
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.title("Randstad Risk CoPilot Chatbot")
    st.subheader("Generative AI BOT")


    st.session_state.embeddings = OpenAIEmbeddings(api_key="sk-proj-xMvVuPJN0DyZ65czTmTYT3BlbkFJW2PFU95Sl7L0GasG8CYC")


    vectorstore = Chroma(persist_directory="./vectorembeddings/", embedding_function=st.session_state.embeddings)
    st.session_state.conversation = get_conversation_chain(vectorstore)
    handle_userinput()


if __name__ == '__main__':
    main()