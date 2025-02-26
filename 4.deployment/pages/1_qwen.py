import requests
import streamlit as st

API_URL = "http://localhost:8000/process_all_local"

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("Dukies: AI Course Advisor")
# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Message Dukies"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    payload = {"user_input": prompt}
    response = requests.post(API_URL, json=payload)
    response = response.json().get("response", "No response returned.")
    
    # Display bot response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)
    
    # Add bot response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})