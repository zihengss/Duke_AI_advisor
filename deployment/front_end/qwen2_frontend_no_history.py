import streamlit as st
import requests
from langchain.schema import (
    AIMessage,
    HumanMessage
)

# Set the title of the Streamlit app
st.title("Dukies: AI advisor for Duke University computer science major students.")

# Initialize the session state to store the chat history
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'history' not in st.session_state:
    st.session_state.history = [{"role": "system", "content": "You are Dukies, a AI advisor for Duke University computer science major students. You answer student's questions friendly and concisely"}]



for message in st.session_state["messages"]:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(message.content)

# Create a text input for the user to enter their prompt
prompt = st.chat_input("Enter your prompt:")

# Define the URL of the FastAPI endpoint
url = "http://127.0.0.1:8000/generate/"

# Create a button to send the request
if prompt:
    st.session_state['messages'].append(HumanMessage(content=prompt))
    st.session_state.history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
            st.markdown(prompt)
    try:
        
        payload = {
            "messages": [{"role": "system", "content": "You are Dukies, a AI advisor for Duke University computer science major students. You answer student's questions friendly and concisely"},
                         {"role": "user", "content": prompt}]
        }
        # Send the POST request to the FastAPI endpoint
        response = requests.post(url, json=payload)

        # Check if the request was successful
        if response.status_code == 200:
            # Parse the JSON response
            result = response.json()
            response_text = result["response"]
            file_name = result["file_name"]
            file_content = result["file_content"]

            # Update the chat history with the user's prompt and the generated response
            st.session_state.messages.append(AIMessage(content=response_text))
            st.session_state.history.append({"role": "assistant", "content": response_text})
            with st.chat_message("assistant"):
                st.markdown(f":blue-background[**File name:** {file_name}] \n\n :blue-background[**File content:** {file_content}]")
                st.markdown(response_text)
        else:
            st.error(f"Error: {response.status_code}")
            st.write(response.text)
    except requests.exceptions.RequestException as e:
        st.error(f"Request failed: {e}")

