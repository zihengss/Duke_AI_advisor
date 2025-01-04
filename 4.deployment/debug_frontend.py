import streamlit as st
import requests

# Streamlit app title
st.title("Process All Endpoint Frontend")

# Backend API URL (adjust this to match your backend server's URL)
API_URL = "http://localhost:8000/process_all"

# Input section for user query
query_input = st.text_area("Enter your query:", placeholder="Type your query here...")

# Submit button
if st.button("Submit"):
    if query_input.strip():
        try:
            # Prepare the payload
            payload = {"user_input": query_input}
            
            # Send POST request to the backend
            response = requests.post(API_URL, json=payload)
            
            # Process the response
            if response.status_code == 200:
                data = response.json()
                st.success("Query processed successfully!")
                
                # Display the results
                st.subheader("Response:")
                st.write(data.get("response", "No response returned."))

                st.subheader("Retrieved Documents:")
                retrieved_docs = data.get("retrieved_docs", [])
                for i, doc in enumerate(retrieved_docs, 1):
                    st.markdown(f"**Document {i}:** {doc}")

                st.subheader("Reranked Documents:")
                reranked_docs = data.get("reranked_docs", [])
                for i, doc in enumerate(reranked_docs, 1):
                    st.markdown(f"**Document {i}:** {doc}")

                st.subheader("Model Used:")
                st.write(data.get("model_used", "Unknown"))
            else:
                st.error(f"Error: {response.status_code} - {response.text}")
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
    else:
        st.warning("Please enter a query.")
