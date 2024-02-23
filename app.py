import os
import time
from pathlib import Path

import numpy as np
import streamlit as st
from faiss import IndexFlatL2
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
from pypdf import PdfReader


@st.cache_resource
def get_client():
    """Returns a cached instance of the MistralClient."""
    api_key = os.environ["MISTRAL_API_KEY"]
    return MistralClient(api_key=api_key)


CLIENT: MistralClient = get_client()

PROMPT = """
An excerpt from a document is given below.

---------------------
{context}
---------------------

Given the document excerpt, answer the following query.
If the context does not provide enough information, decline to answer.
Do not output anything that can't be answered from the context.

Query: {query}
Answer:
"""

# Initialize session state variables if not already present
if "messages" not in st.session_state:
    st.session_state.messages = []


# Function to add a message to the chat
def add_message(msg, agent="ai", stream=True, store=True):
    """Adds a message to the chat interface, optionally streaming the output."""
    if stream and isinstance(msg, str):
        msg = stream_str(msg)

    with st.chat_message(agent):
        if stream:
            output = st.write_stream(msg)
        else:
            output = msg
            st.write(msg)

    if store:
        st.session_state.messages.append(dict(agent=agent, content=output))


# Function to stream a string with a delay
def stream_str(s, speed=250):
    """Yields characters from a string with a delay to simulate streaming."""
    for c in s:
        yield c
        time.sleep(1 / speed)


# Function to stream the response from the AI
def stream_response(response):
    """Yields responses from the AI, replacing placeholders as needed."""
    for r in response:
        content = r.choices[0].delta.content
        # prevent $ from rendering as LaTeX
        content = content.replace("$", "\$")
        yield content


# Decorator to cache the embedding computation
@st.cache_data
def embed(text: str):
    """Returns the embedding for a given text, caching the result."""
    return CLIENT.embeddings("mistral-embed", text).data[0].embedding


# Function to build and cache the index from PDFs in a directory
@st.cache_resource
def build_and_cache_index():
    """Builds and caches the index from PDF documents in the specified directory."""
    pdf_files = Path("data").glob("*.pdf")
    text = ""

    for pdf_file in pdf_files:
        reader = PdfReader(pdf_file)
        for page in reader.pages:
            text += page.extract_text() + "\n\n"

    chunk_size = 500
    chunks = [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]

    embeddings = np.array([embed(chunk) for chunk in chunks])
    dimension = embeddings.shape[1]
    index = IndexFlatL2(dimension)
    index.add(embeddings)

    return index, chunks


# Function to reply to queries using the built index
def reply(query: str, index: IndexFlatL2, chunks):
    """Generates a reply to the user's query based on the indexed PDF content."""
    embedding = embed(query)
    embedding = np.array([embedding])

    _, indexes = index.search(embedding, k=2)
    context = [chunks[i] for i in indexes.tolist()[0]]

    messages = [
        ChatMessage(role="user", content=PROMPT.format(context=context, query=query))
    ]
    response = CLIENT.chat_stream(model="mistral-medium", messages=messages)
    add_message(stream_response(response))


# Main application logic
def main():
    """Main function to run the application logic."""
    if st.sidebar.button("🔴 Reset conversation"):
        st.session_state.messages = []

    index, chunks = build_and_cache_index()

    for message in st.session_state.messages:
        with st.chat_message(message["agent"]):
            st.write(message["content"])

    query = st.chat_input("Ask something about your PDF")

    if not st.session_state.messages:
        add_message("Ask me anything!")

    if query:
        add_message(query, agent="human", stream=False, store=True)
        reply(query, index, chunks)


if __name__ == "__main__":
    main()
