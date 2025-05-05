import streamlit as st
from dotenv import load_dotenv
import os
from main import (
    graph, llm, vector_index, entity_chain, showGraph, generate_full_text_query,
    structured_retriever, retriever, _search_query, chain
)

# Load environment variables
load_dotenv()

# Streamlit app
st.title("KL Rahul Graph Chatbot")
st.write("Ask questions about KL Rahul and explore the knowledge graph.")

# Initialize session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
if question := st.chat_input("Ask a question about KL Rahul"):
    # Display user message
    with st.chat_message("user"):
        st.markdown(question)
    st.session_state.messages.append({"role": "user", "content": question})

    # Get response
    with st.spinner("Processing..."):
        # Query graph for visualization
        try:
            graph_output = showGraph().to_json()  # Convert GraphWidget to JSON for text representation
        except Exception as e:
            graph_output = f"Error displaying graph: {str(e)}\nRunning default query..."
            # Fallback to text-based graph query
            default_cypher = "MATCH (s)-[r:!MENTIONS]->(t) RETURN s.id, type(r), t.id LIMIT 50"
            result = graph.query(default_cypher)
            graph_output = "\n".join([f"{r['s.id']} - {r['type(r)']} -> {r['t.id']}" for r in result]) or "No results found"

        # Get chatbot response
        response = chain.invoke({
            "question": question,
            "chat_history": st.session_state.chat_history
        })

    # Display assistant response
    with st.chat_message("assistant"):
        st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
        
        # Update chat history
        st.session_state.chat_history.append((question, response))
        
        # Display graph data
        with st.expander("Graph Data"):
            st.text_area("Graph Relationships", graph_output, height=200)

# Clear chat history button
if st.button("Clear Chat History"):
    st.session_state.chat_history = []
    st.session_state.messages = []
    st.experimental_rerun()