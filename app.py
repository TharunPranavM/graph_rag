import streamlit as st
from dotenv import load_dotenv
import os
from main import (
    graph, llm, vector_index, entity_chain, showGraph, generate_full_text_query,
    structured_retriever, retriever, _search_query, chain
)
from langchain.document_loaders import WikipediaLoader
from langchain.text_splitter import TokenTextSplitter
from langchain_experimental.graph_transformers import LLMGraphTransformer

# Load environment variables
load_dotenv()

# Function to load graph data based on user input
@st.cache_data
def load_graph_data(query: str):
    raw_documents = WikipediaLoader(query=query).load()
    text_splitter = TokenTextSplitter(chunk_size=512, chunk_overlap=24)
    documents = text_splitter.split_documents(raw_documents[:3])
    
    llm_transformer = LLMGraphTransformer(llm=llm)
    graph_documents = llm_transformer.convert_to_graph_documents(documents)
    
    graph.add_graph_documents(
        graph_documents,
        baseEntityLabel=True,
        include_source=True
    )
    return f"Graph data loaded successfully for topic: {query}"

# Streamlit app
st.title("Graph Chatbot")
st.write("Enter a Wikipedia topic to build a knowledge graph and ask questions about it.")

# Initialize session state for chat history and topic
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "messages" not in st.session_state:
    st.session_state.messages = []
if "topic" not in st.session_state:
    st.session_state.topic = None
if "graph_loaded" not in st.session_state:
    st.session_state.graph_loaded = False

# User input for Wikipedia topic
topic = st.text_input("Enter Wikipedia topic (e.g., KL Rahul, Virat Kohli):", key="topic_input")
if topic and topic != st.session_state.topic:
    st.session_state.topic = topic
    st.session_state.graph_loaded = False
    st.session_state.chat_history = []
    st.session_state.messages = []

# Load graph data when topic is provided
if st.session_state.topic and not st.session_state.graph_loaded:
    with st.spinner(f"Loading graph data for {st.session_state.topic}..."):
        try:
            st.write(load_graph_data(st.session_state.topic))
            st.session_state.graph_loaded = True
        except Exception as e:
            st.error(f"Error loading graph data: {str(e)}")

# Display chat interface only if graph is loaded
if st.session_state.graph_loaded:
    st.subheader(f"Chat about {st.session_state.topic}")
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User input for questions
    if question := st.chat_input(f"Ask a question about {st.session_state.topic}"):
        # Display user message
        with st.chat_message("user"):
            st.markdown(question)
        st.session_state.messages.append({"role": "user", "content": question})

        # Get response
        with st.spinner("Processing..."):
            # Query graph for visualization
            try:
                graph_output = showGraph().to_json()  # Convert GraphWidget to JSON
            except Exception as e:
                graph_output = f"Error displaying graph: {str(e)}\nRunning default query..."
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

# Button to reset topic
if st.session_state.topic:
    if st.button("Change Topic"):
        st.session_state.topic = None
        st.session_state.graph_loaded = False
        st.session_state.chat_history = []
        st.session_state.messages = []
        st.experimental_rerun()