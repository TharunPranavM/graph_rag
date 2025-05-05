
# 🚀 Graph Chatbot

**Graph Chatbot** is a powerful **Streamlit-based web application** that builds a **knowledge graph** from any **Wikipedia topic** and allows users to chat with it. It leverages:

* 🧠 **LangChain** for natural language processing
* 🤖 **Grok** for large language model (LLM) capabilities
* 🌐 **Neo4j** to store and query graph-based knowledge

Users can input topics like `"KL Rahul"` or `"Virat Kohli"` via **terminal (main.py)** or the **Streamlit UI (app.py)** and interactively ask questions about the topic.

---

## 🔧 Components

### `main.py`

* Ingests data from Wikipedia
* Builds the Neo4j knowledge graph
* Handles query and retrieval logic

### `app.py`

* Provides a **Streamlit web interface**
* Accepts user input for topics
* Offers a **chat interface** to query the graph

---

## 🌟 Key Features

* 🔄 Dynamic knowledge graph creation from any Wikipedia topic
* 💬 Interactive chatbot with follow-up question support
* 📊 Text-based relationship visualization
* 🔍 Hybrid search (structured + vector-based)
* ♻️ Option to change topics and reset conversations

---

## ⚙️ Prerequisites

* Python 3.8+
* Neo4j (local or cloud)
* Grok API key (from [xAI](https://x.ai))
* Internet access for Wikipedia and API calls
* `.env` file for environment variables

---

## 🛠️ Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd graph-chatbot
```

### 2. Set Up Virtual Environment & Install Dependencies

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Or manually install:

```bash
pip install streamlit python-dotenv langchain langchain-community langchain-groq neo4j sentence-transformers yfiles-jupyter-graphs
```

### 3. Create `.env` File

```env
GROQ_API_KEY=your_groq_api_key
NEO4J_URI=your_neo4j_uri
NEO4J_USERNAME=your_neo4j_username
NEO4J_PASSWORD=your_neo4j_password
```

---

## 🚀 How to Use

### 🔹 Run via Terminal

```bash
python main.py
```

* Enter a topic (e.g., `KL Rahul`)
* Wikipedia data is processed and stored in Neo4j
* Use internal functions to query the graph

### 🔹 Launch Streamlit App

```bash
streamlit run app.py
```

* Enter topic (e.g., `Virat Kohli`)
* Chat with the bot about the topic
* Explore structured graph data and follow-up queries
* Use:

  * ✅ **Change Topic** to switch
  * 🧹 **Clear Chat** to reset session

---

## 🧠 Workflow

### 📥 Data Ingestion

* Topic fetched using `WikipediaLoader`
* Text split using `TokenTextSplitter`
* Transformed into a graph with `LLMGraphTransformer`
* Stored in **Neo4j**

### 🔗 Graph + Vector Index

* Nodes + relationships modeled in Neo4j
* Full-text and vector indices built using HuggingFace embeddings

### 🧠 Chatbot Logic

* Extracts entities from questions
* Queries Neo4j and vector DB for context
* Uses Grok LLM for generating answers

### 💻 Streamlit Frontend

* `app.py` integrates with `main.py`
* Handles UI, topic input, chat interface, session state
* Graph shown as **text output** (future visual support planned)

---

## ✅ Testing

```bash
# Test in terminal
python main.py

# Test UI
streamlit run app.py
```

* Try various topics
* Validate follow-up Q\&A
* Check topic switching and reset functionality


---

## ❗ Troubleshooting

| Issue                   | Solution                                            |
| ----------------------- | --------------------------------------------------- |
| 🔌 Neo4j connection     | Check `.env` credentials and DB status              |
| 🚫 Grok API error       | Validate API key and quota                          |
| ❓ Invalid topic         | Use specific, well-known Wikipedia terms            |
| 📉 No graph output      | Graph shown as text; add `pyvis` for richer visuals |
| 🔀 Dependency conflicts | Use a fresh virtual environment                     |

---

## 🔮 Future Enhancements

* 🌐 **Graph visualization** with `pyvis` or `networkx`
* 🧠 **Multi-topic graph merging/comparison**
* ⚡ **Session caching** for faster reloads
* 🛡️ Better **error handling** for API and input issues


