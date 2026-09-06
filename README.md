# Personal Document RAG Chatbot

A simple **Retrieval-Augmented Generation (RAG)** chatbot built with Python that allows users to ask questions about their own documents.

The system reads `.txt` and `.pdf` files, converts them into embeddings, stores them in a **FAISS vector database**, retrieves the most relevant document chunks, and uses **DeepSeek** to generate an answer based on the retrieved context.

## Features

* 📄 Supports `.txt` and `.pdf` documents
* 🔍 Semantic document search using vector embeddings
* 🧠 Retrieval-Augmented Generation (RAG)
* ⚡ FAISS vector database for fast similarity search
* 🤗 HuggingFace `all-MiniLM-L6-v2` embeddings
* 🤖 DeepSeek API for answer generation
* 🚀 FastAPI REST API
* 💻 Command-line chatbot interface
* 🔐 API key stored securely using `.env`

## Project Structure

```text
Personal-Document-RAG-Chatbot/
│
├── documents/
│   └── sample.txt
│
├── vectorstore/
│   └── # Generated FAISS vector database
│
├── ingest.py
├── rag.py
├── main.py
├── requirements.txt
├── .env
└── .gitignore
```

## How It Works

The project follows a simple RAG pipeline:

```text
Documents
    │
    ▼
Document Loading
    │
    ▼
Text Splitting
    │
    ▼
HuggingFace Embeddings
    │
    ▼
FAISS Vector Database
    │
    ▼
User Question
    │
    ▼
Similarity Search
    │
    ▼
Relevant Document Chunks
    │
    ▼
DeepSeek LLM
    │
    ▼
Generated Answer
```

## Technologies Used

* **Python**
* **LangChain**
* **LangChain Community**
* **LangChain Text Splitters**
* **FAISS**
* **HuggingFace Sentence Transformers**
* **DeepSeek API**
* **FastAPI**
* **python-dotenv**
* **PyPDF**

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/NawrozHaseen/Personal-Document-RAG-Chatbot.git
cd Personal-Document-RAG-Chatbot
```

### 2. Create a virtual environment

Windows:

```powershell
python -m venv .venv
```

Activate it:

```powershell
.venv\Scripts\activate
```

### 3. Install dependencies

```powershell
pip install -r requirements.txt
```

## Environment Variables

Create a `.env` file in the project root:

```env
DEEPSEEK_API_KEY=your_deepseek_api_key
```

Do **not** commit your `.env` file to GitHub.

The `.gitignore` file should contain:

```gitignore
.env
.venv/
venv/
__pycache__/
vectorstore/
```

## Adding Documents

Place your documents inside:

```text
documents/
```

Supported formats:

```text
.txt
.pdf
```

For example:

```text
documents/
├── sample.txt
├── notes.txt
└── research_paper.pdf
```

## Creating the Vector Database

After adding your documents, run:

```powershell
python ingest.py
```

This will:

1. Load the documents.
2. Split them into smaller chunks.
3. Generate embeddings.
4. Create a FAISS vector database.
5. Save the vector database inside `vectorstore/`.

You only need to run ingestion again when you add, remove, or modify documents.

## Running the Chatbot

### Command Line Interface

Run:

```powershell
python rag.py
```

You can then ask questions about your documents.

Example:

```text
Question: What is the main topic of the document?

Answer: ...
```

To exit:

```text
exit
```

## Running the FastAPI Server

Start the API:

```powershell
uvicorn main:app --reload
```

The server will run locally at:

```text
http://127.0.0.1:8000
```

You can also open the interactive API documentation at:

```text
http://127.0.0.1:8000/docs
```

### API Endpoint

```http
POST /ask
```

Example request:

```json
{
    "question": "What is this document about?"
}
```

Example response:

```json
{
    "answer": "The document discusses..."
}
```

## RAG Components

### 1. Document Loader

The application loads `.txt` and `.pdf` documents from the `documents` directory.

### 2. Text Splitter

Documents are divided into smaller chunks using:

```python
RecursiveCharacterTextSplitter
```

Current configuration:

```text
Chunk Size: 500
Chunk Overlap: 50
```

Chunking allows the system to retrieve only the relevant portions of a document instead of sending the entire document to the LLM.

### 3. Embeddings

The project uses:

```text
sentence-transformers/all-MiniLM-L6-v2
```

to convert text into numerical vectors.

### 4. Vector Database

The embeddings are stored using:

```text
FAISS
```

FAISS allows the system to efficiently find document chunks that are semantically similar to a user's question.

### 5. Retrieval

For every question, the system searches the FAISS database and retrieves the most relevant document chunks.

### 6. Generation

The retrieved context is provided to the DeepSeek LLM along with the user's question.

The model then generates an answer based on the retrieved information.

## Example

Suppose `documents/sample.txt` contains:

```text
Python is a high-level programming language.
It is widely used in web development, data science,
machine learning, and automation.
```

The user asks:

```text
What is Python used for?
```

The RAG system retrieves the relevant section and sends it to DeepSeek.

The chatbot can respond:

```text
Python is used for web development, data science,
machine learning, and automation.
```

## Why RAG?

A normal LLM does not automatically know the contents of your private documents.

RAG solves this by:

```text
User Question
      ↓
Search Documents
      ↓
Retrieve Relevant Information
      ↓
Give Information to LLM
      ↓
Generate Answer
```

This allows an LLM to answer questions using a user's own knowledge base without requiring the model itself to be trained on those documents.

## Future Improvements

Possible improvements for this project include:

* [ ] Support DOCX and Markdown files
* [ ] Add conversation memory
* [ ] Add source citations to answers
* [ ] Improve chunking strategy
* [ ] Add metadata filtering
* [ ] Add authentication to the API
* [ ] Build a web-based chat interface
* [ ] Add streaming responses
* [ ] Add conversation history
* [ ] Use a persistent production vector database
* [ ] Add document upload through the API
* [ ] Deploy the application using Docker
* [ ] Add evaluation metrics for RAG quality

## Learning Objectives

This project demonstrates the fundamental concepts behind a RAG application:

* Document processing
* Text preprocessing
* Text chunking
* Embeddings
* Vector databases
* Similarity search
* Prompt construction
* LLM integration
* REST API development
* Environment variable management
* Building an end-to-end AI application

## License

This project is intended for educational and learning purposes.
