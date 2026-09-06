# Import os to work with files and directories
import os

# Load environment variables from the .env file
from dotenv import load_dotenv

# Import document loaders for TXT and PDF files
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader
)

# Import the text splitter used to divide documents into smaller chunks
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Import the HuggingFace embedding model
from langchain_huggingface import HuggingFaceEmbeddings

# Import FAISS, which will be used as our vector database
from langchain_community.vectorstores import FAISS


# Load variables from .env
load_dotenv()


# Folder containing our source documents
DOCUMENT_FOLDER = "documents"

# Folder where the FAISS vector database will be saved
VECTORSTORE_FOLDER = "vectorstore"


# Function responsible for loading all supported documents
def load_documents():

    # Create an empty list to store loaded documents
    documents = []

    # Go through every file inside the documents folder
    for filename in os.listdir(DOCUMENT_FOLDER):

        # Create the complete path of the file
        filepath = os.path.join(
            DOCUMENT_FOLDER,
            filename
        )

        # If the file is a text file, use TextLoader
        if filename.endswith(".txt"):

            loader = TextLoader(filepath)

            # Load the document and add it to our list
            documents.extend(loader.load())

        # If the file is a PDF, use PyPDFLoader
        elif filename.endswith(".pdf"):

            loader = PyPDFLoader(filepath)

            # Load the PDF pages and add them to our list
            documents.extend(loader.load())

    # Return all loaded documents
    return documents


# Function responsible for creating the vector database
def create_vector_database():

    print("Loading documents...")

    # Load documents from the documents folder
    documents = load_documents()

    print(f"Loaded {len(documents)} documents")

    # Create a text splitter
    # chunk_size controls the approximate size of each chunk
    # chunk_overlap keeps some text from the previous chunk
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    # Split the documents into smaller chunks
    chunks = splitter.split_documents(documents)

    print(f"Created {len(chunks)} chunks")

    # Create an embedding model
    # The model converts text into numerical vectors
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    print("Creating vector database...")

    # Convert all document chunks into embeddings
    # and store them inside FAISS
    vectorstore = FAISS.from_documents(
        chunks,
        embeddings
    )

    # Save the FAISS database locally
    vectorstore.save_local(
        VECTORSTORE_FOLDER
    )

    print("Vector database created successfully!")


# Run this code only when ingest.py is executed directly
if __name__ == "__main__":

    # Start the document ingestion process
    create_vector_database()