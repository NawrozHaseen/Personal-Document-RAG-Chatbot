# Import os so we can read environment variables
import os

# Load environment variables from the .env file
from dotenv import load_dotenv

# Import the HuggingFace embedding model
from langchain_huggingface import HuggingFaceEmbeddings

# Import FAISS vector database
from langchain_community.vectorstores import FAISS

# Import the OpenAI chat model
from langchain_openai import ChatOpenAI

# Import LangChain's prompt template
from langchain_core.prompts import ChatPromptTemplate


# Load environment variables
# This allows the application to access OPENAI_API_KEY
load_dotenv()


# Location of the saved FAISS vector database
VECTORSTORE_FOLDER = "vectorstore"


# Function to load the existing vector database
def load_vectorstore():

    # Use the same embedding model that was used
    # during document ingestion
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Load the previously saved FAISS database
    vectorstore = FAISS.load_local(
        VECTORSTORE_FOLDER,
        embeddings,
        allow_dangerous_deserialization=True
    )

    # Return the vector database
    return vectorstore


# Main RAG question-answering function
def ask_question(question):

    # Load our vector database
    vectorstore = load_vectorstore()

    # Convert the vector database into a retriever
    # k=3 means we retrieve the 3 most relevant chunks
    retriever = vectorstore.as_retriever(
        search_kwargs={
            "k": 3
        }
    )

    # Search the vector database using the user's question
    # The retriever finds semantically similar chunks
    documents = retriever.invoke(question)

    # Combine all retrieved chunks into one context string
    context = "\n\n".join(
        document.page_content
        for document in documents
    )

    # Create the prompt that will be sent to the LLM
    prompt = ChatPromptTemplate.from_template(
        """
        You are a helpful AI assistant.

        Answer the user's question using ONLY
        the provided context.

        If the answer cannot be found in the context,
        say "I don't know based on the provided documents."

        Context:
        {context}

        Question:
        {question}

        Answer:
        """
    )

    # Create the DeepSeek language model
    # DeepSeek provides an OpenAI-compatible API
    # Create the DeepSeek language model
    llm = ChatOpenAI(
        model="deepseek-v4-flash",
        temperature=0,
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com"
    )
    
    # Connect the prompt to the LLM
    chain = prompt | llm

    # Send the retrieved context and user question
    # to the language model
    response = chain.invoke({
        "context": context,
        "question": question
    })

    # Return only the generated text
    return response.content


# Run this section only when rag.py is executed directly
if __name__ == "__main__":

    # Keep asking questions until the user types "exit"
    while True:

        # Take a question from the terminal
        question = input("\nAsk a question: ")

        # Stop the program if the user types exit
        if question.lower() == "exit":
            break

        # Send the question through the RAG pipeline
        answer = ask_question(question)

        # Display the generated answer
        print("\nAnswer:")
        print(answer)