# Import FastAPI to create our API
from fastapi import FastAPI

# Import BaseModel to validate API request data
from pydantic import BaseModel

# Import our RAG function
from rag import ask_question


# Create the FastAPI application
app = FastAPI(
    title="Mini RAG API",
    description="A simple Retrieval Augmented Generation API"
)


# Define the structure of the incoming question
class QuestionRequest(BaseModel):

    # The user must provide a question as a string
    question: str


# Define the structure of the API response
class QuestionResponse(BaseModel):

    # Return the original question
    question: str

    # Return the generated answer
    answer: str


# Simple GET endpoint to check whether the API is running
@app.get("/")
def home():

    # Return a JSON response
    return {
        "message": "Mini RAG API is running"
    }


# POST endpoint for asking questions
@app.post("/ask", response_model=QuestionResponse)
def ask(request: QuestionRequest):

    # Send the user's question to our RAG pipeline
    answer = ask_question(
        request.question
    )

    # Return both the question and generated answer
    return {
        "question": request.question,
        "answer": answer
    }