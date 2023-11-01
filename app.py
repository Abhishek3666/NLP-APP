from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()

class InputData(BaseModel):
    question: str
    quora_title: str

# Load your machine learning model
model = joblib.load("model.pkl")

# Vectorizer for transforming text data to numerical vectors
vectorizer = TfidfVectorizer()

@app.post("/predict")
def predict(data: InputData):
    # Transform input text data into vectors
    question_vector = vectorizer.transform([data.question])
    title_vector = vectorizer.transform([data.quora_title])

    # Calculate cosine similarity
    similarity_score = cosine_similarity(question_vector, title_vector)[0][0]

    return {"similarity_score": similarity_score}


