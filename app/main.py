import sys
from app import model as app_model
sys.modules["model"] = app_model

from typing import Union
from pydantic import BaseModel
from fastapi import FastAPI
import torch as th
from .model import predict, device

# Define the input model
class Text(BaseModel):
    text: str

# Define the output model
class Prediction(BaseModel):
    prediction_class: str
    positive_probability: float
    negative_probability: float 


app = FastAPI()

model = th.load("./app/model/model.pth", map_location=device, weights_only=False)
model.eval()


@app.get("/")
def read_root():
    return {"Hello": "World"}



@app.post("/predict", response_model=Prediction)
def predict_sentiment(text: Text):
    sentiment_class, probs = predict(model, text.text)
    return Prediction(
        prediction_class=sentiment_class,
        positive_probability=probs[1].item(),
        negative_probability=probs[0].item()
    )