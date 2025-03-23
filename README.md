# Sentiment Analysis API

This project implements a FastAPI-based REST API for sentiment analysis using a LSTM neural network model with BERT tokenization.

## Overview

The API provides sentiment analysis predictions for text input, classifying text as either "Positive" or "Negative" with associated probabilities.

## Technology Stack

- Python 3.11
- FastAPI
- PyTorch
- Transformers (BERT)
- Docker

## Project Structure

```
.
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── model.py
│   └── model/
│       └── model.pth
├── Dockerfile
├── README.md
└── requirements.txt
```

## Getting Started

### Using Docker

1. Build the Docker image:
```bash
docker build -t sentiment-api .
```

2. Run the container:
```bash
docker run -d -p 80:80 sentiment-api
```

### API Endpoints

- `GET /`: Health check endpoint
- `POST /predict`: Sentiment analysis endpoint
    - Input: JSON with `text` field
    - Output: Prediction with sentiment class and probabilities

### Example Usage

```bash
curl -X POST "http://localhost:80/predict" \
         -H "Content-Type: application/json" \
         -d '{"text": "I love this product!"}'
```

Response:
```json
{
        "prediction_class": "Positive",
        "positive_probability": 0.92,
        "negative_probability": 0.08
}
```

## Requirements

See `requirements.txt` for complete list of dependencies.

## License

This project is open source and available under the MIT License.