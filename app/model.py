import torch as th
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer

class Model(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(Model, self).__init__()
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.output = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.3)
        self.sig = nn.Sigmoid()
        self.num_layers = num_layers



    def forward(self, x):
        out = self.embedding(x)
        out, hidden = self.lstm(out)
        out = self.dropout(out[:, -1, :])
        out = self.output(out)
        return out
    

device = th.device("cuda" if th.cuda.is_available() else "cpu")

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def predict(model : Model, sentence : str):
    encodings = tokenizer(sentence, truncation=True, padding=True, return_tensors="pt")
    input_ids = encodings['input_ids'].to(device)
    attention_mask = encodings['attention_mask'].to(device)
    
    with th.no_grad():
        logits = model(input_ids)

    probabilities = th.softmax(logits, dim=1).squeeze()
    predicted_label = probabilities.argmax().item()

    label_mapping = {0: "Negative", 1: "Positive"}
    predicted_sentiment = label_mapping[predicted_label]

    return predicted_sentiment, probabilities
