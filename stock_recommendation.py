# src/data_ingestion.py
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import pickle
from feature_engineering import create_features
from data_ingestion import fetch_data

class StockDataset(Dataset):
    def __init__(self, data, seq_len=10):
        self.data = data
        self.seq_len = seq_len
    def __len__(self):
        return len(self.data) - self.seq_len
    def __getitem__(self, idx):
        return (torch.tensor(self.data[idx:idx+self.seq_len], dtype=torch.float),
                torch.tensor(self.data[idx+self.seq_len], dtype=torch.float))

class LSTMModel(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=50, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        out = self.fc(hn[-1])
        return out

# Prepare data
df = fetch_data('AAPL')
df_feat = create_features(df)
data = df_feat[['Close']].values

train_size = int(len(data)*0.8)
train_data = data[:train_size]
test_data = data[train_size:]
train_dataset = StockDataset(train_data)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

model = LSTMModel()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(10):
    for X_batch, y_batch in train_loader:
        X_batch = X_batch.unsqueeze(-1)
        optimizer.zero_grad()
        y_pred = model(X_batch)
        loss = criterion(y_pred.squeeze(), y_batch)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item()}")

with open('stock_lstm_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# src/recommendation.py
import pickle
import torch

def generate_signal(last_price, predicted_price, threshold=0.01):
    change = (predicted_price - last_price)/last_price
    if change > threshold:
        return 'BUY'
    elif change < -threshold:
        return 'SELL'
    else:
        return 'HOLD'

# src/predict_api.py
from fastapi import FastAPI
import pickle
import torch
from feature_engineering import create_features
from data_ingestion import fetch_data
from recommendation import generate_signal

app = FastAPI()

with open('stock_lstm_model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.get('/predict_price')
def predict_price(ticker: str = 'AAPL'):
    df = fetch_data(ticker)
    df_feat = create_features(df)
    last_seq = torch.tensor(df_feat['Close'].values[-10:], dtype=torch.float).unsqueeze(0).unsqueeze(-1)
    pred = model(last_seq).item()
    return {'predicted_price': pred}

@app.get('/recommendation')
def recommendation(ticker: str = 'AAPL'):
    df = fetch_data(ticker)
    df_feat = create_features(df)
    last_price = df_feat['Close'].values[-1]
    last_seq = torch.tensor(df_feat['Close'].values[-10:], dtype=torch.float).unsqueeze(0).unsqueeze(-1)
    pred_price = model(last_seq).item()
    signal = generate_signal(last_price, pred_price)
    return {'recommendation': signal, 'predicted_price': pred_price}

# dashboard/app.py
import streamlit as st
import requests

ticker = st.text_input('Enter Ticker', 'AAPL')
if st.button('Predict'):
    pred = requests.get(f'http://localhost:8000/predict_price?ticker={ticker}').json()
    rec = requests.get(f'http://localhost:8000/recommendation?ticker={ticker}').json()
    st.write('Predicted Price:', pred['predicted_price'])
    st.write('Recommendation:', rec['recommendation'])

# requirements.txt
fastapi
uvicorn
pandas
scikit-learn
torch
yfinance
streamlit
pickle5

# Dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src/ ./src
COPY dashboard/ ./dashboard
EXPOSE 8501
CMD ["streamlit", "run", "dashboard/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
