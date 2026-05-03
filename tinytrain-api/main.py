import asyncio
import base64
import hashlib
import io
import json
import os
import pickle
import time
import traceback
import uuid

import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
import torch.optim as optim
from dotenv import load_dotenv
from fastapi import FastAPI, Response, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_cache(cache_key):
    try:
        response = requests.get(
            f"{os.getenv('UPSTASH_REDIS_REST_URL')}/get/{cache_key}",
            headers={
                "Authorization": f"Bearer {os.getenv('UPSTASH_REDIS_REST_TOKEN')}",
            },
            timeout=15,
        )
        result = response.json().get("result")
        if result is not None:
            return json.loads(result)
        return None
    except Exception:
        return None


def set_cache(cache_key, data):
    try:
        requests.post(
            f"{os.getenv('UPSTASH_REDIS_REST_URL')}/set/{cache_key}",
            headers={
                "Authorization": f"Bearer {os.getenv('UPSTASH_REDIS_REST_TOKEN')}",
            },
            json={"value": json.dumps(data), "ex": 86400},
            timeout=15,
        )
    except Exception as error:
        print(error)


def save_to_supabase(
    session_id,
    file_name,
    target_column,
    task_type,
    epochs,
    final_accuracy,
    final_loss,
    training_history,
    ai_summary,
):
    response = requests.post(
        f"{os.getenv('SUPABASE_URL')}/rest/v1/training_sessions",
        headers={
            "apikey": os.getenv("SUPABASE_ANON_KEY", ""),
            "Authorization": f"Bearer {os.getenv('SUPABASE_ANON_KEY', '')}",
            "Content-Type": "application/json",
            "Prefer": "return=minimal",
        },
        json={
            "session_id": session_id,
            "file_name": file_name,
            "target_column": target_column,
            "task_type": task_type,
            "epochs": epochs,
            "final_accuracy": final_accuracy,
            "final_loss": final_loss,
            "training_history": json.dumps(training_history),
            "ai_summary": ai_summary,
        },
        timeout=20,
    )
    print(response.status_code)
    if response.status_code != 201:
        print(response.text)


@app.get("/history")
def get_history():
    try:
        response = requests.get(
            (
                f"{os.getenv('SUPABASE_URL')}/rest/v1/training_sessions"
                "?select=*&order=created_at.desc&limit=10"
            ),
            headers={
                "apikey": os.getenv("SUPABASE_ANON_KEY", ""),
                "Authorization": f"Bearer {os.getenv('SUPABASE_ANON_KEY', '')}",
            },
            timeout=20,
        )
        response.raise_for_status()
        return response.json()
    except Exception:
        return []


def preprocess_data(df, target_column):
    df = df.dropna(subset=[target_column]).copy()
    unique_count = df[target_column].nunique()
    task_type = "classification" if unique_count < 20 else "regression"

    missing_ratio = df.isnull().mean()
    df = df.loc[:, missing_ratio <= 0.5].copy()
    if target_column not in df.columns:
        raise ValueError("Target column was removed during preprocessing due to missing values.")

    for column in df.columns:
        if df[column].isnull().any():
            if pd.api.types.is_numeric_dtype(df[column]):
                df[column] = df[column].fillna(df[column].median())
            else:
                mode = df[column].mode(dropna=True)
                fill_value = mode.iloc[0] if not mode.empty else "unknown"
                df[column] = df[column].fillna(fill_value)

    X = df.drop(columns=[target_column]).copy()
    y = df[target_column].copy()

    for column in X.select_dtypes(include=["object", "category", "bool"]).columns:
        encoder = LabelEncoder()
        X[column] = encoder.fit_transform(X[column].astype(str))

    label_encoder = None
    class_count = 1

    if task_type == "classification":
        label_encoder = LabelEncoder()
        y_array = label_encoder.fit_transform(y.astype(str)).astype(np.int64)
        class_count = len(label_encoder.classes_)
    else:
        y_array = pd.to_numeric(y, errors="coerce")
        y_median = y_array.median()
        y_array = y_array.fillna(y_median).to_numpy(dtype=np.float32)

    scaler = StandardScaler()
    X_array = scaler.fit_transform(X).astype(np.float32)

    feature_count = X_array.shape[1]
    return X_array, y_array, feature_count, class_count, task_type, label_encoder


class TinyNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, task_type):
        super().__init__()
        self.task_type = task_type
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size // 2, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x


async def train_model(X, y, task_type, epochs, learning_rate, hidden_size, websocket, class_count):
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)

    if task_type == "classification":
        y_train_tensor = torch.tensor(y_train, dtype=torch.long)
        y_val_tensor = torch.tensor(y_val, dtype=torch.long)
        output_size = class_count
    else:
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
        y_val_tensor = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)
        output_size = 1

    model = TinyNet(
        input_size=X.shape[1],
        hidden_size=hidden_size,
        output_size=output_size,
        task_type=task_type,
    )

    criterion = nn.CrossEntropyLoss() if task_type == "classification" else nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    training_history = []
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        train_outputs = model(X_train_tensor)
        train_loss = criterion(train_outputs, y_train_tensor)
        train_loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_tensor)
            val_loss = criterion(val_outputs, y_val_tensor)
            if task_type == "classification":
                predictions = torch.argmax(val_outputs, dim=1).cpu().numpy()
                val_accuracy = accuracy_score(y_val, predictions)
            else:
                val_accuracy = 0

        history_item = {
            "epoch": epoch,
            "train_loss": round(float(train_loss.item()), 4),
            "val_loss": round(float(val_loss.item()), 4),
            "val_accuracy": round(float(val_accuracy), 4),
        }
        training_history.append(history_item)

        progress = 20 + int((epoch / epochs) * 70)
        await websocket.send_json(
            {
                "type": "epoch",
                "epoch": epoch,
                "total_epochs": epochs,
                "train_loss": float(train_loss.item()),
                "val_loss": float(val_loss.item()),
                "val_accuracy": float(val_accuracy),
                "progress": progress,
            }
        )
        await asyncio.sleep(0)

    training_time = round(time.time() - start_time, 2)
    final_accuracy = training_history[-1]["val_accuracy"]
    final_loss = training_history[-1]["val_loss"]
    model_bytes = pickle.dumps(model)
    model_base64 = base64.b64encode(model_bytes).decode("utf-8")

    return (
        model,
        training_history,
        final_accuracy,
        final_loss,
        training_time,
        model_base64,
        X_val_tensor,
        y_val,
    )


def get_confusion_matrix(model, X_val_tensor, y_val):
    try:
        model.eval()
        with torch.no_grad():
            predictions = model(X_val_tensor)
            predicted_labels = torch.argmax(predictions, dim=1).cpu().numpy()
        matrix = confusion_matrix(y_val, predicted_labels)
        return matrix.tolist()
    except Exception:
        return []


def generate_summary(final_accuracy, final_loss, epochs, task_type, file_name):
    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {os.getenv('GROQ_API_KEY')}",
                "Content-Type": "application/json",
            },
            json={
                "model": "llama-3.1-8b-instant",
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are an ML training expert. Given neural network training "
                            "results write a plain English summary of the training "
                            "performance and suggestions for improvement. Under 100 words."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"File: {file_name} Task: {task_type} Final accuracy: "
                            f"{final_accuracy} Final loss: {final_loss} Epochs: {epochs}"
                        ),
                    },
                ],
            },
            timeout=30,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except Exception:
        return "Summary unavailable"


@app.websocket("/ws/train")
async def websocket_train(websocket: WebSocket):
    await websocket.accept()
    print("WebSocket connected")

    try:
        payload = await websocket.receive_json()
        filename = payload.get("filename", "dataset.csv")
        encoded_data = payload.get("data", "")
        target_column = payload.get("target_column")
        epochs = int(payload.get("epochs", 50))
        learning_rate = float(payload.get("learning_rate", 0.001))
        hidden_size = int(payload.get("hidden_size", 64))
        print("Data received")

        if not target_column:
            raise ValueError("target_column is required")

        file_bytes = base64.b64decode(encoded_data)

        await websocket.send_json(
            {"type": "progress", "step": "Parsing CSV...", "progress": 5}
        )
        df = pd.read_csv(io.BytesIO(file_bytes))

        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in CSV")

        await websocket.send_json(
            {"type": "progress", "step": "Preprocessing data...", "progress": 10}
        )
        X, y, feature_count, class_count, task_type, label_encoder = preprocess_data(
            df, target_column
        )

        await websocket.send_json(
            {"type": "progress", "step": "Building neural network...", "progress": 15}
        )
        _ = feature_count
        _ = label_encoder

        await websocket.send_json(
            {"type": "progress", "step": "Starting training...", "progress": 20}
        )
        (
            model,
            training_history,
            final_accuracy,
            final_loss,
            training_time,
            model_base64,
            X_val_tensor,
            y_val,
        ) = await train_model(
            X,
            y,
            task_type,
            epochs,
            learning_rate,
            hidden_size,
            websocket,
            class_count,
        )

        await websocket.send_json(
            {
                "type": "progress",
                "step": "Computing confusion matrix...",
                "progress": 91,
            }
        )
        confusion_matrix_list = (
            get_confusion_matrix(model, X_val_tensor, y_val)
            if task_type == "classification"
            else []
        )

        await websocket.send_json(
            {"type": "progress", "step": "Generating summary...", "progress": 94}
        )
        summary = generate_summary(final_accuracy, final_loss, epochs, task_type, filename)

        await websocket.send_json(
            {"type": "progress", "step": "Saving to history...", "progress": 97}
        )
        session_id = str(uuid.uuid4())
        save_to_supabase(
            session_id,
            filename,
            target_column,
            task_type,
            epochs,
            final_accuracy,
            final_loss,
            training_history,
            summary,
        )

        await websocket.send_json(
            {
                "type": "complete",
                "progress": 100,
                "session_id": session_id,
                "final_accuracy": final_accuracy,
                "final_loss": final_loss,
                "training_time": training_time,
                "training_history": training_history,
                "confusion_matrix": confusion_matrix_list,
                "model_base64": model_base64,
                "task_type": task_type,
                "ai_summary": summary,
            }
        )
    except WebSocketDisconnect:
        pass
    except Exception as error:
        traceback.print_exc()
        try:
            await websocket.send_json({"type": "error", "error": str(error)})
        except Exception:
            pass


@app.get("/health")
def health_check():
    return {"status": "ok"}
