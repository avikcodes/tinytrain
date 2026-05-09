# TinyTrain 🔬

> Upload any CSV. Train a neural network. Watch it learn.

Most ML tools give you a final accuracy number and nothing else.
TinyTrain streams live training loss curve epoch by epoch so you can see exactly what your model is learning in real time.

![demo](https://raw.githubusercontent.com/avikcodes/tinytrain/main/demo.gif)
---

## The Problem

```
You train a model.
You wait 10 minutes.
You get: "Accuracy: 0.72"
You have no idea if it overfit.
You have no idea when it converged.
You have no idea if more epochs would help.
```

**TinyTrain shows you everything as it happens.**

---

## Features

- **Live loss streaming** — train and validation loss updates after every epoch via WebSockets
- **Live accuracy tracking** — watch accuracy improve in real time
- **Confusion matrix** — full breakdown of classification performance after training
- **Model download** — download your trained `.pkl` model file
- **AI training summary** — plain English explanation of training performance
- **Session history** — every past training run saved to Supabase
- **Redis caching** — repeated runs return instantly

---

## Neural Network Architecture

```
Input Layer  →  [input_size features]
                        ↓
              Linear(input_size → hidden_size)
                        ↓
                      ReLU
                        ↓
                   Dropout(0.3)
                        ↓
          Linear(hidden_size → hidden_size // 2)
                        ↓
                      ReLU
                        ↓
           Linear(hidden_size // 2 → output_size)
                        ↓
Output Layer →  [class_count outputs]
```

---

## Training Configuration

| Parameter | Options | Default |
|-----------|---------|---------|
| Epochs | 10 — 100 | 50 |
| Learning Rate | 0.001, 0.01, 0.1 | 0.001 |
| Hidden Size | 32, 64, 128 | 64 |
| Optimizer | Adam | Adam |
| Loss (Classification) | CrossEntropyLoss | CrossEntropyLoss |
| Loss (Regression) | MSELoss | MSELoss |
| Train/Val Split | 80/20 | 80/20 |

---

## Task Auto-Detection

TinyTrain automatically detects the task type from your target column:

| Condition | Task Type |
|-----------|-----------|
| Target has fewer than 20 unique values | Classification |
| Target has 20 or more unique values | Regression |

---

## Data Preprocessing Pipeline

```
Raw CSV
    ↓
Drop rows where target column is NaN
    ↓
Drop columns with more than 50% missing values
    ↓
Fill missing numeric values with column median
    ↓
Fill missing categorical values with column mode
    ↓
Label encode all categorical feature columns
    ↓
Label encode target column (classification only)
    ↓
Apply StandardScaler on all features
    ↓
Convert to float32 PyTorch tensors
    ↓
Split 80% train / 20% validation
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | Next.js 14 + TypeScript |
| Styling | Tailwind CSS |
| Charts | Recharts |
| Backend | Python FastAPI |
| Server | Uvicorn |
| Neural Network | PyTorch |
| Data Processing | Pandas + NumPy |
| ML Utilities | scikit-learn |
| AI Summary | Groq llama-3.1-8b-instant |
| Database | Supabase (PostgreSQL) |
| Cache | Upstash Redis |
| Realtime | WebSockets |

---

## How It Works

1. Upload any CSV file with a target column
2. Select target column and configure training parameters
3. Frontend reads file and encodes as base64
4. Opens WebSocket connection to Python backend
5. Backend preprocesses data and builds TinyNet model
6. Training starts — after every epoch backend sends loss and accuracy via WebSocket
7. Frontend receives epoch messages and updates live loss chart in real time
8. After training completes backend computes confusion matrix
9. Generates AI plain English training summary via Groq
10. Saves session to Supabase and caches in Redis
11. Frontend renders full results dashboard with model download

---

## WebSocket Message Flow

```
Frontend                          Backend
   |                                 |
   |-- {filename, data, config} ---> |
   |                                 |
   | <-- {type: "progress", 5%} ---- |  Parsing CSV
   | <-- {type: "progress", 10%} --- |  Preprocessing
   | <-- {type: "progress", 15%} --- |  Building model
   | <-- {type: "progress", 20%} --- |  Starting training
   |                                 |
   | <-- {type: "epoch", epoch:1} -- |  Epoch 1 complete
   | <-- {type: "epoch", epoch:2} -- |  Epoch 2 complete
   |          ... x epochs ...       |
   | <-- {type: "epoch", epoch:N} -- |  Final epoch
   |                                 |
   | <-- {type: "progress", 91%} --- |  Confusion matrix
   | <-- {type: "progress", 94%} --- |  Generating summary
   | <-- {type: "progress", 97%} --- |  Saving to history
   |                                 |
   | <-- {type: "complete", 100%} -- |  Full results
   |                                 |
```

---

## Project Structure

```
TinyTrain/
├── app/
│   ├── page.tsx              ← Full UI + WebSocket client + live charts
│   ├── layout.tsx
│   └── globals.css
├── tinytrain-api/
│   ├── main.py               ← FastAPI + WebSocket server
│   │   ├── preprocess_data()     ← Data cleaning + encoding + scaling
│   │   ├── TinyNet               ← PyTorch neural network class
│   │   ├── train_model()         ← Async training loop with WebSocket streaming
│   │   ├── get_confusion_matrix() ← Post-training evaluation
│   │   ├── generate_summary()    ← Groq AI report
│   │   ├── get_cache()           ← Redis read
│   │   ├── set_cache()           ← Redis write
│   │   ├── save_to_supabase()    ← History storage
│   │   ├── /ws/train             ← WebSocket training endpoint
│   │   ├── /history              ← GET past sessions
│   │   └── /health               ← Health check
│   ├── requirements.txt
│   └── .env
├── .env.local
└── README.md
```

---

## API Reference

### WebSocket `/ws/train`

**Send:**

```json
{
  "filename": "titanic.csv",
  "data": "base64_encoded_content",
  "target_column": "Survived",
  "epochs": 50,
  "learning_rate": 0.001,
  "hidden_size": 64
}
```

**Receive (epoch update):**

```json
{
  "type": "epoch",
  "epoch": 23,
  "total_epochs": 50,
  "train_loss": 0.4821,
  "val_loss": 0.5103,
  "val_accuracy": 0.7842,
  "progress": 52
}
```

**Receive (complete):**

```json
{
  "type": "complete",
  "progress": 100,
  "session_id": "uuid",
  "final_accuracy": 0.8134,
  "final_loss": 0.4203,
  "training_time": 12.4,
  "training_history": [...],
  "confusion_matrix": [[120, 18], [22, 51]],
  "model_base64": "base64_encoded_pkl",
  "task_type": "classification",
  "ai_summary": "The model converged well..."
}
```

### GET `/history`

Returns last 10 training sessions from Supabase.

### GET `/health`

```json
{"status": "ok"}
```

---

## Database Schema

```sql
create table training_sessions (
  id uuid default gen_random_uuid() primary key,
  session_id text not null,
  file_name text not null,
  target_column text not null,
  task_type text not null,
  epochs int,
  final_accuracy float,
  final_loss float,
  training_history jsonb,
  ai_summary text,
  created_at timestamp default now()
);
```

---

## Getting Started

### Prerequisites

- Node.js 18+
- Python 3.10 to 3.13
- Groq API key — free at console.groq.com
- Supabase project — free at supabase.com
- Upstash Redis — free at upstash.com

### Installation

```
git clone https://github.com/avikcodes/TinyTrain
cd TinyTrain
npm install
cd tinytrain-api
pip install -r requirements.txt
```

### Environment Variables

Create `tinytrain-api/.env`:

```
GROQ_API_KEY=your_key
SUPABASE_URL=your_url
SUPABASE_KEY=your_anon_key
UPSTASH_REDIS_REST_URL=your_url
UPSTASH_REDIS_REST_TOKEN=your_token
```

Create `.env.local` in root:

```
NEXT_PUBLIC_SUPABASE_URL=your_url
NEXT_PUBLIC_SUPABASE_ANON_KEY=your_anon_key
```

### Run

Terminal 1:

```
cd tinytrain-api
uvicorn main:app --reload
```

Terminal 2:

```
npm run dev
```

Open `http://localhost:3000`

---

## Example — Titanic Dataset

```
File: titanic.csv
Target: Survived
Task: Classification (auto-detected)
Epochs: 50
Learning Rate: 0.001
Hidden Size: 64

Results:
Final Accuracy:  81.3%
Final Loss:      0.4203
Training Time:   12.4 seconds

Confusion Matrix:
              Predicted 0    Predicted 1
Actual 0          120             18
Actual 1           22             51

AI Summary:
The model achieved 81% validation accuracy on the Titanic dataset.
Training and validation loss curves converged smoothly with no
signs of severe overfitting. Consider increasing hidden size to
128 or adding more epochs to push accuracy higher.
```

---

## Roadmap

- [x] Live loss streaming via WebSockets
- [x] Auto task type detection
- [x] Confusion matrix
- [x] Model download as .pkl
- [x] AI training summary
- [x] Supabase session history
- [x] Redis caching
- [ ] Early stopping support
- [ ] Learning rate scheduler
- [ ] Multiple architecture options
- [ ] GPU support
- [ ] Export to ONNX format
- [ ] Batch size configuration

---

## Part of 30 Projects

This is **Project 7 of 30** in my open-source build sprint — building 30 open-source AI and ML tools from March to December 2026.

Follow on X: [@Avikzx]https://x.com/Avikzx)

All projects: [github.com/avikcodes](https://github.com/avikcodes)

---

## License

MIT
