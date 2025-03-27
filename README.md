# Cyberbullying Detection Model

A BERT-based model for detecting cyberbullying in text content.

## Setup

1. Clone the repository:
```bash
git clone https://github.com/WWharf416/ThoughtShield
cd ThoughtShield
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On Linux/Mac
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Login to Weights & Biases (optional, for experiment tracking):
```bash
wandb login
```

## Dataset

Place your cyberbullying dataset (`cyberbullying_tweets.csv`) in the root directory. The dataset should contain:
- A 'tweet_text' column with the tweet content
- A 'cyberbullying_type' column with the cyberbullying categories

## Training

To train the model:
```bash
python train.py
```

Training configurations (in `train.py`):
- Model: BERT-base-uncased
- Learning rate: 2e-5
- Epochs: 50
- Batch size: 64
- FP16 training enabled
- Evaluation and checkpointing after each epoch

The model and tokenizer will be saved to `saved_model/` directory.

## Evaluation

To evaluate the model on new texts:
```bash
python evaluate.py
```

## Monitoring

Training progress can be monitored:
1. In the terminal output
2. In the generated log file
3. Through the Weights & Biases dashboard (if enabled)

## Model Performance

The model evaluates the following metrics:
- Accuracy
- F1 Score
- Precision
- Recall