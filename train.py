import numpy as np
import pandas as pd
import torch
from transformers import BertForSequenceClassification, BertTokenizerFast, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from dataloader_prep import prepare_dataloader
import wandb

# Initialize wandb
wandb.init(
    project="cyberbullying",  # Choose your project name
    config={
        "model": "bert-base-uncased",
        "learning_rate": 2e-5,
        "epochs": 50,
        "batch_size": 64,
        "warmup_steps": 30,
        "weight_decay": 0.01
    }
)

print("Loading dataset...")
train_loader, test_loader, train_dataset, test_dataset, num_labels, label_mapping = prepare_dataloader("cyberbullying_tweets.csv")
print("Dataset Loaded in main")

print("Loading model...")
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_labels)
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
print("Model loaded")

def tokenize(batch):
    return tokenizer(batch['text'], padding=True, truncation=False)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

training_args = TrainingArguments(
    output_dir='./results',
    learning_rate=2e-5,
    num_train_epochs=50,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    warmup_steps=30,
    logging_steps=20,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=1,
    logging_dir='./logs',
    load_best_model_at_end=True,
    report_to="wandb",
    run_name=wandb.run.name
)

trainer = Trainer(
    model=model,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)

print("Training model...")
trainer.train()

# Save the trained model and tokenizer
model.save_pretrained("saved_model")
tokenizer.save_pretrained("saved_model")
print("Model saved successfully.")

print("Evaluating model...")
trainer.evaluate()

def get_prediction(text):
    inputs = tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors="pt").to("cuda")
    outputs = model(**inputs)
    probs = outputs[0].softmax(1)
    reverse_mapping = {v: k for k, v in label_mapping.items()}
    predicted_label = reverse_mapping[probs.argmax().item()]
    return predicted_label, probs

print("Testing model...")
test_texts = [
    "the women race is the stupidest race in the world",
    "i respect the women race",
    "you are dumb because you are indian",
    "fuck you asshole!"
]

for text in test_texts:
    print("Text:", text)
    label, probs = get_prediction(text)
    print("Predicted category:", label)
    print("Probabilities:", probs)
    print()

# After training is complete
wandb.finish()
