import numpy as np
import pandas as pd
import torch
from transformers import BertForSequenceClassification, BertTokenizerFast, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from dataloader_prep import prepare_dataloader

print("Loading model...")
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
print("Model loaded")

print("Loading dataset...")
train_loader, test_loader, train_dataset, test_dataset = prepare_dataloader("cleaned_cyberbullying_dataset.csv")
print("Dataset Loaded")

def tokenize(batch):
    return tokenizer(batch['text'], padding=True, truncation=False)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
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
    num_train_epochs=3,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    fp16=True,
    warmup_steps=30,
    logging_steps=20,
    weight_decay=0.01,
    save_strategy="epoch",  # Save at the end of each epoch
    logging_dir='./logs',
    load_best_model_at_end=True
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

target_names = ['good', 'bad']

def get_prediction(text):
    inputs = tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors="pt").to("cuda")
    outputs = model(**inputs)
    probs = outputs[0].softmax(1)
    return probs

print("Testing model...")
test_texts = [
    "girls just cant sing country as well as guys",
    "no love dumb dumb ! girls just cant sing country as well as guys",
    "bitch bitch !",
    "love for you !"
]

for text in test_texts:
    print("Text: ", text)
    print("Prediction: ", get_prediction(text))
