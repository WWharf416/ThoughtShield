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
print("Dateset Loaded")

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
    output_dir='./results',          # output directory
    num_train_epochs=3,              # total number of training epochs
    per_device_train_batch_size=8,  # batch size per device during training
    per_device_eval_batch_size=20,   # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    load_best_model_at_end=True,     # load the best model when finished training (default metric is loss)
    # but you can specify `metric_for_best_model` argument to change to accuracy or other metric
    logging_steps=400,               # log & save weights each logging_steps
    save_steps=400,
    evaluation_strategy="steps",     # evaluate each `logging_steps`
)

trainer = Trainer(
    model=model,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)

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
    #evaluate_during_training=True,
    logging_dir='./logs'
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

print("Evaluating model...")
trainer.evaluate()

target_names = ['good','bad']
def get_prediction(text):
    # prepare our text into tokenized sequence
    inputs = tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors="pt").to("cuda")
    # perform inference to our model
    outputs = model(**inputs)
    # get output probabilities by doing softmax
    probs = outputs[0].softmax(1)
    # executing argmax function to get the candidate label
    return probs #probs.argmax() #target_names[probs.argmax()]

"""Test Model to Detect Cyberbullying Tweets"""

print("Testing model...")
text = """
girls just cant sing country as well as guys
"""
print("Text: ", text)
print("Prediction: ", get_prediction(text))

text = """
no love dumb dumb ! girls just cant sing country as well as guys
"""
print("Text: ", text)
print("Prediction: ", get_prediction(text))

text = """
bitch bitch !
"""
print("Text: ", text)
print("Prediction: ", get_prediction(text))

text = """
love for you !
"""
print("Text: ", text)
print("Prediction: ", get_prediction(text))
