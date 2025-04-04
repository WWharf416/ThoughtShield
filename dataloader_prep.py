# -*- coding: utf-8 -*-
"""dataloader_prep.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1CESOrE3WyhIxd2nhMiZWjbxxYXVdDIHb
"""



import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('punkt')

def prepare_dataloader(file_path, max_words=20000, max_len=100, batch_size=32, test_size=0.1):
    """
    Function to load dataset, preprocess text, tokenize for LSTM & Transformer models,
    and return DataLoaders for training and testing.

    Parameters:
    file_path (str): Path to the CSV dataset.
    max_words (int): Maximum words for LSTM tokenizer.
    max_len (int): Maximum sequence length for tokenization.
    batch_size (int): Batch size for DataLoader.
    test_size (float): Fraction of data to be used for testing.

    Returns:
    train_loader (DataLoader): DataLoader for training.
    test_loader (DataLoader): DataLoader for testing.
    train_dataset (Dataset): Custom dataset for training.
    test_dataset (Dataset): Custom dataset for testing.
    num_labels (int): Number of unique labels in the dataset.
    label_mapping (dict): Mapping from label names to numeric labels.
    """
    # Load Dataset
    df = pd.read_csv(file_path)
    
    # Create label mapping for cyberbullying types
    label_mapping = {label: idx for idx, label in enumerate(df['cyberbullying_type'].unique())}
    num_labels = len(label_mapping)
    print("Label mapping:", label_mapping)
    
    # Convert cyberbullying_type to numeric labels
    df['label'] = df['cyberbullying_type'].map(label_mapping)
    print("Label distribution:\n", df['label'].value_counts())

    # Temporarily reduce dataset size for development
    df = df.sample(n=1000, random_state=42)  # Using 1000 samples for quick testing
    print("Dataset Loaded\n", df.head())

    # Drop missing values
    df.dropna(inplace=True)

    # Text Preprocessing
    def clean_text(text):
        text = text.lower()
        text = re.sub(r'http\S+|www\S+', '', text)  # Remove URLs
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove special characters
        text = ' '.join([word for word in text.split() if word not in stopwords.words('english')])
        return text

    df['cleaned_text'] = df['tweet_text'].apply(clean_text)  # Changed from 'text' to 'tweet_text'

    # Splitting Dataset
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        df['cleaned_text'], df['label'], test_size=test_size, random_state=42)

    # Tokenization & Padding for LSTM
    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(train_texts)

    X_train = pad_sequences(tokenizer.texts_to_sequences(train_texts), maxlen=max_len)
    X_test = pad_sequences(tokenizer.texts_to_sequences(test_texts), maxlen=max_len)
    y_train = np.array(train_labels)
    y_test = np.array(test_labels)

    # Tokenization for Transformer Model (BERT)
    tokenizer_bert = AutoTokenizer.from_pretrained("bert-base-uncased")
    def tokenize_bert(text_list):
        return tokenizer_bert(text_list, padding=True, truncation=True, max_length=max_len, return_tensors="pt")

    train_encodings = tokenize_bert(train_texts.tolist())
    test_encodings = tokenize_bert(test_texts.tolist())

    # Custom Dataset Class
    class CyberbullyingDataset(Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            item = {key: val[idx].clone().detach() for key, val in self.encodings.items()}
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
            return item

    train_dataset = CyberbullyingDataset(train_encodings, y_train)
    test_dataset = CyberbullyingDataset(test_encodings, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, train_dataset, test_dataset, num_labels, label_mapping

# Example usage
if __name__ == "__main__":
    train_loader, test_loader, train_dataset, test_dataset, num_labels, label_mapping = prepare_dataloader("cyberbullying_tweets.csv")
    print("Dataloader is ready!")

