import torch
from transformers import BertForSequenceClassification, BertTokenizerFast
import pandas as pd

def load_model_and_tokenizer(model_path="saved_model"):
    """Load the saved model and tokenizer"""
    model = BertForSequenceClassification.from_pretrained(model_path)
    tokenizer = BertTokenizerFast.from_pretrained(model_path)
    
    # Move model to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()  # Set model to evaluation mode
    
    return model, tokenizer, device

def predict_text(text, model, tokenizer, device):
    """Make prediction for a single text input"""
    # Tokenize and prepare input
    inputs = tokenizer(
        text,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    ).to(device)
    
    # Get prediction
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        predicted_class = torch.argmax(probs, dim=1)
    
    return predicted_class.item(), probs[0].tolist()

def main():
    # Load model and tokenizer
    print("Loading model and tokenizer...")
    model, tokenizer, device = load_model_and_tokenizer()
    
    # Example texts for testing
    test_texts = [
        "I love you",
        "fuck you asshole!",
        "you are a black nigger",
        "muslim is a terrorist",
        "i hate jews",
        "Amey is a pookie"
    ]
    
    # Make predictions
    print("\nMaking predictions...")
    for text in test_texts:
        predicted_class, probabilities = predict_text(text, model, tokenizer, device)
        class_map = {
            0: 'not_cyberbullying', 
            1: 'gender', 
            2: 'religion', 
            3: 'other_cyberbullying', 
            4: 'age', 
            5: 'ethnicity'
        }

        print("\nText:", text)
        if predicted_class == 0:
            print("Not Cyberbullying")  
        else:
            print(f"Cyberbullying Detected.\nProbable Category: {class_map[predicted_class]}")
        print("Class probabilities:", probabilities)

if __name__ == "__main__":
    main() 