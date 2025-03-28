from Text_Extraction_Final import TextExtractor
from evaluate import predict_text
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

def main():
    # Initialize models
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForSequenceClassification.from_pretrained("saved_model").to(device)
    tokenizer = AutoTokenizer.from_pretrained("saved_model")
    text_extractor = TextExtractor()

    while True:
        # Get file path from user
        file_path = input("\nEnter file path (or 'q' to quit): ")
        if file_path.lower() == 'q':
            break

        # Extract text from file
        extracted_text = text_extractor.extract_text(file_path)
        if extracted_text.startswith("Error"):
            print(extracted_text)
            continue

        print("\nExtracted text:", extracted_text)

        predicted_class, probabilities = predict_text(extracted_text, model, tokenizer, device)
        class_map = {
            0: 'not_cyberbullying', 
            1: 'gender', 
            2: 'religion', 
            3: 'other_cyberbullying', 
            4: 'age', 
            5: 'ethnicity'
        }

        if predicted_class == 0:
            print("Not Cyberbullying")  
        else:
            print(f"Cyberbullying Detected.\nProbable Category: {class_map[predicted_class]}")
            
        print("Class probabilities:", probabilities)

if __name__ == "__main__":
    main()
