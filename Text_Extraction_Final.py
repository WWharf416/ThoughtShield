import os
import cv2
import easyocr
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import whisper
from pydub import AudioSegment

class TextExtractor:
    def __init__(self):
        self.ocr_reader = None
        self.whisper_model = None
        self.initialized = False
    
    def initialize_models(self):
        """Lazy initialization of models"""
        if not self.initialized:
            print("Initializing models...")
            self.ocr_reader = easyocr.Reader(['en'], gpu=False)
            self.whisper_model = whisper.load_model("base")
            self.initialized = True
    
    def extract_text(self, file_path):
        """Unified function to extract text from either image or audio"""
        self.initialize_models()
        
        if not os.path.exists(file_path):
            return "Error: File not found"
        
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext in ['.jpg', '.jpeg', '.png']:
            return self._extract_from_image(file_path)
        elif ext in ['.mp3', '.wav', '.m4a', '.mp4']:
            return self._extract_from_audio(file_path)
        else:
            return "Error: Unsupported file type"
    
    def _extract_from_image(self, image_path):
        try:
            img = cv2.imread(image_path)
            if img is None:
                return "Error: Could not load image"
            
            results = self.ocr_reader.readtext(image_path)
            if not results:
                return "No text detected"
            
            # Visualization
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            plt.figure(figsize=(10, 10))
            plt.imshow(img)
            
            for detection in results:
                box = detection[0]
                text = detection[1]
                top_left = tuple(map(int, box[0]))
                bottom_right = tuple(map(int, box[2]))
                plt.plot([top_left[0], bottom_right[0], bottom_right[0], top_left[0], top_left[0]],
                         [top_left[1], top_left[1], bottom_right[1], bottom_right[1], top_left[1]], 'r')
                plt.text(top_left[0], top_left[1]-10, text[:20], color='red', fontsize=12)
            
            plt.title("Detected Text")
            plt.axis('off')
            plt.show()
            
            return " ".join([d[1] for d in results])
        
        except Exception as e:
            return f"Image processing error: {str(e)}"
    
    def _extract_from_audio(self, audio_path):
        try:
            result = self.whisper_model.transcribe(audio_path)
            return result["text"]
        except Exception as e:
            return f"Audio processing error: {str(e)}"


# Usage example:
if __name__ == "__main__":
    extractor = TextExtractor()
    
    # Just call this one function with your file path
    result = extractor.extract_text("Arthur.mp3")  # or "your_audio.mp3"
    
    print("\nExtracted Text:")
    print("=" * 50)
    print(result)
    print("=" * 50)