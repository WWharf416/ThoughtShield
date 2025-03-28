import os
import cv2
import easyocr
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import whisper
import subprocess  # Added for FFmpeg processing

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
    
    def _extract_audio_from_video(self, video_path):
        """Extract audio from video file using FFmpeg"""
        try:
            # Create a temporary audio file
            temp_audio_path = os.path.splitext(video_path)[0] + "_temp.wav"
            
            # Use FFmpeg to extract audio
            ffmpeg_command = [
                'ffmpeg', 
                '-i', video_path, 
                '-vn',  # Disable video
                '-acodec', 'pcm_s16le',  # Set audio codec
                '-ar', '44100',  # Set sample rate
                '-ac', '1',  # Set to mono
                temp_audio_path
            ]
            
            # Run FFmpeg command
            subprocess.run(ffmpeg_command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            return temp_audio_path
        except subprocess.CalledProcessError as e:
            print(f"FFmpeg error: {e}")
            return None
        except Exception as e:
            print(f"Error extracting audio from video: {str(e)}")
            return None
    
    def extract_text(self, file_path):
        """Unified function to extract text from image, audio, or video"""
        self.initialize_models()
        
        if not os.path.exists(file_path):
            return "Error: File not found"
        
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext in ['.jpg', '.jpeg', '.png']:
            return self._extract_from_image(file_path)
        elif ext in ['.mp3', '.wav', '.m4a']:
            return self._extract_from_audio(file_path)
        elif ext in ['.mp4', '.avi', '.mov', '.mkv']:
            # For video files, first extract audio
            temp_audio_path = self._extract_audio_from_video(file_path)
            if temp_audio_path:
                try:
                    # Transcribe the extracted audio
                    audio_text = self._extract_from_audio(temp_audio_path)
                    
                    # Clean up temporary audio file
                    os.remove(temp_audio_path)
                    
                    return audio_text
                except Exception as e:
                    # Clean up temporary audio file in case of error
                    if os.path.exists(temp_audio_path):
                        os.remove(temp_audio_path)
                    return f"Video processing error: {str(e)}"
            else:
                return "Error: Could not extract audio from video"
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
            # Uncomment to see the image with detected text
            
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # plt.figure(figsize=(10, 10))
            # plt.imshow(img)
            
            # for detection in results:
            #     box = detection[0]
            #     text = detection[1]
            #     top_left = tuple(map(int, box[0]))
            #     bottom_right = tuple(map(int, box[2]))
            #     plt.plot([top_left[0], bottom_right[0], bottom_right[0], top_left[0], top_left[0]],
            #              [top_left[1], top_left[1], bottom_right[1], bottom_right[1], top_left[1]], 'r')
            #     plt.text(top_left[0], top_left[1]-10, text[:20], color='red', fontsize=12)
            
            # plt.title("Detected Text")
            # plt.axis('off')
            # plt.show()
            
            return " ".join([d[1] for d in results])
        
        except Exception as e:
            return f"Image processing error: {str(e)}"
    
    def _extract_from_audio(self, audio_path):
        try:
            result = self.whisper_model.transcribe(audio_path)
            return result["text"]
        except Exception as e:
            return f"Audio processing error: {str(e)}"

# Usage 
if __name__ == "__main__":
    extractor = TextExtractor()
    
    # Demonstrate usage with different file types
    result = extractor.extract_text("trial.png")  
    
    print("\nExtracted Text:")
    print("=" * 50)
    print(result)
    print("=" * 50)
