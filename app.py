import streamlit as st
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from evaluate import predict_text
from Text_Extraction_Final import TextExtractor
import tempfile
import os

# Page config
st.set_page_config(
    page_title="Cyberbullying Detection",
    page_icon="🛡️",
    layout="wide"
)

@st.cache_resource
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForSequenceClassification.from_pretrained("saved_model").to(device)
    tokenizer = AutoTokenizer.from_pretrained("saved_model")
    return model, tokenizer, device

@st.cache_resource
def load_extractors():
    text_extractor = TextExtractor()
    text_extractor.initialize_models()
    return text_extractor

def main():
    # Header
    st.title("🛡️ Cyberbullying Detection System")
    st.markdown("""
    This application uses a BERT-based model to detect cyberbullying in text content
    from text, images, audio, or video files.
    """)

    # Load models
    try:
        model, tokenizer, device = load_model()
        text_extractor = load_extractors()
        st.success("✅ Models loaded successfully!")
    except Exception as e:
        st.error(f"❌ Error loading models: {str(e)}")
        return

    # Create tabs for different input types
    tab1, tab2 = st.tabs(["Text Input", "File Input"])

    with tab1:
        text_input = st.text_area(
            "Enter text to analyze:",
            height=150,
            placeholder="Type or paste your text here..."
        )
        analyze_text = st.button("Analyze Text", key="text_button")
        text_to_analyze = text_input if analyze_text else None

    with tab2:
        uploaded_file = st.file_uploader(
            "Upload a file (images, audio, or video)", 
            type=['png', 'jpg', 'jpeg', 'mp3', 'wav', 'm4a', 'mp4', 'avi', 'mov', 'mkv']
        )
        
        if uploaded_file is not None:
            # Create a temporary file to process
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                temp_path = tmp_file.name

            # If it's an image, show preview
            if uploaded_file.type.startswith('image/'):
                st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
            
            analyze_file = st.button("Extract and Analyze", key="file_button")
            if analyze_file:
                with st.spinner("Extracting text from file..."):
                    extracted_text = text_extractor.extract_text(temp_path)
                    os.unlink(temp_path)  # Clean up temporary file
                    
                    if extracted_text and not extracted_text.startswith("Error"):
                        st.subheader("Extracted Text:")
                        st.write(extracted_text)
                        text_to_analyze = extracted_text
                    else:
                        st.warning(f"Could not extract text from file: {extracted_text}")
                        text_to_analyze = None

    class_map = {
        0: 'Not Cyberbullying', 
        1: 'Gender-based Bullying', 
        2: 'Religious Discrimination', 
        3: 'Other Cyberbullying', 
        4: 'Age-based Discrimination', 
        5: 'Ethnic/Racial Discrimination'
    }

    # Analysis logic
    if text_to_analyze and text_to_analyze.strip() != "":
        with st.spinner("Analyzing text..."):
            predicted_class, probabilities = predict_text(text_to_analyze, model, tokenizer, device)

            # Display results
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Prediction Result")
                if predicted_class == 0:
                    st.success("✅ No cyberbullying detected")
                else:
                    st.error(f"⚠️ Cyberbullying Detected\nCategory: {class_map[predicted_class]}")

            with col2:
                st.subheader("Confidence Scores")
                for class_idx, prob in enumerate(probabilities):
                    st.progress(prob, text=f"{class_map[class_idx]}: {prob:.2%}")

    # Add information about the model
    with st.expander("ℹ️ About the Model"):
        st.markdown("""
        This model is trained on a dataset of cyberbullying tweets and can detect various types of cyberbullying:
        - Gender-based bullying
        - Religious discrimination
        - Age-based discrimination
        - Ethnic/racial discrimination
        - Other forms of cyberbullying
        
        The model uses BERT (Bidirectional Encoder Representations from Transformers) architecture
        and has been fine-tuned specifically for cyberbullying detection.
        """)

if __name__ == "__main__":
    main() 