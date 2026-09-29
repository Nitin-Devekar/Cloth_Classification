import streamlit as st
import tensorflow as tf
import joblib
import numpy as np
from PIL import Image

# 1. Page Configuration
st.set_page_config(
    page_title="Clothing Classifier",
    page_icon="👕",
    layout="centered"
)

# 2. Load Model and Metadata
@st.cache_resource
def load_assets():
    model = tf.keras.models.load_model("fashion_classifier.keras")
    metadata = joblib.load("fashion_classifier_metadata.joblib")
    
    # Invert the class_indices dictionary to map integers to string labels
    class_indices = metadata['class_indices']
    labels_map = {v: k for k, v in class_indices.items()}
    
    return model, labels_map, metadata['image_size']

try:
    model, labels_map, image_size = load_assets()
except Exception as e:
    st.error(f"Error loading model or metadata. Ensure 'fashion_classifier.keras' and 'fashion_classifier_metadata.joblib' are in the same folder.")
    st.stop()

# 3. UI Elements
st.title("👕 Clothing Classification App")
st.write("Upload an image of a clothing item to predict its category.")

uploaded_file = st.file_uploader("Choose a clothing image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    # Preprocessing Pipeline
    with st.spinner("Processing image and predicting..."):
        # 1. Resize based on metadata configuration
        img_resized = image.resize(image_size)
        
        # 2. Convert to numpy array and scale pixels (assuming 1./255 scaling used in training)
        img_array = np.array(img_resized) / 255.0
        
        # 3. Add batch dimension (Shape becomes: [1, height, width, channels])
        img_array = np.expand_dims(img_array, axis=0)
        
        # Model Inference
        predictions = model.predict(img_array)
        
        # Post-processing
        predicted_class_idx = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class_idx] * 100
        predicted_label = labels_map[predicted_class_idx]
        
    # Display Results
    st.success(f"### Prediction: **{predicted_label.upper()}**")
    st.metric(label="Confidence Score", value=f"{confidence:.2f}%")
    
    # Optional: Expandable Breakdown of All Probabilities
    with st.expander("View detailed prediction probabilities"):
        for idx, prob in enumerate(predictions[0]):
            st.write(f"**{labels_map[idx]}**: {prob * 100:.2f}%")
