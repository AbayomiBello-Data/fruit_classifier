import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
from pathlib import Path

# Constants
IMAGE_SIZE = 160
MODEL_PATH = "MobileNetV2 (1).h5"

@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    return model

def preprocess_image(image: Image.Image) -> np.ndarray:
    """Preprocess the input image to make it ready for prediction."""
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
    image = np.array(image) / 255.0
    image = image.astype(np.float32)
    image = np.expand_dims(image, axis=0)
    return image

@st.cache
def load_class_names():
    _, unique_idx = np.unique(y_train, return_index=True)
    unique_fruit_names = [class_names_train[label] for label in y_train[unique_idx]]
    return sorted(unique_fruit_names)

# Load the model and class names
model = load_model()
class_names = load_class_names()

# Streamlit UI code
st.title("Fruit Classifier")
st.write("""
Upload an image of a fruit and the model will predict its name.
""")

uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image.", use_column_width=True)
    st.write("")
    st.write("Classifying...")
    
    processed_image = preprocess_image(image)
    predictions = model.predict(processed_image)
    predicted_class = np.argmax(predictions)
    
    st.write(f"The model predicts this fruit as: {class_names[predicted_class]}")
