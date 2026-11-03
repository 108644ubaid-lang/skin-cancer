
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

st.set_page_config(page_title="Skin Cancer Classification")

MODEL_PATH = "skin_cancer_model_small.keras"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )

    return tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )

st.title("Skin Cancer Classification")
st.write("Upload a skin image for classification.")

try:
    model = load_model()
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Model loading failed: {e}")
    st.stop()

uploaded_file = st.file_uploader(
    "Upload Skin Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image")

    image = image.resize((128, 128))
    img_array = np.array(image, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = float(
        model.predict(img_array, verbose=0)[0][0]
    )

    if prediction >= 0.5:
        st.subheader("Prediction: Malignant")
        st.write(f"Model score: {prediction * 100:.2f}%")
    else:
        st.subheader("Prediction: Benign")
        st.write(f"Model score: {(1-prediction) * 100:.2f}%")

    st.warning(
        "Educational demonstration only. "
        "This prediction is not a medical diagnosis."
    )
