import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ---------------------------------
# Page Settings
# ---------------------------------

st.set_page_config(
    page_title="Skin Cancer Classifier",
    page_icon="🩺",
    layout="centered"
)

# ---------------------------------
# Load Model
# ---------------------------------

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "skin_cancer_model.keras",
        compile=False
    )

model = load_model()

# ---------------------------------
# App Title
# ---------------------------------

st.title("🩺 Skin Cancer Classification")

st.write(
    "Upload a skin lesion image to classify it as "
    "Benign or Malignant."
)

# ---------------------------------
# Upload Image
# ---------------------------------

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

# ---------------------------------
# Prediction
# ---------------------------------

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        width=350
    )

    # Resize
    image = image.resize((128, 128))

    # Convert to array
    image_array = np.array(image)

    # Normalize
    image_array = image_array / 255.0

    # Add batch dimension
    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    # Predict
    prediction = model.predict(
        image_array,
        verbose=0
    )[0][0]

    # Result
    if prediction >= 0.5:
        result = "Malignant"
        confidence = prediction * 100
    else:
        result = "Benign"
        confidence = (1 - prediction) * 100

    # Display
    st.subheader("Prediction")

    if result == "Benign":
        st.success("Result: Benign")
    else:
        st.error("Result: Malignant")

    st.info(
        f"Confidence: {confidence:.2f}%"
    )

# ---------------------------------
# Disclaimer
# ---------------------------------

st.divider()

st.caption(
    "This application is developed for educational purposes "
    "and should not be used as a medical diagnosis."
)
