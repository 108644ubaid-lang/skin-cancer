import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Load CNN model
model = tf.keras.models.load_model("skin_cancer_model.keras")

# Page settings
st.set_page_config(
    page_title="Skin Cancer Classification",
    page_icon="🩺"
)

st.title("🩺 Skin Cancer Classification")
st.write("Upload a skin lesion image to classify it as Benign or Malignant.")

# Upload image
uploaded_file = st.file_uploader(
    "Upload Skin Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        width=300
    )

    # Preprocessing
    image = image.resize((128, 128))

    image_array = np.array(image) / 255.0

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    # Prediction
    prediction = model.predict(image_array, verbose=0)[0][0]

    if prediction >= 0.5:
        result = "Malignant"
        confidence = prediction * 100
    else:
        result = "Benign"
        confidence = (1 - prediction) * 100

    # Result
    st.subheader("Prediction Result")

    st.success(result)

    st.write(
        "Confidence:",
        round(confidence, 2),
        "%"
    )

st.warning(
    "This application is for educational purposes only "
    "and should not be used for medical diagnosis."
)
