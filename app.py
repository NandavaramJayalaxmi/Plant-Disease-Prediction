import json
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_ROOT / "models"

MODEL_PATH = MODEL_DIR / "plant_disease_mobilenetv2.keras"
CLASS_NAMES_PATH = MODEL_DIR / "class_names.json"
METADATA_PATH = MODEL_DIR / "metadata.json"


# ============================================================
# LOAD MODEL AND CLASS NAMES
# ============================================================

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


@st.cache_data
def load_class_names():
    with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


@st.cache_data
def load_metadata():
    with open(METADATA_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


model = load_model()
class_names = load_class_names()
metadata = load_metadata()

IMG_SIZE = tuple(metadata["input_size"])


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_disease(image):
    # Convert image to RGB
    image = image.convert("RGB")

    # Resize
    image = image.resize(IMG_SIZE)

    # Convert to NumPy array
    image_array = np.array(image, dtype=np.float32)

    # Normalize pixel values to 0-1
    image_array = image_array / 255.0

    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)

    # Prediction
    predictions = model.predict(image_array, verbose=0)

    predicted_index = np.argmax(predictions[0])
    confidence = float(predictions[0][predicted_index])

    predicted_class = class_names[predicted_index]

    return predicted_class, confidence


# ============================================================
# FORMAT DISEASE NAME
# ============================================================

def format_disease_name(disease_name):
    name = disease_name.replace("___", " - ")
    name = name.replace("_", " ")
    return name


# ============================================================
# STREAMLIT PAGE
# ============================================================

st.set_page_config(
    page_title="Plant Disease Prediction",
    page_icon="🌱",
    layout="centered"
)


# ============================================================
# TITLE
# ============================================================

st.title("🌱 Plant Disease Prediction")

st.write(
    "Upload a plant leaf image and let the deep learning "
    "model predict the most likely disease."
)

st.info(
    "🧠 Model: MobileNetV2 Transfer Learning | "
    "38 Plant Disease Classes"
)


# ============================================================
# IMAGE UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "📷 Upload a leaf image",
    type=["jpg", "jpeg", "png"]
)


# ============================================================
# DISPLAY IMAGE AND PREDICT
# ============================================================

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Uploaded Leaf Image",
        use_container_width=True
    )

    if st.button(
        "🔍 Predict Disease",
        use_container_width=True
    ):

        with st.spinner("Analyzing the leaf image..."):

            predicted_class, confidence = predict_disease(image)

        readable_prediction = format_disease_name(
            predicted_class
        )

        confidence_percentage = confidence * 100

        st.success("✅ Prediction completed!")

        st.subheader("🌿 Prediction Result")

        st.markdown(
            f"### 🦠 {readable_prediction}"
        )

        st.write(
            f"**Confidence:** {confidence_percentage:.2f}%"
        )

        st.progress(
            min(confidence, 1.0)
        )

        # Confidence message
        if confidence_percentage >= 90:

            st.success(
                "High confidence prediction"
            )

        elif confidence_percentage >= 70:

            st.warning(
                "Moderate confidence prediction"
            )

        else:

            st.warning(
                "Low confidence prediction. "
                "Consider consulting an agricultural expert."
            )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Plant Disease Prediction • "
    "Deep Learning Capstone Project"
)

st.caption(
    "⚠️ This system provides an AI-based prediction "
    "and should not replace professional agricultural diagnosis."
)