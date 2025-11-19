import streamlit as st
import tensorflow as tf
from pathlib import Path
import keras
import random
import pandas as pd
import numpy as np
from PIL import Image, ImageOps
import os

def side_bar():
    st.sidebar.title("Medical Prediction")
    add_selectbox = st.sidebar.selectbox(
        label='Type of Prediction',
        options=("Image", "Text"),
    )
    return add_selectbox

def image_prediction_panel():
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    with st.spinner("Model is being loaded..."):
        model = load_cv_model()

    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

        if st.button("Predict"):
            with st.spinner("Analyzing image..."):
                image = Image.open(uploaded_file)
                img_array = preprocess_image(image)

                preds = model.predict(img_array)[0]
                class_idx = np.argmax(preds)
                confidence = preds[class_idx]

                CLASS_NAMES = ["COVID", "NORMAL", "PNEUMONIA"]
                label = CLASS_NAMES[class_idx]

                st.success(f"### Prediction: **{label}**")
                st.write(f"Confidence: `{confidence:.4f}`")

                st.write("### Probability Scores")
                for cls, p in zip(CLASS_NAMES, preds):
                    st.write(f"- **{cls}**: {p:.4f}")


@st.cache_resource
def load_cv_model():
    model = keras.models.load_model('src/models/cv_model.keras')
    return model

def preprocess_image(image_data):
    target_size = (224, 224)

    image = image_data.convert("RGB")
    image = image.resize(target_size)
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, 0)

    return img_array

def app():
    user_selection = side_bar()
    if user_selection == "Image":
        image_prediction_panel()

app()
