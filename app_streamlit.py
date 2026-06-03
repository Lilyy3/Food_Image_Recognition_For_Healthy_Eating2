import os
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image

from nutrition_rules import get_food_result


# -----------------------------
# Project paths
# -----------------------------
PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "efficientnetb0_food101_over75_best.keras"
NUTRITION_CSV_PATH = PROJECT_DIR / "food101_nutrition.csv"
CLASS_NAMES_PATH = PROJECT_DIR / "class_names.txt"

IMG_SIZE = (224, 224)


# -----------------------------
# Load model and class names
# -----------------------------
@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        st.error(f"Model file not found: {MODEL_PATH.name}")
        st.stop()

    return tf.keras.models.load_model(MODEL_PATH)


@st.cache_data
def load_class_names():
    if not CLASS_NAMES_PATH.exists():
        st.error(
            "class_names.txt was not found. Run create_class_names.py first, "
            "or place class_names.txt in the project folder."
        )
        st.stop()

    with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as file:
        class_names = [line.strip() for line in file if line.strip()]

    return class_names


def preprocess_uploaded_image(uploaded_file):
    img = Image.open(uploaded_file).convert("RGB")
    img = img.resize(IMG_SIZE)

    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    return img, img_array


def predict_food(model, img_array, class_names):
    prediction = model.predict(img_array)
    predicted_index = int(np.argmax(prediction))
    confidence = float(np.max(prediction))

    if predicted_index >= len(class_names):
        st.error("Predicted class index is outside class_names.txt range.")
        st.stop()

    food_label = class_names[predicted_index]
    return food_label, confidence


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(
    page_title="Food Image Recognition for Healthy Eating",
    page_icon="🍽️",
    layout="centered"
)

st.title("Food Image Recognition for Healthy Eating")
st.write(
    "Upload a food image. The system predicts the food class, retrieves nutrition values, "
    "and gives rule-based health advice."
)

uploaded_file = st.file_uploader(
    "Upload food image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    model = load_model()
    class_names = load_class_names()

    preview_img, img_array = preprocess_uploaded_image(uploaded_file)

    st.image(preview_img, caption="Uploaded Image", use_container_width=True)

    if st.button("Predict"):
        food_label, confidence = predict_food(model, img_array, class_names)

        result = get_food_result(
            food_label=food_label,
            confidence=confidence,
            csv_path=str(NUTRITION_CSV_PATH)
        )

        st.subheader("Prediction Result")

        if "error" in result:
            st.error(result["error"])
            st.write("Predicted food label:", food_label)
            st.write("Confidence:", round(confidence * 100, 2), "%")
        else:
            st.write("**Food:**", result["food"])
            st.write("**Confidence:**", str(result["confidence"]) + "%")

            st.subheader("Nutrition Information")
            st.table({
                "Nutrient": [
                    "Calories",
                    "Protein",
                    "Carbs",
                    "Fat",
                    "Sodium",
                    "Sugar",
                    "Fiber"
                ],
                "Value": [
                    f'{result["calories_kcal"]} kcal',
                    f'{result["protein_g"]} g',
                    f'{result["carbs_g"]} g',
                    f'{result["fat_g"]} g',
                    f'{result["sodium_mg"]} mg',
                    f'{result["sugar_g"]} g',
                    f'{result["fiber_g"]} g'
                ]
            })

            st.subheader("Health Advice")
            st.write("**Health Level:**", result["health_level"])
            st.write("**Health Notes:**", result["health_notes"])
            st.write("**Recommendation:**", result["recommendation"])
else:
    st.info("Upload an image to start.")
