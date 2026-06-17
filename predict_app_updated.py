
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import kagglehub
import logging

# Prepare log file
logging.basicConfig(
    filename='app.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def log_prediction(image_name, predicted_class, confidence, success=True):
    """Recording the details of each prediction process"""
    if success:
        logging.info(f"Image: {image_name} | Predicted: {predicted_class} | Confidence: {confidence:.2f}")
    else:
        logging.warning(f"Image: {image_name} | Prediction failed.")

from nutrition_rules import get_food_result

# Load trained model
model = tf.keras.models.load_model("efficientnetb0_food101_over75_best.keras")

# Load Food-101 class names
DATASET_PATH = kagglehub.dataset_download("kmader/food41")
IMAGE_DIR = os.path.join(DATASET_PATH, "images")
class_names = sorted(os.listdir(IMAGE_DIR))

# Change this image path
img_path = r"C:\Users\nanai\OneDrive\Desktop\AISD_Project\test_food.jpg.jpg"

# Preprocess image
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)

img_array = np.expand_dims(img_array, axis=0)

# Predict
prediction = model.predict(img_array)
predicted_index = np.argmax(prediction)
confidence = float(np.max(prediction))
food_label = class_names[predicted_index]

# Nutrition + rule-based recommendation
result = get_food_result(
    food_label=food_label,
    confidence=confidence,
    csv_path="food101_nutrition.csv"
)

# Display output
print("Food:", result["food"])
print("Confidence:", result["confidence"], "%")
print("Calories:", result["calories_kcal"], "kcal")
print("Protein:", result["protein_g"], "g")
print("Carbs:", result["carbs_g"], "g")
print("Fat:", result["fat_g"], "g")
print("Sodium:", result["sodium_mg"], "mg")
print("Sugar:", result["sugar_g"], "g")
print("Fiber:", result["fiber_g"], "g")
print("Health Level:", result["health_level"])
print("Health Notes:", result["health_notes"])
print("Recommendation:", result["recommendation"])

# Recording the prediction process in the log
log_prediction(os.path.basename(img_path), food_label, confidence)
