# The purpose of this file:
# Reload saved model
# Evaluate accuracy/loss/top-5 accuracy
# Calculate precision, recall, F1-score, confusion matrix
# Save final model performance results

import os
import kagglehub
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd

# 1. Load dataset path
DATASET_PATH = kagglehub.dataset_download("kmader/food41")
IMAGE_DIR = os.path.join(DATASET_PATH, "images")

# 2. Prepare validation data
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# IMPORTANT:
# Do NOT use rescale=1./255 for EfficientNetB0
datagen = ImageDataGenerator(
    validation_split=0.2
)

val_data = datagen.flow_from_directory(
    IMAGE_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
    shuffle=False
)

# 3. Load final EfficientNetB0 model
model = tf.keras.models.load_model("efficientnetb0_food101_over75_best.keras")

# 4. Evaluate
eval_results = model.evaluate(val_data)

print("Model metrics:", model.metrics_names)
print("Evaluation results:", eval_results)

val_loss = eval_results[0]
val_acc = eval_results[1]

# If top5_accuracy exists
val_top5 = eval_results[2] if len(eval_results) > 2 else None

print("Validation Loss:", val_loss)
print("Validation Accuracy:", val_acc)

if val_top5 is not None:
    print("Validation Top-5 Accuracy:", val_top5)

# 5. Predictions
predictions = model.predict(val_data)

# Convert probabilities to class indices
predicted_classes = np.argmax(predictions, axis=1)

# True labels
true_classes = val_data.classes

# Class names
class_names = list(val_data.class_indices.keys())

# 6. Classification metrics
report = classification_report(
    true_classes,
    predicted_classes,
    target_names=class_names,
    zero_division=0
)

print(report)

# 7. Confusion matrix
cm = confusion_matrix(true_classes, predicted_classes)

print("Confusion Matrix Shape:", cm.shape)

# 8. Save results
os.makedirs("results", exist_ok=True)

with open("results/efficientnetb0_final_evaluation.txt", "w") as f:
    f.write("Final EfficientNetB0 Evaluation\n")
    f.write("Validation Loss: " + str(val_loss) + "\n")
    f.write("Validation Accuracy: " + str(val_acc) + "\n")

    if val_top5 is not None:
        f.write("Validation Top-5 Accuracy: " + str(val_top5) + "\n")

    f.write("\nClassification Report:\n")
    f.write(report)

# Save confusion matrix as CSV
cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
cm_df.to_csv("results/efficientnetb0_final_confusion_matrix.csv")

# Save classification report as CSV
report_dict = classification_report(
    true_classes,
    predicted_classes,
    target_names=class_names,
    output_dict=True,
    zero_division=0
)

report_df = pd.DataFrame(report_dict).transpose()
report_df.to_csv("results/efficientnetb0_final_classification_report.csv")

print("Results saved in results/ folder.")