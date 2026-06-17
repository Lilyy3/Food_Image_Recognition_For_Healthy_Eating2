
model_evaluation.py
Purpose: Reload saved model, evaluate metrics, generate classification report.
Usage:
    - For CI/CD (fast test): python model_evaluation.py --test-sample
    - For full evaluation: python model_evaluation.py
"""

import os
import sys
import kagglehub
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd

# FULL EVALUATION FUNCTION
def run_full_evaluation():
    """
    Load the full Food41 dataset, evaluate the model, and save results.
    This function is heavy and should NOT be run in CI/CD pipelines.
    """
    print(" Starting FULL evaluation on Food41 dataset")
    
    # 1. Load dataset path
    DATASET_PATH = kagglehub.dataset_download("kmader/food41")
    IMAGE_DIR = os.path.join(DATASET_PATH, "images")
    
    # 2. Prepare validation data
    IMG_SIZE = (224, 224)
    BATCH_SIZE = 32
    
    # IMPORTANT: Do NOT use rescale=1./255 for EfficientNetB0
    datagen = ImageDataGenerator(validation_split=0.2)
    
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
    val_top5 = eval_results[2] if len(eval_results) > 2 else None
    
    print("Validation Loss:", val_loss)
    print("Validation Accuracy:", val_acc)
    if val_top5 is not None:
        print("Validation Top-5 Accuracy:", val_top5)
    
    # 5. Predictions
    predictions = model.predict(val_data)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = val_data.classes
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
    
    print(" Full results saved in 'results/' folder.")

#  SAMPLE TEST FUNCTION (FOR CI/CD) 
def run_sample_test():
    """
    Lightweight test for CI/CD pipeline.
    Only checks if the model can load and produce a valid output shape.
    No dataset downloading or heavy processing.
    """
    print(" Running SAMPLE evaluation for CI/CD...")
    
    try:
        # 1. Check if model file exists
        model_path = "efficientnetb0_food101_over75_best.keras"
        if not os.path.exists(model_path):
            print(" Model file not found. Skipping actual model test.")
            return
        
        # 2. Load the model
        model = tf.keras.models.load_model(model_path)
        print(" Model loaded successfully.")
        
        # 3. Test prediction on dummy data (shape: 1, 224, 224, 3)
        dummy_input = np.random.rand(1, 224, 224, 3).astype(np.float32)
        output = model.predict(dummy_input)
        
        # 4. Verify output shape (should be 101 classes for Food101)
        assert output.shape == (1, 101), f"Unexpected output shape: {output.shape}"
        print(f" Sample test passed! Output shape: {output.shape}")
        
    except Exception as e:
        print(f" Sample test failed: {e}")
        sys.exit(1)  # Exit with error code so CI/CD knows it failed

#  MAIN ENTRY POINT 
if __name__ == "__main__":
    # Check command-line arguments
    if len(sys.argv) > 1 and sys.argv[1] == "--test-sample":
        # THIS is what CI/CD runs -> FAST & LIGHT
        run_sample_test()
    else:
        # THIS is what you run locally -> HEAVY & FULL
        run_full_evaluation()
