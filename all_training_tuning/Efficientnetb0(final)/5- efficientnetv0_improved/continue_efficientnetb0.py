import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

DATASET_PATH = "/home/nanai/datasets/food41"
IMAGE_DIR = os.path.join(DATASET_PATH, "images")

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

train_datagen = ImageDataGenerator(
    validation_split=0.2,
    rotation_range=25,
    zoom_range=0.2,
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    brightness_range=[0.8, 1.2]
)

val_datagen = ImageDataGenerator(
    validation_split=0.2
)

train_data = train_datagen.flow_from_directory(
    IMAGE_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training"
)

val_data = val_datagen.flow_from_directory(
    IMAGE_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
    shuffle=False
)

# Load the model that stopped early
model = tf.keras.models.load_model("efficientnetb0_food101_final.keras")

# Fine-tune more layers
for layer in model.layers[:-80]:
    layer.trainable = False

for layer in model.layers[-80:]:
    layer.trainable = True

# Keep BatchNorm frozen for stable fine-tuning
for layer in model.layers:
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=5e-6),
    loss="categorical_crossentropy",
    metrics=[
        "accuracy",
        tf.keras.metrics.TopKCategoricalAccuracy(k=5, name="top5_accuracy")
    ]
)

callbacks = [
    ModelCheckpoint(
        "efficientnetb0_food101_continued_best.keras",
        monitor="val_accuracy",
        save_best_only=True,
        mode="max"
    ),
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-7
    )
]

model.fit(
    train_data,
    validation_data=val_data,
    epochs=7,
    callbacks=callbacks
)

model.save("efficientnetb0_food101_continued_final.keras")

print("Continued EfficientNetB0 training saved.")
