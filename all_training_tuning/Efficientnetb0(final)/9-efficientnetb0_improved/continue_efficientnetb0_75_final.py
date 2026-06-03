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
    rotation_range=15,
    zoom_range=0.10,
    horizontal_flip=True,
    width_shift_range=0.05,
    height_shift_range=0.05,
    brightness_range=[0.90, 1.10]
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

model = tf.keras.models.load_model("efficientnetb0_food101_reach75_best.keras")

for layer in model.layers[:-120]:
    layer.trainable = False

for layer in model.layers[-120:]:
    layer.trainable = True

for layer in model.layers:
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=5e-7),
    loss="categorical_crossentropy",
    metrics=[
        "accuracy",
        tf.keras.metrics.TopKCategoricalAccuracy(k=5, name="top5_accuracy")
    ]
)

callbacks = [
    ModelCheckpoint(
        "efficientnetb0_food101_over75_best.keras",
        monitor="val_accuracy",
        save_best_only=True,
        mode="max"
    ),
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=1e-7
    )
]

model.fit(
    train_data,
    validation_data=val_data,
    epochs=5,
    callbacks=callbacks
)

model.save("efficientnetb0_food101_over75_final.keras")

print("Over 75 attempt saved.")
