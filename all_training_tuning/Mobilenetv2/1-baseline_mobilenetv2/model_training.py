 # %% Importing the dataset
import kagglehub
import os

DATASET_PATH = kagglehub.dataset_download("kmader/food41")

print("Dataset path:", DATASET_PATH)
print("Files:", os.listdir(DATASET_PATH)) 






# %% Checking dataset structure 
import os

for root, dirs, files in os.walk(DATASET_PATH):
    print("ROOT:", root)
    print("DIRS:", dirs[:10])
    print("FILES:", files[:10])
    print("-" * 50)
    break



for item in os.listdir(DATASET_PATH):
    item_path = os.path.join(DATASET_PATH, item)
    print(item, "->", "folder" if os.path.isdir(item_path) else "file")





# %% Data loading and preprocessing


from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

IMAGE_DIR = os.path.join(DATASET_PATH, "images")

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_data = datagen.flow_from_directory(
    IMAGE_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training"
)

val_data = datagen.flow_from_directory(
    IMAGE_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation"
)    



# %% Building baseline MobileNetV2 model

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model

NUM_CLASSES = train_data.num_classes

base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

base_model.trainable = False  # freeze pretrained layers

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
output = Dense(NUM_CLASSES, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()






# %% training the baseline model
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=5
)



# %% Evaluation

val_loss, val_acc = model.evaluate(val_data)

print("Validation Loss:", val_loss)
print("Validation Accuracy:", val_acc)



# %% saving the model

model.save("mobilenetv2_food101_baseline.keras")

# %%
