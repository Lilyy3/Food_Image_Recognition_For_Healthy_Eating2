#purpose of this: 
#add data augmentation
#fine-tune mobilenetv2
#compare with baseline.
#No overwrite the baseline model. save improved as new


# %% 
# importing libraries
import os
import kagglehub
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator




#load the dataset
DATASET_PATH = kagglehub.dataset_download("kmader/food41")
IMAGE_DIR = os.path.join(DATASET_PATH, "images")


#this is the image size required for MobileNetV2
IMG_SIZE = (224, 224)
#No of images processed at once
BATCH_SIZE = 32

#augmentation starts here


#training daat generator
#this rescales pixel values
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,         #normalize pixels from 0-255 to 0-1
    validation_split=0.2,   # use 20% of data for validation
    rotation_range=15,      #randomly rotate images
    zoom_range=0.15,         #randomly zoom images
    width_shift_range=0.1,   #added new
    height_shift_range=0.1,  #added new
    horizontal_flip=True,   #randomly flip images horizontally
    brightness_range=[0.9, 1.1]     #randomly change brightness.



)


#validation daata generator. 
#No validation here bc validation should represent real data

val_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.2
)

#load training images
train_data = train_datagen.flow_from_directory(
    IMAGE_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training"
)

#load validation images.

val_data = val_datagen.flow_from_directory(
    IMAGE_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation"
)

#get num of classes automatically

NUM_CLASSES = train_data.num_classes


#load MobileNetV2 without its head classifier
base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

#freezing base layers
#they will extract features but will not train yet

base_model.trainable = False

#new classifier head
x = base_model.output
#convert feature map into one featrure vector
x = GlobalAveragePooling2D()(x)
#dropout to reduce overfitting
x = Dropout(0.3)(x)
#final output layer
output = Dense(NUM_CLASSES, activation="softmax")(x)




#create full model

model = Model(inputs=base_model.input, outputs=output)

#compile model for first training stage
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)





#callbacks 
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint
)

callbacks = [
    EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True
    ),

    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.2,
        patience=2
    ),

    ModelCheckpoint(
        "best_food_model_improved2.keras",
        monitor="val_accuracy",
        save_best_only=True,
        mode="max"
    )
]




#only train classification head
print("Training classification head...")
model.fit(
    train_data,
    validation_data=val_data,
    epochs=10, #was5
    callbacks=callbacks
)

#fine-tuning top layers

print("Fine-tuning top layers...")

#unfreeze base model
base_model.trainable = True

#keep early later frozen, only last 30 will be trained

for layer in base_model.layers[:-50]: #was 30
    layer.trainable = False


#recompile with very small learning rate to avoid damaging pretrained weights
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

#train again after unfreezing the last layer
model.fit(
    train_data,
    validation_data=val_data,
    epochs=10, #was5
    callbacks=callbacks
)

#saving the improved model

model.save("mobilenetv2_food101_improved2.keras")

print("Improved model 2 saved.")

# %%
