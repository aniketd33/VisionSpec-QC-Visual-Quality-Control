import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model

IMAGE_SIZE = (224, 224)

# Input Layer
inputs = tf.keras.Input(shape=(224, 224, 3))

# Base Model (MobileNetV2)
base_model = MobileNetV2(
    include_top=False,
    weights="imagenet",
    input_tensor=inputs
)

base_model.trainable = False  # Freeze base model

# Custom Head
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation="relu")(x)
x = Dropout(0.5)(x)
outputs = Dense(1, activation="sigmoid")(x)

# Final Model
model = Model(inputs=inputs, outputs=outputs)

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()
