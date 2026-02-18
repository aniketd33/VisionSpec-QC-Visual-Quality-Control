import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from model import model

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 5

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    "data/train",
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary"
)

val_generator = val_datagen.flow_from_directory(
    "data/val",
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary"
)

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS
)

# Save history
import pickle
with open("training_history.pkl", "wb") as f:
    pickle.dump(history.history, f)

model.save("quality_model.h5")

print("Model trained and saved successfully!")
