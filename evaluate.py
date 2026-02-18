import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report

# Load trained model
model = tf.keras.models.load_model("quality_model.h5")

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

# Test Data Generator
test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    "data/test",
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=False
)

# Evaluate model
loss, accuracy = model.evaluate(test_generator)

print("\nFinal Test Accuracy:", round(accuracy * 100, 2), "%")
print("Final Test Loss:", round(loss, 4))

# Predictions
predictions = model.predict(test_generator)
predicted_classes = (predictions > 0.5).astype("int32")

true_classes = test_generator.classes
class_labels = list(test_generator.class_indices.keys())

# Confusion Matrix
cm = confusion_matrix(true_classes, predicted_classes)
print("\nConfusion Matrix:\n", cm)

# Classification Report
print("\nClassification Report:\n")
print(classification_report(true_classes, predicted_classes, target_names=class_labels))

model.summary()
