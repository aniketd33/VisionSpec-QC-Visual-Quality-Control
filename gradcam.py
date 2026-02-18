import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

# ==============================
# 1️⃣ Load Trained Functional Model
# ==============================
model = tf.keras.models.load_model("quality_model.h5")

# ==============================
# 2️⃣ Select Image
# ==============================
img_path = "data/test/good/cast_ok_0_1002.jpeg"

if not os.path.exists(img_path):
    print("Image path wrong ❌")
    exit()

print("Using image:", img_path)

# ==============================
# 3️⃣ Image Preprocessing
# ==============================
IMAGE_SIZE = (224, 224)

img = cv2.imread(img_path)
img = cv2.resize(img, IMAGE_SIZE)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

img_array = img / 255.0
img_array = np.expand_dims(img_array, axis=0)

# ==============================
# 4️⃣ Get Last Conv Layer
# ==============================
last_conv_layer = model.get_layer("Conv_1")


# ==============================
# 5️⃣ Create Grad Model
# ==============================
grad_model = tf.keras.models.Model(
    inputs=model.inputs,
    outputs=[last_conv_layer.output, model.output]
)

# ==============================
# 6️⃣ Compute Grad-CAM
# ==============================
with tf.GradientTape() as tape:
    conv_outputs, predictions = grad_model(img_array)
    loss = predictions[:, 0]

grads = tape.gradient(loss, conv_outputs)

pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

conv_outputs = conv_outputs[0]

heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
heatmap = tf.squeeze(heatmap)

heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
heatmap = heatmap.numpy()

# ==============================
# 7️⃣ Overlay Heatmap
# ==============================
heatmap = cv2.resize(heatmap, IMAGE_SIZE)
heatmap = np.uint8(255 * heatmap)
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

superimposed_img = cv2.addWeighted(
    cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
    0.6,
    heatmap,
    0.4,
    0
)

# ==============================
# 8️⃣ Show Result
# ==============================
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(img)
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title("Grad-CAM")
plt.imshow(cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB))
plt.axis("off")

plt.show()
