import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import time
from PIL import Image
import matplotlib.pyplot as plt

# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(page_title="VisionSpec QC", layout="wide")
st.title("🏭 VisionSpec QC – Visual Quality Control System")
st.write("Upload or capture a PCB image to classify PASS or DEFECT with Grad-CAM.")

IMAGE_SIZE = (224, 224)

# -----------------------------
# LOAD MODEL (cached)
# -----------------------------
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("quality_model.h5")
    return model

model = load_model()

# -----------------------------
# PREDICTION FUNCTION
# -----------------------------
def predict_frame(frame):
    img = cv2.resize(frame, IMAGE_SIZE)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    start_time = time.time()
    prediction = model.predict(img, verbose=0)
    end_time = time.time()

    latency = (end_time - start_time) * 1000
    confidence = float(prediction[0][0])
    label = "DEFECT ❌" if confidence > 0.5 else "PASS ✅"

    return label, confidence, latency, img


# -----------------------------
# GRAD-CAM FUNCTION
# -----------------------------
def make_gradcam_heatmap(img_array, model):
    try:
        base_model = model.get_layer("mobilenetv2_1.00_224")
        last_conv_layer = base_model.get_layer("Conv_1")

        grad_model = tf.keras.models.Model(
            inputs=model.inputs,
            outputs=[last_conv_layer.output, model.output]
        )

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            loss = predictions[:, 0]

        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        heatmap = np.maximum(heatmap, 0) / np.max(heatmap)
        return heatmap

    except Exception as e:
        st.warning(f"Grad-CAM issue: {e}")
        return None


# -----------------------------
# DISPLAY FUNCTION
# -----------------------------
def process_and_display(frame):
    label, confidence, latency, img_array = predict_frame(frame)

    col1, col2 = st.columns(2)

    with col1:
        st.image(frame, caption="Original Image", use_column_width=True)

    heatmap = make_gradcam_heatmap(img_array, model)

    with col2:
        if heatmap is not None:
            heatmap = cv2.resize(heatmap, (frame.shape[1], frame.shape[0]))
            heatmap = np.uint8(255 * heatmap)
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            superimposed = cv2.addWeighted(frame, 0.6, heatmap, 0.4, 0)
            st.image(superimposed, caption="Grad-CAM", use_column_width=True)
        else:
            st.info("Grad-CAM not available")

    st.markdown(f"### 🔍 Prediction: **{label}**")
    st.write(f"Confidence: {confidence:.3f}")
    st.write(f"Latency: {latency:.2f} ms")


# =========================================================
# 📂 IMAGE UPLOAD
# =========================================================
st.subheader("📂 Upload Image")

uploaded_file = st.file_uploader("Upload PCB Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    frame = np.array(image)
    process_and_display(frame)

# =========================================================
# 📷 CAMERA CAPTURE
# =========================================================
st.subheader("📷 Capture from Camera")

camera_image = st.camera_input("Take a photo")

if camera_image is not None:
    file_bytes = np.asarray(bytearray(camera_image.read()), dtype=np.uint8)
    frame = cv2.imdecode(file_bytes, 1)
    process_and_display(frame)
