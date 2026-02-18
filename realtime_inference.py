from datetime import datetime
import cv2
import time
import numpy as np
import tensorflow as tf

# Load trained model
model = tf.keras.models.load_model("quality_model.h5")

IMAGE_SIZE = (224, 224)

def predict_frame(frame):
    img = cv2.resize(frame, IMAGE_SIZE)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    start_time = time.time()
    prediction = model.predict(img, verbose=0)
    end_time = time.time()

    latency = (end_time - start_time) * 1000  # ms

    label = "DEFECT" if prediction[0][0] > 0.5 else "PASS"
    confidence = float(prediction[0][0])

    return label, confidence, latency


# Simulate live camera (webcam)
cap = cv2.VideoCapture(0)

print("Press Q to exit...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    label, confidence, latency = predict_frame(frame)

    text = f"{label} | {confidence:.2f} | {latency:.2f} ms"

    # 🔥 Save defect frames
    if label == "DEFECT":
        filename = f"defect_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        cv2.imwrite(filename, frame)
        print("Defect frame saved:", filename)

    # Show text on frame
    cv2.putText(frame,
                text,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2)

    cv2.imshow("VisionSpec QC - Live", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
