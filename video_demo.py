import cv2
import numpy as np
import tensorflow as tf
import time

model = tf.keras.models.load_model("quality_model.h5")

IMAGE_SIZE = (224, 224)

cap = cv2.VideoCapture("sample_production_video.mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img = cv2.resize(frame, IMAGE_SIZE)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    start = time.time()
    prediction = model.predict(img, verbose=0)
    latency = (time.time() - start) * 1000

    label = "DEFECT" if prediction[0][0] > 0.5 else "PASS"

    cv2.putText(frame,
                f"{label} | {latency:.2f} ms",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 0, 0),
                2)

    cv2.imshow("Production QC Demo", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
