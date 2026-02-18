import cv2
import glob
import random
import os

# Output video name
output_video = "sample_production_video.mp4"

# Collect images from both folders
good_images = glob.glob("data/test/good/*")
defect_images = glob.glob("data/test/defective/*")

all_images = good_images + defect_images

if len(all_images) == 0:
    print("No images found!")
    exit()

# Shuffle for random production simulation
random.shuffle(all_images)

# Read first image to get size
frame = cv2.imread(all_images[0])
height, width, layers = frame.shape

# Create VideoWriter
video = cv2.VideoWriter(
    output_video,
    cv2.VideoWriter_fourcc(*'mp4v'),
    5,  # FPS (speed of production line)
    (width, height)
)

print("Creating synthetic production video...")

for img_path in all_images:
    frame = cv2.imread(img_path)
    frame = cv2.resize(frame, (width, height))
    video.write(frame)

video.release()

print("Video created successfully!")
print("Saved as:", output_video)
