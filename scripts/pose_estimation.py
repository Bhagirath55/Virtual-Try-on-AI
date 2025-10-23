import cv2
import mediapipe as mp
import numpy as np
import os

# Paths
input_dir = "data/"
output_base_dir = "outputs/pose/"
os.makedirs(output_base_dir, exist_ok=True)

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

# Get all image files in the input directory
image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

if not image_files:
    print("No images found in the data/ folder.")
else:
    for img_file in image_files:
        img_path = os.path.join(input_dir, img_file)
        image = cv2.imread(img_path)
        if image is None:
            print(f"Skipping {img_file}, unable to read.")
            continue

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width, _ = image.shape

        # Process image to get keypoints
        results = pose.process(image_rgb)

        # Create output folder per image
        img_name = os.path.splitext(img_file)[0]
        output_dir = os.path.join(output_base_dir, img_name)
        os.makedirs(output_dir, exist_ok=True)

        # Generate and save 18 heatmaps
        for i in range(18):
            heatmap = np.zeros((height, width), dtype=np.uint8)
            if results.pose_landmarks:
                x = int(results.pose_landmarks.landmark[i].x * width)
                y = int(results.pose_landmarks.landmark[i].y * height)
                cv2.rectangle(heatmap, (x-2, y-2), (x+2, y+2), 255, -1)
            cv2.imwrite(os.path.join(output_dir, f"keypoint_{i}.png"), heatmap)

        print(f"Processed {img_file}, heatmaps saved in {output_dir}")

print("All images processed.")
