import os
import sys
import torch
import numpy as np
from PIL import Image
import cv2
from torchvision import transforms

# ---------------- Paths ----------------
repo_path = "Self-Correction-Human-Parsing"
input_dir = "data/"
body_mask_dir = "outputs/parsing/body_mask/"
reserved_dir = "outputs/parsing/reserved_regions/"
clothing_dir = "outputs/parsing/clothing_mask/"

os.makedirs(body_mask_dir, exist_ok=True)
os.makedirs(reserved_dir, exist_ok=True)
os.makedirs(clothing_dir, exist_ok=True)

# ---------------- Add repo root to Python path ----------------
sys.path.append(repo_path)

# ---------------- Import SCHP ----------------
from utils.schp import SCHP

# ---------------- Device ----------------
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    print("CUDA not found. Running on CPU (slower).")
    device = torch.device("cpu")

# ---------------- Load Model ----------------
model = SCHP(num_classes=20)  # LIP dataset has 20 classes

checkpoint_path = os.path.join(repo_path, "pretrained_models/SCHP_56_LIP.pth")
checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint['state_dict'])
model.to(device)
model.eval()

# ---------------- Transform ----------------
transform = transforms.Compose([
    transforms.Resize((256, 192)),
    transforms.ToTensor()
])

# ---------------- Segment IDs ----------------
face_hair_ids = [1, 2]             # Face + Hair
upper_cloth_ids = [5, 6, 7, 8]     # Upper clothing IDs
lower_cloth_ids = [9, 10]          # Lower clothing IDs

# ---------------- Process all images ----------------
image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

if not image_files:
    print("No images found in data/")
else:
    for img_file in image_files:
        img_path = os.path.join(input_dir, img_file)
        img = Image.open(img_path).convert("RGB")
        img_tensor = transform(img).unsqueeze(0).to(device)

        # Forward pass
        with torch.no_grad():
            parsing_map = model(img_tensor)[0].argmax(0).cpu().numpy()

        # ---------- Body Mask ----------
        body_mask = np.where(parsing_map > 0, 255, 0).astype(np.uint8)
        body_mask_img = Image.fromarray(body_mask).resize(img.size)
        body_mask_img.save(os.path.join(body_mask_dir, img_file))

        # ---------- Reserved Regions (Face + Hair) ----------
        reserved_mask = np.isin(parsing_map, face_hair_ids).astype(np.uint8) * 255
        reserved_mask_img = Image.fromarray(reserved_mask).resize(img.size)
        reserved_rgb = cv2.bitwise_and(np.array(img), np.array(img), mask=np.array(reserved_mask_img))
        cv2.imwrite(os.path.join(reserved_dir, img_file), reserved_rgb)

        # ---------- Clothing Mask ----------
        clothing_mask = np.isin(parsing_map, upper_cloth_ids + lower_cloth_ids).astype(np.uint8) * 255
        clothing_mask_img = Image.fromarray(clothing_mask).resize(img.size)
        clothing_mask_img.save(os.path.join(clothing_dir, img_file))

        print(f"Processed {img_file}: Body Mask + Reserved Regions + Clothing Mask")

print("All images processed successfully.")
