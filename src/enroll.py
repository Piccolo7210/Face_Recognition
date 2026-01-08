import os
import shutil
import cv2
import numpy as np
from tkinter import Tk, filedialog

from db.db import get_connection

# ------------------ PATHS ------------------

BASE = os.path.dirname(os.path.dirname(__file__))
MODELS = os.path.join(BASE, "models")
GALLERY = os.path.join(BASE, "data", "gallery")

os.makedirs(GALLERY, exist_ok=True)

DETECTOR_MODEL = os.path.join(MODELS, "face_detection_yunet_2023mar.onnx")
RECOG_MODEL = os.path.join(MODELS, "face_recognition_sface_2021dec.onnx")

# ------------------ MODELS ------------------

detector = cv2.FaceDetectorYN.create(
    DETECTOR_MODEL, "", (320, 320), 0.6, 0.3, 5000
)
recognizer = cv2.FaceRecognizerSF.create(RECOG_MODEL, "")

# ------------------ USER INPUT ------------------

print("\n=== FACE ENROLLMENT ===\n")

name = input("Name: ").strip()
age = int(input("Age: ").strip())
national_id = input("National ID: ").strip()

person_dir = os.path.join(GALLERY, name)
os.makedirs(person_dir, exist_ok=True)

# ------------------ IMAGE UPLOAD PROMPT ------------------

print("\nSelect images for enrollment (Ctrl / Shift for multiple selection)")

Tk().withdraw()  # hide root window
image_paths = filedialog.askopenfilenames(
    title="Select face images",
    filetypes=[("Image Files", "*.jpg *.jpeg *.png")]
)

if not image_paths:
    raise RuntimeError("No images selected.")

# ------------------ COPY IMAGES ------------------

saved_paths = []
for idx, src_path in enumerate(image_paths, start=1):
    ext = os.path.splitext(src_path)[1]
    dst_path = os.path.join(person_dir, f"img_{idx:03d}{ext}")
    shutil.copy(src_path, dst_path)
    saved_paths.append(dst_path)

print(f"\nSaved {len(saved_paths)} images to {person_dir}")

# ------------------ FACE EMBEDDING ------------------

embeddings = []

for img_path in saved_paths:
    img = cv2.imread(img_path)
    if img is None:
        continue

    h, w = img.shape[:2]
    scale = 640 / w
    resized = cv2.resize(img, (640, int(h * scale)))

    detector.setInputSize((640, resized.shape[0]))
    _, faces = detector.detect(resized)

    if faces is None:
        print(f"No face detected: {os.path.basename(img_path)}")
        continue

    face = faces[0].copy()
    face[:14] /= scale

    aligned = recognizer.alignCrop(img, face)
    emb = recognizer.feature(aligned).flatten().astype(np.float32)

    embeddings.append(emb)

if not embeddings:
    raise RuntimeError("No valid faces found in selected images.")

# ------------------ AVERAGE EMBEDDING ------------------

avg_embedding = np.mean(embeddings, axis=0)

# ------------------ DATABASE INSERT ------------------

conn = get_connection()
cur = conn.cursor()

cur.execute(
    """
    INSERT INTO persons (name, age, national_id, embedding)
    VALUES (%s, %s, %s, %s)
    """,
    (name, age, national_id, avg_embedding.tolist())
)

conn.commit()
cur.close()
conn.close()

print("\n✅ Enrollment completed successfully")
