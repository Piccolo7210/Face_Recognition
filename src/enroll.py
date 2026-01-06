import os
import cv2
import numpy as np
from utils import ensure_dir

BASE = os.path.dirname(os.path.dirname(__file__))
MODELS = os.path.join(BASE, "models")
GALLERY = os.path.join(BASE, "data", "gallery")
DB = os.path.join(BASE, "db")

DETECTOR_MODEL = os.path.join(MODELS, "face_detection_yunet_2023mar.onnx")
RECOG_MODEL = os.path.join(MODELS, "face_recognition_sface_2021dec.onnx")

ensure_dir(DB)

detector = cv2.FaceDetectorYN.create(
    DETECTOR_MODEL,
    "",
    (320, 320),
    score_threshold=0.6,   # ⬅️ LOWERED
    nms_threshold=0.3,
    top_k=5000
)


recognizer = cv2.FaceRecognizerSF.create(RECOG_MODEL, "")

names = []
embeddings = []

for person in sorted(os.listdir(GALLERY)):
    person_dir = os.path.join(GALLERY, person)
    if not os.path.isdir(person_dir):
        continue

    for img_name in sorted(os.listdir(person_dir)):
        if not img_name.lower().endswith((".jpg", ".png", ".jpeg")):
            continue

        img_path = os.path.join(person_dir, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue

        h, w = img.shape[:2]
        target_w = 640
        scale = target_w / w
        target_h = int(h * scale)
        resized_img = cv2.resize(img, (target_w, target_h))
        detector.setInputSize((target_w, target_h))
        _, faces = detector.detect(resized_img)

        if faces is None or len(faces) == 0:
            print(f"No face found: {img_path}")
            continue

        face = faces[0].copy()
        face[:14] /= scale  # rescale to original image size
        aligned = recognizer.alignCrop(img, face)
        emb = recognizer.feature(aligned).flatten().astype(np.float32)
        

        names.append(person)
        embeddings.append(emb)

        print(f"Enrolled: {person} from {img_name}")

if not embeddings:
    raise RuntimeError("No faces enrolled. Add better images.")

np.savez_compressed(
    os.path.join(DB, "embeddings.npz"),
    embeddings=np.array(embeddings)
)

with open(os.path.join(DB, "names.txt"), "w", encoding="utf-8") as f:
    for n in names:
        f.write(n + "\n")

print("\n✅ Enrollment completed successfully")
