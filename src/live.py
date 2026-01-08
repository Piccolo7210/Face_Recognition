import os
import cv2
import numpy as np
from db.db import get_connection
from .utils import draw_label
def parse_pgvector(v):
    """
    Convert pgvector string like '[1,2,3]' → np.ndarray
    """
    if isinstance(v, list):
        return np.array(v, dtype=np.float32)

    v = v.strip("[]")
    return np.array([float(x) for x in v.split(",")], dtype=np.float32)

BASE = os.path.dirname(os.path.dirname(__file__))
MODELS = os.path.join(BASE, "models")

DETECTOR_MODEL = os.path.join(MODELS, "face_detection_yunet_2023mar.onnx")
RECOG_MODEL = os.path.join(MODELS, "face_recognition_sface_2021dec.onnx")

detector = cv2.FaceDetectorYN.create(
    DETECTOR_MODEL, "", (320, 320), 0.9, 0.3, 5000
)
recognizer = cv2.FaceRecognizerSF.create(RECOG_MODEL, "")

conn = get_connection()
cur = conn.cursor()

# Load all persons ONCE
cur.execute("SELECT name, national_id, embedding FROM persons")
rows = cur.fetchall()

cur.close()
conn.close()

names = []
db_embeddings = []

for name, nid, emb in rows:
    names.append(f"{name} ({nid})")
    db_embeddings.append(parse_pgvector(emb))

THRESHOLD = 0.5
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Camera not available")

print("Press ESC to exit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    scale = 640 / w
    resized = cv2.resize(frame, (640, int(h * scale)))

    detector.setInputSize((640, resized.shape[0]))
    _, faces = detector.detect(resized)

    if faces is not None:
        for face in faces:
            face = face.copy()
            face[:14] /= scale

            aligned = recognizer.alignCrop(frame, face)
            emb = recognizer.feature(aligned).flatten().astype(np.float32)

            best_score = -1
            best_name = "Unknown"

            for name, db_emb in zip(names, db_embeddings):
                score = recognizer.match(
                    emb.reshape(1, -1),
                    db_emb.reshape(1, -1),
                    cv2.FaceRecognizerSF_FR_COSINE
                )
                if score > best_score:
                    best_score = score
                    best_name = name

            if best_score < THRESHOLD:
                best_name = "Unknown"

            x, y, bw, bh = map(int, face[:4])
            cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
            draw_label(frame, f"{best_name} ({best_score:.2f})", x, y - 10)

    cv2.imshow("Live Face Recognition (pgvector)", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
