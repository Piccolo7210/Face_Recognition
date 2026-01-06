import os
import cv2
import numpy as np
from utils import cosine_similarity, draw_label

BASE = os.path.dirname(os.path.dirname(__file__))
MODELS = os.path.join(BASE, "models")
DB = os.path.join(BASE, "db")

DETECTOR_MODEL = os.path.join(MODELS, "face_detection_yunet_2023mar.onnx")
RECOG_MODEL = os.path.join(MODELS, "face_recognition_sface_2021dec.onnx")

data = np.load(os.path.join(DB, "embeddings.npz"))
db_embeddings = data["embeddings"]
names = open(os.path.join(DB, "names.txt"), encoding="utf-8").read().splitlines()

detector = cv2.FaceDetectorYN.create(
    DETECTOR_MODEL, "", (320, 320), 0.9, 0.3, 5000
)
recognizer = cv2.FaceRecognizerSF.create(RECOG_MODEL, "")

THRESHOLD = 0.45  # adjust if needed

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Camera not available")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    target_w = 640
    scale = target_w / w
    target_h = int(h * scale)
    resized_frame = cv2.resize(frame, (target_w, target_h))
    detector.setInputSize((target_w, target_h))
    _, faces = detector.detect(resized_frame)

    if faces is not None:
        for face in faces:
            x, y, bw, bh = face[:4].astype(int)
            face = face.copy()
            face[:14] /= scale  # rescale to original image size

            aligned = recognizer.alignCrop(frame, face)
            emb = recognizer.feature(aligned).flatten().astype(np.float32)

            best_name = "Unknown"
            best_score = -1.0

            for name, db_emb in zip(names, db_embeddings):
                score = recognizer.match(
                    emb.reshape(1,-1),
                    db_emb.reshape(1,-1),
                    cv2.FaceRecognizerSF_FR_COSINE
                )
                if score > best_score:
                    best_score = score
                    best_name = name

            if best_score < THRESHOLD:
                best_name = "Unknown"

            cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
            draw_label(frame, f"{best_name} ({best_score:.2f})", x, y - 10)
            print("DEBUG SCORE:", best_score)

    cv2.imshow("Live Face Recognition (Python 3.14)", frame)
    # cv2.imshow("Aligned Face", aligned)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
