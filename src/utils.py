import os
import numpy as np
import cv2

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def l2_normalize(v):
    return v / np.linalg.norm(v)

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = l2_normalize(a)
    b = l2_normalize(b)
    return float(np.dot(a, b))

def draw_label(img, text, x, y):
    cv2.putText(
        img,
        text,
        (x, max(20, y)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2,
        cv2.LINE_AA
    )
