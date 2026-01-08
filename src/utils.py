import numpy as np
import cv2

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
