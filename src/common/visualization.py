"""Save PNG overlays and display pointer and scale masks consistently."""
import numpy as np


PALETTE = np.array([[0, 0, 0], [255, 255, 0], [0, 255, 0]], dtype=np.uint8)  # BGR
OPACITY = np.array([0.0, 0.80, 0.45], dtype=np.float32)


def save_png(path, image):
    import cv2

    ok, encoded = cv2.imencode('.png', image)
    if not ok:
        raise OSError(f'Could not encode PNG: {path}')
    path.write_bytes(encoded.tobytes())


def blend_labels(image, labels):
    foreground = labels != 0
    classes = labels[foreground]
    opacity = OPACITY[classes, None]
    colors = PALETTE[classes]
    image[foreground] = np.rint(image[foreground] * (1 - opacity) + colors * opacity).astype(np.uint8)
