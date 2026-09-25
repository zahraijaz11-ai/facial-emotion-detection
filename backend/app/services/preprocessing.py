import cv2
import numpy as np


def preprocess_face(
    face: np.ndarray,
    image_size=(75, 75),
    grayscale=True
):
    if face is None or face.size == 0:
        raise ValueError("Face image is empty.")

    # Resize to V20 model input size
    face = cv2.resize(face, image_size)

    # Convert to grayscale
    if grayscale:
        if len(face.shape) == 3:
            face = cv2.cvtColor(
                face,
                cv2.COLOR_BGR2GRAY
            )

    # Normalize pixel values from 0-255 to 0-1
    face = face.astype(np.float32) / 255.0

    # Add channel dimension
    # (75, 75) -> (75, 75, 1)
    if grayscale:
        face = np.expand_dims(face, axis=-1)

    # Add batch dimension
    # (75, 75, 1) -> (1, 75, 75, 1)
    face = np.expand_dims(face, axis=0)

    return face