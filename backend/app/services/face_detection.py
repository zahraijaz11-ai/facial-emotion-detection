import cv2
import numpy as np
import mediapipe as mp


# ============================================================
# MEDIAPIPE FACE DETECTOR
# ============================================================

mp_face_detection = mp.solutions.face_detection

media_pipe_detector = mp_face_detection.FaceDetection(
    model_selection=1,
    min_detection_confidence=0.3
)


# ============================================================
# HAAR CASCADE FALLBACK
# ============================================================

# If MediaPipe cannot detect a face, we try OpenCV Haar Cascade
# as a second option.

haar_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)


# ============================================================
# DETECT FACES
# ============================================================

def detect_faces(image: np.ndarray):

    if image is None or image.size == 0:
        raise ValueError("Invalid image provided.")

    height, width = image.shape[:2]

    # --------------------------------------------------------
    # Convert BGR -> RGB for MediaPipe
    # --------------------------------------------------------

    rgb_image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    # --------------------------------------------------------
    # MediaPipe detection
    # --------------------------------------------------------

    results = media_pipe_detector.process(rgb_image)

    faces = []

    if results.detections:

        for detection in results.detections:

            bounding_box = detection.location_data.relative_bounding_box

            x = int(bounding_box.xmin * width)
            y = int(bounding_box.ymin * height)

            box_width = int(
                bounding_box.width * width
            )

            box_height = int(
                bounding_box.height * height
            )

            # ------------------------------------------------
            # Keep coordinates inside image
            # ------------------------------------------------

            x = max(0, x)
            y = max(0, y)

            box_width = min(
                box_width,
                width - x
            )

            box_height = min(
                box_height,
                height - y
            )

            if box_width > 0 and box_height > 0:

                faces.append([
                    x,
                    y,
                    box_width,
                    box_height
                ])

    # --------------------------------------------------------
    # If MediaPipe found faces, return them
    # --------------------------------------------------------

    if len(faces) > 0:
        return faces

    # ========================================================
    # HAAR FALLBACK
    # ========================================================

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    gray = cv2.equalizeHist(gray)

    haar_faces = haar_detector.detectMultiScale(
        gray,
        scaleFactor=1.05,
        minNeighbors=4,
        minSize=(20, 20)
    )

    if isinstance(haar_faces, tuple):
        return []

    return haar_faces.tolist()


# ============================================================
# CROP FACE
# ============================================================

def crop_face(image: np.ndarray, face):

    x, y, width, height = face

    cropped_face = image[
        y:y + height,
        x:x + width
    ]

    return cropped_face