from fastapi import APIRouter, UploadFile, File, HTTPException
import cv2
import numpy as np

from app.services.face_detection import detect_faces, crop_face
from app.services.preprocessing import preprocess_face
from app.services.emotion_model import predict_emotion


router = APIRouter()


@router.post("/predict")
async def predict(file: UploadFile = File(...)):

    # Check file type
    if file.content_type not in [
        "image/jpeg",
        "image/png",
        "image/webp"
    ]:
        raise HTTPException(
            status_code=400,
            detail="Please upload a JPG, PNG, or WebP image."
        )

    # Read uploaded image
    contents = await file.read()

    image_array = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(
            status_code=400,
            detail="Invalid image."
        )

    # Detect faces
    faces = detect_faces(image)

    if len(faces) == 0:
        raise HTTPException(
            status_code=400,
            detail="No face detected in the image."
        )

    # Use the first detected face
    face = crop_face(image, faces[0])

    # Preprocess face
    processed_face = preprocess_face(face)

    # Predict emotion
    emotion, confidence = predict_emotion(processed_face)

    return {
        "emotion": emotion,
        "confidence": round(confidence * 100, 2)
    }