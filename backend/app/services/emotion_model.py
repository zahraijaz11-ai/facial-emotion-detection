import os
import numpy as np
import tensorflow as tf


# ============================================================
# MODEL PATH
# ============================================================

# emotion_model.py is inside:
# backend/app/services/
#
# We need to go:
# services -> app -> backend

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../..")
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "emotion_model_v24_best.keras"
)


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 60)
print("Loading V24 Emotion Model...")
print("=" * 60)
print("Model path:", MODEL_PATH)

model = tf.keras.models.load_model(MODEL_PATH)

print("V24 model loaded successfully!")
print("Model input shape:", model.input_shape)
print("=" * 60)


# ============================================================
# EMOTION LABELS
# ============================================================

EMOTION_LABELS = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Neutral",
    "Sad",
    "Surprise"
]


# ============================================================
# PREDICT EMOTION
# ============================================================

def predict_emotion(face):

    predictions = model.predict(face, verbose=0)[0]

    print("\nEmotion probabilities:")

    for label, probability in zip(EMOTION_LABELS, predictions):
        print(f"{label}: {probability * 100:.2f}%")

    predicted_index = np.argmax(predictions)

    confidence = float(predictions[predicted_index])

    emotion = EMOTION_LABELS[predicted_index]

    return emotion, confidence