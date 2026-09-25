from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes.emotion import router as emotion_router

app = FastAPI(
    title="Facial Emotion Identification API",
    version="1.0.0"
)

# Allow frontend to communicate with backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {
        "message": "Facial Emotion Identification API is running"
    }


app.include_router(
    emotion_router,
    prefix="/api/emotion",
    tags=["Emotion Detection"]
)