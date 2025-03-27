from fastapi import FastAPI, UploadFile, File, HTTPException
import cv2
import numpy as np
from PIL import Image
import io
import tensorflow as tf
import os
from dotenv import load_dotenv
import requests
import shutil
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Adjust this if needed
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods (POST, GET, etc.)
    allow_headers=["*"],  # Allow all headers
)

# Face++ API Credentials
FACE_API_URL = "https://api-us.faceplusplus.com/facepp/v3/detect"
API_KEY = "twZOWirK5l8ymNhccVJF1uetKAzQvEHx"
API_SECRET = "OuRkSXNxDtW4BNKP-QiHzKRqPaD7ET7a"


FACEPP_API_KEY = os.getenv("FACEPP_API_KEY")
FACEPP_API_SECRET = os.getenv("FACEPP_API_SECRET")
FACEPP_URL = "https://api-us.faceplusplus.com/facepp/v3/detect"

# Define skin tone categories
SKIN_TONES = [
    ("Very Fair", (220, 200, 190)),
    ("Fair", (200, 180, 160)),
    ("Light-Medium", (180, 160, 140)),
    ("Medium", (160, 140, 120)),
    ("Tan", (140, 120, 100)),
    ("Deep Tan", (120, 100, 80)),
    ("Deep", (100, 80, 60)),
    ("Very Deep", (80, 60, 40)),
]


# Undertone classification reference
UNDERTONES = {
    "Cool": [(200, 180, 220), (180, 160, 200), (160, 140, 180)],
    "Warm": [(220, 200, 160), (200, 180, 140), (180, 160, 120)],
    "Neutral": [(190, 170, 150), (170, 150, 130), (150, 130, 110)]
}


def preprocess_image(image_bytes):
    """Preprocess image for model prediction."""
    image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    image = cv2.resize(image, (224, 224))
    image = np.expand_dims(image, axis=0) / 255.0  # Normalize
    return image


def classify_skin_tone(avg_color):
    """Determine closest skin tone match."""
    min_diff = float("inf")
    best_match = "Unknown"
    for tone, ref_color in SKIN_TONES:
        diff = np.linalg.norm(np.array(avg_color) - np.array(ref_color))
        if diff < min_diff:
            min_diff = diff
            best_match = tone
    return best_match


def classify_undertone(avg_lab_color):
    """Determine undertone based on closest RGB match."""
    avg_rgb_color = cv2.cvtColor(
        np.uint8([[avg_lab_color]]), cv2.COLOR_LAB2RGB
    )[0][0]

    best_match = "Neutral"
    min_diff = float("inf")

    for undertone, colors in UNDERTONES.items():
        for ref_color in colors:
            diff = np.linalg.norm(np.array(avg_rgb_color) - np.array(ref_color))
            if diff < min_diff:
                min_diff = diff
                best_match = undertone

    return best_match if min_diff < 60 else "Neutral"  # Adjust threshold




@app.post("/analyze/")
async def analyze_skin(file: UploadFile = File(...)):
    """Analyze skin tone and undertone from an uploaded image."""
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    np_image = np.array(image)

    # Convert image to LAB color space for better skin tone extraction
    lab_image = cv2.cvtColor(np_image, cv2.COLOR_RGB2LAB)
    avg_color = np.mean(lab_image.reshape(-1, 3), axis=0)

    # Convert LAB avg color to RGB before classification
    skin_tone = classify_skin_tone(avg_color)
    undertone = classify_undertone(avg_color)  # Now uses correct color format

    return {"skin_tone": skin_tone, "undertone": undertone}


#face++
def analyze_face(image_path):
    """Send image to Face++ API and analyze face attributes."""
    with open(image_path, "rb") as image_file:
        files = {"image_file": image_file}
        data = {
            "api_key": API_KEY,
            "api_secret": API_SECRET,
            "return_attributes": "gender,skinstatus"
        }
        response = requests.post(FACE_API_URL, data=data, files=files)
    
    if response.status_code == 200:
        return response.json()
    return None

@app.post("/detect/")
async def detect_face(file: UploadFile = File(...)):
    """Handles image upload and calls Face++ for analysis."""
    file_path = f"temp_{file.filename}"
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Analyze the image
    face_data = analyze_face(file_path)

    # Remove temp file
    os.remove(file_path)

    if not face_data or "faces" not in face_data or not face_data["faces"]:
        return {"error": "No face detected"}

    attributes = face_data["faces"][0]["attributes"]
    gender = attributes["gender"]["value"]
    skin_status = attributes.get("skinstatus", {})

    acne = skin_status.get("acne", "Unknown")
    dark_circles = skin_status.get("dark_circle", "Unknown")
    stain = skin_status.get("stain", "Unknown")

    return {
        "gender": gender,
        "acne": acne,
        "dark_circles": dark_circles,
        "stain": stain
    }


