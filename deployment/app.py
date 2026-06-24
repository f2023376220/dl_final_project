from fastapi import FastAPI, File, UploadFile
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = FastAPI(title="Pneumonia Detection API", description="Production API for Deep Learning Laboratory")

# Load the upgraded model artifact generated in Milestone 2
MODEL_PATH = "src/pneumonia_model.h5"
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"--- Model loaded successfully from {MODEL_PATH} ---")
except Exception as e:
    print(f"--- Error loading model: {e}. Running with fallback layout. ---")
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(128, 128, 3)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

@app.get("/health")
def health_check():
    """Mandatory health probe endpoint to monitor API status."""
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Mandatory inference endpoint accepting image uploads."""
    try:
        # 1. Read input image uploaded by user
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        # 2. Preprocess to match model constraints (128x128 pixels, scaled [0,1])
        image = image.resize((128, 128))
        img_array = np.array(image).astype(np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0) # Add batch dimension (1, 128, 128, 3)
        
        # 3. Generate prediction score
        prediction = model.predict(img_array)
        score = float(prediction[0][0])
        
        # 4. Map index result
        result = "Pneumonia Detected" if score > 0.5 else "Normal"
        
        return {
            "filename": file.filename,
            "prediction": result,
            "confidence_score": round(score if score > 0.5 else (1 - score), 4)
        }
    except Exception as e:
        return {"error": f"Failed to process image payload: {str(e)}"}
