from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
import io
import os
import joblib

# --- Configuración inicial ---
app = FastAPI(
    title="Brain Tumor Classifier API",
    description="Clasificación de imágenes de tumores cerebrales (benigno vs maligno)",
    version="1.0.0"
)

# --- Cargar modelo y clases ---
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'brain_tumor_classifier.keras')
CLASS_PATH = os.path.join(BASE_DIR, 'models', 'class_indices.pkl')

try:
    model = load_model(MODEL_PATH)
    class_indices = joblib.load(CLASS_PATH)
    idx_to_class = {v: k for k, v in class_indices.items()}
except Exception as e:
    raise RuntimeError(f"Error cargando modelo o clases: {e}")

IMG_HEIGHT, IMG_WIDTH = 150, 150


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Validar tipo MIME
        if file.content_type not in ["image/jpeg", "image/png"]:
            raise HTTPException(status_code=400, detail="Formato no válido. Solo se permiten JPG o PNG.")

        # Leer imagen
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image = image.resize((IMG_WIDTH, IMG_HEIGHT))
        img_array = img_to_array(image) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predicción
        prediction = model.predict(img_array)[0][0]
        label_idx = int(round(prediction))
        label = idx_to_class[label_idx]
        confidence = float(prediction) if label_idx == 1 else float(1 - prediction)

        return JSONResponse({
            "prediction": label,
            "confidence": round(confidence, 3)
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al procesar la imagen: {str(e)}")
