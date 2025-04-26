import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import pandas as pd

# --- Configuración de rutas ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VAL_DIR = os.path.join(BASE_DIR, 'data', 'split', 'val')
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'brain_tumor_classifier.keras')
REPORT_CSV = os.path.join(BASE_DIR, 'models', 'evaluation_results.csv')

# --- Parámetros ---
IMG_HEIGHT = 150
IMG_WIDTH = 150
BATCH_SIZE = 32

# --- Generador de validación ---
datagen = ImageDataGenerator(rescale=1. / 255)
val_generator = datagen.flow_from_directory(
    VAL_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

# --- Cargar modelo ---
model = load_model(MODEL_PATH)

# --- Predicciones ---
pred_probs = model.predict(val_generator)
pred_classes = (pred_probs > 0.5).astype(int).flatten()
true_classes = val_generator.classes
class_labels = list(val_generator.class_indices.keys())

# --- Clasification report como dict ---
report_dict = classification_report(true_classes, pred_classes, target_names=class_labels, output_dict=True)

# --- Guardar en CSV ---
report_df = pd.DataFrame(report_dict).transpose()
report_df.to_csv(REPORT_CSV, index=True)
print(f"\n✅ Resultados guardados en: {REPORT_CSV}")

# --- Mostrar por consola también ---
print("\n📊 Clasification Report:")
print(report_df)
print("\n🧩 Confusion Matrix:")
print(confusion_matrix(true_classes, pred_classes))
