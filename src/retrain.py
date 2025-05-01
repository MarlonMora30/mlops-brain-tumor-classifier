import os
import logging
import argparse
import numpy as np
import json
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.utils.class_weight import compute_class_weight
import joblib

from utils.evaluation import evaluate_model  # ✅ Módulo de evaluación

# ----------------- Logging -----------------
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

# ----------------- Argumentos CLI -----------------
parser = argparse.ArgumentParser(description="Reentrenar modelo de tumor cerebral")
parser.add_argument('--epochs', type=int, default=30, help='Número de épocas')
args = parser.parse_args()

# ----------------- Paths -----------------
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
data_dir = os.path.join(base_dir, 'src', 'data', 'split')
model_dir = os.path.join(base_dir, 'src', 'models')
os.makedirs(model_dir, exist_ok=True)

train_path = os.path.join(data_dir, 'train')
val_path = os.path.join(data_dir, 'val')

# ----------------- Validaciones -----------------
if not os.path.exists(train_path) or not os.path.exists(val_path):
    logging.error("Las carpetas de train/val no existen.")
    exit(1)

if not os.listdir(train_path) or not os.listdir(val_path):
    logging.error("Las carpetas están vacías.")
    exit(1)

logging.info("📂 Datos validados correctamente.")

# ----------------- Parámetros -----------------
IMG_HEIGHT = 150
IMG_WIDTH = 150
BATCH_SIZE = 32

# ----------------- Data Generators -----------------
train_datagen = ImageDataGenerator(rescale=1. / 255)
val_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

val_generator = val_datagen.flow_from_directory(
    val_path,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

# ----------------- Class Weights -----------------
labels = train_generator.classes
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(labels),
    y=labels
)
class_weights = dict(enumerate(class_weights))
logging.info(f"🧮 Class Weights: {class_weights}")

# ----------------- Modelo -----------------
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# ----------------- Entrenamiento -----------------
logging.info(f"Entrenando por {args.epochs} épocas...")
history = model.fit(
    train_generator,
    epochs=args.epochs,
    validation_data=val_generator,
    class_weight=class_weights,
    callbacks=[early_stop]
)

# ----------------- Guardar modelo -----------------
model_path = os.path.join(model_dir, 'brain_tumor_classifier.keras')
model.save(model_path)
logging.info(f"✅ Modelo guardado en: {model_path}")

# ----------------- Guardar class indices -----------------
class_indices = train_generator.class_indices
class_indices_path = os.path.join(model_dir, 'class_indices.pkl')
joblib.dump(class_indices, class_indices_path)
logging.info(f"✅ Class indices guardados en: {class_indices_path}")

# ----------------- Evaluar -----------------
logging.info("🔍 Evaluando modelo en conjunto de validación...\n")
metrics = evaluate_model(model_path, val_path, class_indices_path)
logging.info(f"✅ AUC: {metrics['roc_auc']:.2f}")

# ----------------- Guardar métricas en JSON para DVC -----------------
metrics_path = os.path.join(base_dir, 'metrics.json')
with open(metrics_path, 'w') as f:
    json.dump(metrics, f, indent=4)
print(f"Métricas guardadas en: {metrics_path}")
