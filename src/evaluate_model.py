import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import logging

from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")

def evaluate_model(model_path, val_path, class_indices_path):
    # Cargar modelo y clases
    model = load_model(model_path)
    class_indices = joblib.load(class_indices_path)
    class_labels = list(class_indices.keys())

    # Generador
    val_datagen = ImageDataGenerator(rescale=1. / 255)
    val_generator = val_datagen.flow_from_directory(
        val_path,
        target_size=(150, 150),
        batch_size=1,
        class_mode='binary',
        shuffle=False
    )

    # Predicciones
    Y_pred = model.predict(val_generator)
    y_pred = np.round(Y_pred).astype(int).flatten()
    y_true = val_generator.classes

    # Reporte
    logging.info("📊 Classification Report:")
    logging.info("\n" + classification_report(y_true, y_pred, target_names=class_labels))

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig("src/models/confusion_matrix.png")
    plt.close()

    # ROC Curve
    fpr, tpr, thresholds = roc_curve(y_true, Y_pred)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig("src/models/roc_curve.png")
    plt.close()

    return {
        "roc_auc": roc_auc,
        "confusion_matrix": cm.tolist()
    }

