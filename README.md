**Estructura del Proyecto Fase 1**

mlops-brain-tumor/
│
├── env/                         # Entorno virtual (no incluido en el repo)
├── notebooks/                   # Notebooks para pruebas y EDA
├── src/
│   ├── data/
│   │   ├── BrainTumor/          # Imágenes originales
│   │   ├── data/                
│   │   ├── organized/           # Imágenes organizadas por clase
│   │   │   ├── benigno/
│   │   │   └── maligno/
│   │   ├── bt_dataset_t3.csv    # Dataset con nombres e información de las imágenes
│   │   ├── load_data.py
│   │   └── organize_images.py   # Script para clasificar imágenes por clase
│   ├── models/
│   └── utils/
│
├── Brain Tumor.csv              # Dataset original (opcional)
├── requirements.txt             # Librerías necesarias
├── README.md                    
├── .gitignore
└── ...

**Fase 1: Preparación del Proyecto**

 1.Clonar el repositorio

git clone https://github.com/tu-usuario/mlops-brain-tumor.git
cd mlops-brain-tumor

Asegurate de tener los branches main, develop y staging.

2.Crear y activar el entorno virtual

python3 -m venv env
source env/bin/activate  # En Windows: env\Scripts\activate

3.Instalar dependencias

pip install -r requirements.txt

4.Descargar y ubicar los datos

src/data/BrainTumor/

Y asegurate de tener el archivo: src/data/bt_dataset_t3.csv

5.Organizar las imágenes por clase

python src/data/organize_images.py

**Fase 2: Entrenamiento del modelo CNN.**

1. Preparar entorno y datos

Al tener las imágenes organizadas en carpetas benigno y maligno, permite usar ImageDataGenerator fácilmente.

**Fase 3: Automatización del reentrenamiento.**


🔁 División fija y reproducible	El conjunto de entrenamiento y validación es siempre el mismo.
👀 Transparencia	Podés ver cuántas imágenes hay en cada clase y carpeta.
📊 Métricas comparables	Los resultados entre modelos son más consistentes.
📤 Reutilizable	Podés compartir ese split con otros sin que cambien los resultados.
🔧 Más control	Podés aplicar distintas proporciones, técnicas o validación cruzada si querés.

**Fase 4: API o interfaz para predicción.**

**Fase 5: Contenerización con Docker.**

**Fase 6: CI/CD con GitHub Actions.**

**Fase 7: Documentación final y entrega.**






