import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, save_img

# Definimos las rutas
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
data_dir = os.path.join(base_dir, 'src', 'data', 'split')
train_dir = os.path.join(data_dir, 'train')

benign_dir = os.path.join(train_dir, 'benigno')
malign_dir = os.path.join(train_dir, 'maligno')

# Contamos las imágenes actuales
benign_images = os.listdir(benign_dir)
malign_images = os.listdir(malign_dir)

num_benign = len(benign_images)
num_malign = len(malign_images)

print(f"🔍 Benign images: {num_benign}")
print(f"🔍 Malign images: {num_malign}")

# ¿Cuántas necesitamos?
needed = num_malign - num_benign
if needed <= 0:
    print("✅ Ya hay suficientes imágenes benignas.")
    exit()

# Definimos el generador de augmentaciones
datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Creamos las augmentaciones
generated = 0
while generated < needed:
    for img_name in benign_images:
        img_path = os.path.join(benign_dir, img_name)
        img = load_img(img_path)
        x = img_to_array(img)
        x = np.expand_dims(x, axis=0)

        aug_iter = datagen.flow(x, batch_size=1)
        aug_img = next(aug_iter)[0].astype('uint8')

        new_img_name = f"aug_{generated}_{img_name}"
        save_img(os.path.join(benign_dir, new_img_name), aug_img)

        generated += 1
        if generated >= needed:
            break

print(f"✅ Se generaron {generated} imágenes nuevas para benignos.")
