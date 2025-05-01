import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, save_img

# Se aplicó un oversampling:

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
val_benign = os.path.join(base_dir, 'src', 'data', 'split', 'val', 'benigno')
val_malign = os.path.join(base_dir, 'src', 'data', 'split', 'val', 'maligno')

n_ben = len(os.listdir(val_benign))
n_mal = len(os.listdir(val_malign))
needed = n_mal - n_ben

datagen = ImageDataGenerator(
  rotation_range=40, width_shift_range=0.2, height_shift_range=0.2,
  shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode='nearest'
)

gen=0
orig = os.listdir(val_benign)
while gen < needed:
  for fn in orig:
    img = load_img(os.path.join(val_benign, fn))
    x = img_to_array(img)[None,...]
    batch = next(datagen.flow(x, batch_size=1))
    save_img(os.path.join(val_benign, f"vaug{gen}_{fn}"), batch[0].astype('uint8'))
    gen+=1
    if gen>=needed: break

print(f"✅ Generadas {gen} benignas para validación")
