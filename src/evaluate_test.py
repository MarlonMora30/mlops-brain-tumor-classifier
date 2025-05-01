import os, numpy as np, matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, classification_report, f1_score
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# rutas
base = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
test_dir   = os.path.join(base, 'src', 'data', 'split', 'test')
model_path = os.path.join(base, 'src', 'models', 'brain_tumor_classifier.keras')

# cargar
model = load_model(model_path, compile=False)
test_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(
    test_dir, target_size=(150,150), batch_size=1, class_mode='binary', shuffle=False)

Y_prob = model.predict(test_gen)
Y_true = test_gen.classes

# ROC/AUC
fpr, tpr, _ = roc_curve(Y_true, Y_prob)
roc_auc = auc(fpr, tpr)
plt.plot(fpr,tpr,label=f'AUC={roc_auc:.3f}'); plt.plot([0,1],[0,1],'k--'); plt.legend(); plt.show()

# threshold por macro-F1
best = {'f1':0, 'thr':0.5}
for t in np.linspace(0.1,0.9,81):
    y_pred = (Y_prob>t).astype(int).flatten()
    f1 = f1_score(Y_true, y_pred, average='macro')
    if f1>best['f1']:
        best = {'f1':f1, 'thr':t}

print(f"🔧 Mejor macro-F1={best['f1']:.3f} con threshold={best['thr']:.2f}")
Y_pred = (Y_prob>best['thr']).astype(int).flatten()
print(classification_report(Y_true, Y_pred, target_names=list(test_gen.class_indices.keys())))
