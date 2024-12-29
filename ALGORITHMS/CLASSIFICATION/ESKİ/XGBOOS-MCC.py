import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import glob
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, matthews_corrcoef
import xgboost as xgb
import os

# 1. Eğitim Verilerini Yükleme (Normal ve Hastalıklı)
normal_paths = glob.glob(r'C:\Users\elfal\Desktop\btbs dersler\PROJELER\VERI-SET\VERILER\HAM\OGRENME\NORMAL\*.jpg')
diseased_paths = glob.glob(r'C:\Users\elfal\Desktop\btbs dersler\PROJELER\VERI-SET\VERILER\HAM\OGRENME\OCSS\*.jpg')

# Tüm dosya yollarını birleştirme
file_paths = normal_paths + diseased_paths

# Dosyaların varlığını kontrol et
if len(file_paths) == 0:
    print("Hiçbir görüntü bulunamadı. Dosya yolunu kontrol edin.")
    exit()

# Görüntüleri yükle ve boyutlandır (gri tonlamalı olarak açıyoruz)
images = []
for file in file_paths:
    try:
        img = Image.open(file).convert("RGB").resize((224, 224))  # "L" gri tonlama modu
        images.append(img)
    except Exception as e:
        print(f"{file} yüklenemedi. Hata: {e}")

# Görüntülerin başarılı şekilde yüklendiğini kontrol et
if len(images) == 0:
    print("Hiçbir görüntü işlenemedi.")
    exit()

# 2. Eğitim Verilerini numpy dizisine dönüştürme
image_arrays = np.array([np.array(img).flatten() for img in images])
print("Eğitim görüntü dizisi boyutları:", image_arrays.shape)

# 3. Eğitim verisini ölçekleme
scaler = StandardScaler()
images_scaled = scaler.fit_transform(image_arrays)

# 4. PCA uygulama (Daha fazla bileşen kullanıyoruz)
pca = PCA(n_components=100)  # PCA ile daha fazla boyut indirgeme
images_pca = pca.fit_transform(images_scaled)
print("PCA sonrası eğitim verisi boyutları:", images_pca.shape)

# 5. Etiketleri tanımlama (Normal: 0, Hastalıklı: 1)
labels = np.array([0 if 'NORMAL' in file else 1 for file in file_paths])  # Etiketleme

# 6. Veriyi eğitim ve test olarak ayırma
X_train, X_test, y_train, y_test = train_test_split(images_pca, labels, test_size=0.2, stratify=labels, random_state=42)

# 7. XGBoost Modeli (Hyperparameter Tuning)
xgb_model = xgb.XGBClassifier(
    use_label_encoder=False, 
    eval_metric='logloss',
    learning_rate=0.05,  # Öğrenme oranını düşürüyoruz
    max_depth=6,  # Ağaç derinliğini sınırlandırıyoruz
    n_estimators=1000,  # Daha fazla ağaç
    subsample=0.8,  # Veri örnekleme oranı
    colsample_bytree=0.8  # Her ağacın için örnekleme oranı
)

# Eğitim verisiyle model eğitimi
xgb_model.fit(X_train, y_train)

# 8. Test Verilerini Dışarıdan Sağlama (Test verilerinizin dosya yolları)
test_normal_paths = glob.glob(r'C:\Users\burha\Desktop\VERI-SET\VERILER\KERNEL\TEST\NORMAL\*.jpg')
test_diseased_paths = glob.glob(r'C:\Users\burha\Desktop\VERI-SET\VERILER\KERNEL\TEST\OSCC\*.jpg')

# Test verisi dosya yollarını birleştirme
test_file_paths = test_normal_paths + test_diseased_paths

# Test verilerini yükleyip boyutlandırma
test_images = []
for file in test_file_paths:
    try:
        img = Image.open(file).convert("RGB").resize((224, 224))  # Boyutları aynı yapıyoruz
        test_images.append(img)
    except Exception as e:
        print(f"{file} yüklenemedi. Hata: {e}")

# Test verilerini numpy dizisine dönüştürme
test_image_arrays = np.array([np.array(img).flatten() for img in test_images])
print("Test verisi dizisi boyutları:", test_image_arrays.shape)

# Test verisini ölçekleme (Eğitim verisiyle aynı ölçekleme)
test_images_scaled = scaler.transform(test_image_arrays)  # Bu işlem, eğitim verisiyle aynı ölçeklemeyi kullanarak yapılır

# PCA uygulama (Eğitim verisiyle aynı bileşenleri kullanarak)

test_images_pca = pca.transform(test_images_scaled)  # PCA'dan geçen test verisi
print("Test verisi sonrası PCA boyutları:", test_images_pca.shape)

# 9. Test Etiketlerini Tanımlama (Normal: 0, Hastalıklı: 1)
test_labels = np.array([0 if 'NORMAL' in file else 1 for file in test_file_paths])  # Test etiketlerini belirleme

# 10. Tahminler ve Performans Analizi
y_pred_test = xgb_model.predict(test_images_pca)  # Test verileri ile tahmin yapma
print("Test Doğruluk Skoru:", accuracy_score(test_labels, y_pred_test))
print("Test Sınıflandırma Raporu:\n", classification_report(test_labels, y_pred_test))

# Matthew's Correlation Coefficient (MCC) hesaplama
mcc_score = matthews_corrcoef(test_labels, y_pred_test)
print(f"Matthew's Correlation Coefficient: {mcc_score:.4f}")

# 11. Yeniden Yapılandırılmış Görüntüleri Kaydetme (isteğe bağlı)
images_reconstructed = pca.inverse_transform(images_pca)
images_reconstructed = scaler.inverse_transform(images_reconstructed)
save_path = r'C:\Users\burha\Desktop\VERI-SET\VERILER\CIKTILAR\HAM-DOGRU'
if not os.path.exists(save_path):
    os.makedirs(save_path)

# Yeniden yapılandırılmış ilk 5 görüntüyü kaydetme
for i, img_array in enumerate(images_reconstructed[:5]):
    img = Image.fromarray(np.uint8(img_array.reshape(224, 224)))  # 224x224 boyutunda yeniden yapılandırılmış görüntüler
    img.save(f'{save_path}/reconstructed_image_{i}.jpg')

# 12. Yanlış Sınıflandırmaları Görselleştirme ve Kaydetme
misclassified_indices = np.where(y_pred_test != test_labels)[0]

# Yanlış sınıflandırmaları görselleştirip kaydetme
misclassified_path = r'C:\Users\burha\Desktop\VERI-SET\VERILER\CIKTILAR\HAM-YANLIS'
if not os.path.exists(misclassified_path):
    os.makedirs(misclassified_path)

for idx in misclassified_indices[:5]:  # İlk 5 yanlış sınıflamayı kaydedelim
    img = test_images[idx]  # Yanlış sınıflandırılmış görüntü
    plt.imshow(img, cmap='gray')
    plt.title(f"Gerçek: {test_labels[idx]}, Tahmin: {y_pred_test[idx]}")
    plt.axis('off')
    plt.show()

    # Kaydetme işlemi
    img.save(f'{misclassified_path}/misclassified_image_{idx}.jpg')
