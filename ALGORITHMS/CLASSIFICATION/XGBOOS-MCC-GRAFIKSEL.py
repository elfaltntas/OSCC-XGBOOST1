import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import glob
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, matthews_corrcoef
import xgboost as xgb
import seaborn as sns
import pandas as pd

# 1. Görüntü dosyalarını yükleme (Normal ve Hastalıklı)
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
        img = Image.open(file).resize((224, 224))  # Sadece boyutlandırma, gri tonlama yok
        images.append(img)
    except Exception as e:
        print(f"{file} yüklenemedi. Hata: {e}")

# Görüntülerin başarılı şekilde yüklendiğini kontrol et
if len(images) == 0:
    print("Hiçbir görüntü işlenemedi.")
    exit()

# 2. Görüntüleri numpy dizisine dönüştürme
image_arrays = np.array([np.array(img).flatten() for img in images])
print("Görüntü dizisi boyutları:", image_arrays.shape)

# 3. Veriyi ölçekleme
scaler = StandardScaler()
images_scaled = scaler.fit_transform(image_arrays)

# 4. PCA uygulama (Daha fazla bileşen kullanıyoruz)
pca = PCA(n_components=100)  # PCA ile daha fazla boyut indirgeme
images_pca = pca.fit_transform(images_scaled)
print("PCA sonrası veri boyutları:", images_pca.shape)

# 5. Etiketleri tanımlama (Normal: 0, Hastalıklı: 1)
labels = np.array([0 if 'NORMAL' in file else 1 for file in file_paths])  # Etiketleme

# 6. Veriyi eğitim ve test olarak ayırma
X_train, X_test, y_train, y_test = train_test_split(images_pca, labels, test_size=0.2, stratify=labels, random_state=42)

# 7. XGBoost Modeli (Hyperparameter Tuning)
# Model parametrelerini ayarlama
xgb_model = xgb.XGBClassifier(
    use_label_encoder=False, 
    eval_metric='logloss',
    learning_rate=0.05,  # Öğrenme oranını düşürüyoruz
    max_depth=6,  # Ağaç derinliğini sınırlandırıyoruz
    n_estimators=1000,  # Daha fazla ağaç
    subsample=0.8,  # Veri örnekleme oranı
    colsample_bytree=0.8  # Her ağacın için örnekleme oranı
)

xgb_model.fit(X_train, y_train)

# 8. Tahminler ve Performans Analizi
y_pred = xgb_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Doğruluk Skoru:", accuracy)
print("Sınıflandırma Raporu:\n", classification_report(y_test, y_pred))

# 8.1 Matthew Correlation Coefficient (MCC)
mcc_score = matthews_corrcoef(y_test, y_pred)
print("Matthew Correlation Coefficient (MCC):", mcc_score)

# 9. Doğruluk Oranı ve MCC'yi aynı grafikte gösterme
metrics = ['Doğruluk Oranı', 'MCC']
scores = [accuracy, mcc_score]

# Grafik
plt.figure(figsize=(8, 6))
bars = plt.bar(metrics, scores, color=['skyblue', 'lightgreen'])

# Başlık ve etiketler
plt.title("Model Performans Değerlendirmesi", fontsize=16)
plt.ylabel("Skor", fontsize=12)
plt.ylim([0, 1])  # MCC negatif olabileceğinden, y eksenini 0 ile 1 arasında ayarladık
plt.xlabel("Metrik", fontsize=12)

# Sütunların içine değerleri ekleyelim
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval/2, round(yval, 2), ha='center', va='center', fontsize=12, color='black')

# Göster
plt.show()

# 10. Sınıflandırma Raporu için Görsel
# Sınıflandırma raporunu almak
class_report = classification_report(y_test, y_pred, output_dict=True)
class_report_df = pd.DataFrame(class_report).transpose()

# Grafik için veri hazırlığı
# Precision, Recall, F1-Score ve Support'ı kullanacağız
metrics = ['Precision', 'Recall', 'F1-Score']
classes = ['0', '1']  # Burada sınıf adlarını '0' ve '1' olarak belirliyoruz

# Grafik için her bir metrik için sınıfların değerleri
precision_values = class_report_df.loc[classes, 'precision'].values
recall_values = class_report_df.loc[classes, 'recall'].values
f1_values = class_report_df.loc[classes, 'f1-score'].values

# Her metrik için uygun renkler
colors = ['skyblue', 'lightgreen', 'salmon']

# Grafik
fig, ax = plt.subplots(figsize=(10, 6))

# Her bir sınıf için Precision, Recall ve F1-Score için çubuklar çizme
bar_width = 0.2
index = np.arange(len(classes))

bar1 = ax.bar(index, precision_values, bar_width, label='Precision', color=colors[0])
bar2 = ax.bar(index + bar_width, recall_values, bar_width, label='Recall', color=colors[1])
bar3 = ax.bar(index + 2 * bar_width, f1_values, bar_width, label='F1-Score', color=colors[2])

# Başlık ve etiketler
ax.set_title("Sınıflandırma Raporu", fontsize=16)
ax.set_xlabel("Sınıflar", fontsize=12)
ax.set_ylabel("Skor", fontsize=12)
ax.set_xticks(index + bar_width)
ax.set_xticklabels(['Normal (0)', 'Hastalıklı (1)'])  # Etiketlerin doğru şekilde görünmesini sağlamak
ax.set_ylim([0, 1])

# Her bir çubuğun içine değerleri yazalım
for bar in [bar1, bar2, bar3]:
    for rect in bar:
        height = rect.get_height()
        ax.text(
            rect.get_x() + rect.get_width() / 2,
            height / 2,  # Y değeri, çubuğun ortasına gelecek şekilde ayarlıyoruz
            round(height, 2),
            ha='center', va='center', fontsize=12, color='black'
        )

# Legend ekleyelim
ax.legend()

# Görseli göster
plt.tight_layout()
plt.show()
