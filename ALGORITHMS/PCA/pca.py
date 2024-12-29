import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 1. Resimlerin Yüklenmesi
# Resimlerin bulunduğu klasörün yolunu belirtin
image_folder = r'C:\Users\elfal\Desktop\btbs dersler\PROJELER\VERI-SET\VERILER\CLAHE\OGRENME\NORMAL'

# Tüm resim dosyalarını yükle ve vektörleştir
image_data = []
for filename in os.listdir(image_folder):
    if filename.endswith('.jpg'):
        img_path = os.path.join(image_folder, filename)
        img = Image.open(img_path)
        img = img.resize((100, 100))  # Boyutunu küçültmek için (100x100)
        img_array = np.array(img).flatten()  # Resmi vektöre dönüştür
        image_data.append(img_array)

# 2. Verileri NumPy Array'e çevir
image_data = np.array(image_data)

# 3. Verilerin Standardizasyonu
scaler = StandardScaler()
scaled_data = scaler.fit_transform(image_data)

# 4. PCA Uygulama
pca = PCA(n_components=2)
pca_data = pca.fit_transform(scaled_data)

# 5. PCA Sonuçlarının Görselleştirilmesi
plt.figure(figsize=(8, 6))
plt.scatter(pca_data[:, 0], pca_data[:, 1], c='blue', edgecolor='k', alpha=0.7)
plt.title('PCA KERNEL ile İndirgenmiş Resim Veri Kümesi')
plt.xlabel('Bileşen 1')
plt.ylabel('Bileşen 2')
plt.grid(True)
plt.tight_layout()
plt.show()

# PCA'nın açıkladığı varyans oranlarını yazdır
explained_variance = pca.explained_variance_ratio_
print(f"Varyansın {explained_variance[0]*100:.2f}% ve {explained_variance[1]*100:.2f}%'si ilk iki bileşenle açıklanıyor.")
