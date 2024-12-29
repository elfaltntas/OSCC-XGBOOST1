import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import glob
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import KernelPCA
import os

# Görüntü dosyalarını belirtilen yoldan yükleme
file_paths = glob.glob(r'C:\Users\burha\Desktop\Yeni klasör\HAM\OGRENME\NORMAL\*.jpg')
images = [Image.open(file) for file in file_paths]
images = [img.resize((128, 128)) for img in images]  # Görüntü boyutunu küçültme

# Görüntüleri numpy dizisine dönüştürme
image_arrays = np.array([np.array(img).flatten() for img in images])

# Veriyi ölçekle
scaler = StandardScaler()
images_scaled = scaler.fit_transform(image_arrays)

# Orijinal PCA uygulayın
pca = PCA(n_components=50)  # Korunacak bileşen sayısı
images_pca = pca.fit_transform(images_scaled)

# Kernel PCA uygulayın (örneğin, RBF kernel ile)
kernel_pca = KernelPCA(kernel="rbf", n_components=50, gamma=1)  # Kernel PCA (RBF kernel)
images_kernel_pca = kernel_pca.fit_transform(images_scaled)

# Kaydetme klasörünü belirt
save_path = r'C:\Users\burha\Desktop\Yeni klasör\HAM\cıktı'
if not os.path.exists(save_path):
    os.makedirs(save_path)

# Orijinal PCA bileşenlerinin enerji (projeksiyon) katkısını hesapla
pca_energy_contribution = np.sum(np.square(pca.components_), axis=1)

# KernelPCA projeksiyonlarının enerji katkısını hesapla
# KernelPCA'nın alphas_ bir tür projeksiyon katsayısıdır.
kernel_pca_energy_contribution = np.sum(np.square(images_kernel_pca), axis=1)

# Grafik oluşturma
plt.figure(figsize=(10, 6))

# Orijinal PCA'nın enerji katkısını çiz
plt.plot(np.cumsum(pca_energy_contribution), label='Orijinal PCA Enerji Katkısı', color='b')

# KernelPCA'nın enerji katkısını çiz
plt.plot(np.cumsum(kernel_pca_energy_contribution), label='Kernel PCA Enerji Katkısı', color='r')

plt.title('Orijinal PCA ve Kernel PCA Enerji Katkısı Karşılaştırması')
plt.xlabel('Bileşen Indexi')
plt.ylabel('Açıklanan Enerji (Projeksiyon Katkısı)')
plt.legend()
plt.grid(True)
plt.show()
