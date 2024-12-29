import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import glob
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os

# Görüntü dosyalarını belirtilen yoldan yükleme
file_paths = glob.glob(r'C:\Users\burha\Desktop\ESKİ\OCSS\CLAHE\TEST\OSCC\*.jpg')
images = [Image.open(file) for file in file_paths]
images = [img.resize((128, 128)) for img in images]  # Görüntü boyutunu küçültme

# Görüntüleri numpy dizisine dönüştürme
image_arrays = np.array([np.array(img).flatten() for img in images])

# Veriyi ölçekle
scaler = StandardScaler()
images_scaled = scaler.fit_transform(image_arrays)

# PCA uygulayın
pca = PCA(n_components=50)  # Korunacak bileşen sayısı
images_pca = pca.fit_transform(images_scaled)

# PCA ile yeniden yapılandırılmış görüntüleri elde et
images_reconstructed = pca.inverse_transform(images_pca)
images_reconstructed = scaler.inverse_transform(images_reconstructed)  # Ölçeklemeyi geri al

# Kaydetme klasörünü belirt
save_path = r'C:\Users\burha\Desktop\ESKİ\OCSS\CLAHE-KERNEL\TEST\OSCC'
if not os.path.exists(save_path):
    os.makedirs(save_path)

# Yeniden yapılandırılmış görüntüleri kaydet
for i, img_array in enumerate(images_reconstructed):
    img = Image.fromarray(np.uint8(img_array.reshape(128, 128)))  # Boyut ve kanalı düzenleyin
    img.save(f'{save_path}\\reconstructed_image_{i}.jpg')

# İsteğe bağlı: Bir yeniden yapılandırılmış görüntüyü göster
plt.imshow(img, cmap='gray')
plt.title('Yeniden Yapılandırılmış Görüntü')
plt.axis('off')
plt.show()
