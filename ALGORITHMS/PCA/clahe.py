from PIL import Image
import os
import cv2
import numpy as np

# Resimlerin bulunduğu klasörün tam yolunu belirt
folder_path = r'C:\Users\burha\Desktop\ESKİ\OCSS\KERNEL\TEST\OSCC'

# Çıktı dosyalarının kaydedileceği klasör
output_folder_path = r'C:\Users\burha\Desktop\ESKİ\OCSS\KERNEL-CLAHE\TEST\OSCC'

# Eğer çıktı klasörü mevcut değilse oluştur
if not os.path.exists(output_folder_path):
    os.makedirs(output_folder_path)

def apply_clahe(img, clip_limit=1.0, grid_size=8):
    # Görüntüyü numpy dizisine çevir
    img_np = np.array(img)

    # CLAHE uygulama
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid_size, grid_size))
    clahe_img = clahe.apply(img_np)

    return Image.fromarray(clahe_img)

# Klasördeki tüm dosyaları al
for filename in os.listdir(folder_path):
    # Dosya uzantısını kontrol et
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):  # JPG, JPEG veya PNG dosyalarını al
        # Resmin tam yolunu oluştur
        img_path = os.path.join(folder_path, filename)

        # Resmi oku
        img = Image.open(img_path).convert('L')  # Gri tonlamalı olarak oku

        # Resmin doğru bir şekilde açılıp açılmadığını kontrol et
        if img is not None:
            # CLAHE uygula
            clahe_img = apply_clahe(img)

            # Sonucu farklı klasöre kaydet
            output_path = os.path.join(output_folder_path, 'clahe_' + filename)
            clahe_img.save(output_path)
            print(f"CLAHE uygulandı: {output_path}")
        else:
            print(f"Resim açılamadı: {img_path}")
    else:
        print(f"Geçersiz dosya uzantısı: {filename}")
