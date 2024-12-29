import os
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

# Resimlerin bulunduğu klasörün tam yolunu belirt
folder_path = r'C:\Users\burha\Desktop\Yeni klasör\HAM\OGRENME\NORMAL'

# Çıktı dosyalarının kaydedileceği klasör
output_folder_path = r'C:\Users\burha\Desktop\Yeni klasör\HAM\cıktı'

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

def compute_contrast(img):
    # Görüntüdeki kontrastı ölçmek için standart sapma kullanabiliriz
    return np.std(img)

def plot_contrast_comparison(original_avg_contrast, clahe_avg_contrast):
    # Ortalama kontrast farklarını çizme
    plt.figure(figsize=(8, 5))
    
    labels = ['Orijinal', 'CLAHE Uygulandı']
    avg_contrasts = [original_avg_contrast, clahe_avg_contrast]
    
    bars = plt.bar(labels, avg_contrasts, color=['blue', 'red'])
    
    # Barların üzerine ortalama kontrast değerlerini yazma
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.01, f'{yval:.2f}', ha='center', va='bottom', fontsize=12)
    
    plt.xlabel('Görsel Türü')
    plt.ylabel('Ortalama Kontrast Değeri')
    plt.title('Orijinal ve CLAHE Uygulanan Görsellerin Ortalama Kontrast Karşılaştırması')
    plt.tight_layout()
    plt.show()

# Kontrastları biriktirecek listeler
original_contrasts = []
clahe_contrasts = []

# Klasördeki tüm dosyaları al
for filename in os.listdir(folder_path):
    # Dosya uzantısını kontrol et
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):  # JPG, JPEG veya PNG dosyalarını al
        # Resmin tam yolunu oluştur
        img_path = os.path.join(folder_path, filename)

        # Resmi oku
        img = Image.open(img_path).convert('L')  # Gri tonlamalı olarak oku

        # CLAHE uygula
        clahe_img = apply_clahe(img)

        # Görüntülerdeki kontrastı hesapla
        original_contrast = compute_contrast(np.array(img))
        clahe_contrast = compute_contrast(np.array(clahe_img))

        # Sonuçları listeye ekle
        original_contrasts.append(original_contrast)
        clahe_contrasts.append(clahe_contrast)

        # Sonucu farklı klasöre kaydet
        output_path = os.path.join(output_folder_path, 'clahe_' + filename)
        clahe_img.save(output_path)
        print(f"CLAHE uygulandı ve kaydedildi: {output_path}")

# Ortalama kontrastları hesapla
original_avg_contrast = np.mean(original_contrasts)
clahe_avg_contrast = np.mean(clahe_contrasts)

# Sonuçları karşılaştırmalı olarak göster
plot_contrast_comparison(original_avg_contrast, clahe_avg_contrast)
