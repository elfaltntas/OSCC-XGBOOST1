from PIL import Image
import os

# Görsel dosyalarının bulunduğu dizin
input_directory = r"C:\Users\burha\Desktop\VERI-SET\val\OSCC"  # Görsellerin bulunduğu klasörün yolu
output_directory = r"C:\Users\burha\Desktop\VERI-SET\val\OSCC-1"  # Boyutlandırılmış görsellerin kaydedileceği klasör

# Çıktı klasörü yoksa oluştur
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Tüm jpg dosyalarını işle
for filename in os.listdir(input_directory):
    if filename.endswith(".jpg"):
        # Dosya yolunu oluştur
        img_path = os.path.join(input_directory, filename)
        
        # Görseli aç
        with Image.open(img_path) as img:
            # Görseli 224x224 piksele yeniden boyutlandır
            img_resized = img.resize((224, 224))
            
            # Çıktı dosya yolunu oluştur
            output_path = os.path.join(output_directory, filename)
            
            # Görseli kaydet
            img_resized.save(output_path)
            print(f"Yeniden boyutlandırıldı: {filename}")
