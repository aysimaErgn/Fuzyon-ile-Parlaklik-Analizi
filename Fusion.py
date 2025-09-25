# -*- coding: utf-8 -*-
"""
Created on Sat May  3 22:52:13 2025

@author: aysim
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


#------PARLAKLIK ANALİZİ YAPARKEN KULLANILACAK FOKSİYON-------#
def brightness_map(gray_img):
   brightness_map = np.zeros_like(gray_img)
   brightness_map[gray_img < 70] = 1  
   brightness_map[(gray_img >= 70) & (gray_img < 180)] = 2 
   brightness_map[gray_img >= 180] = 3 
   return brightness_map

#-------------------GÖRÜNTÜ YOLLARI GİRİLİYOR-------------#
rgb_path = r"D:\DOSYALAR\BM4-BAHAR\computervision\FLIR_ADAS_1_3\video\RGB\FLIR_video_02092.jpg"
thermal_path = r"D:\DOSYALAR\BM4-BAHAR\computervision\FLIR_ADAS_1_3\video\thermal_16_bit\FLIR_video_02092.tiff"  

#-------------------GÖRÜNTÜYÜ OKUMA-----------------------#
rgb_img_raw = cv2.imread(rgb_path)
rgb_img_raw = cv2.cvtColor(rgb_img_raw, cv2.COLOR_BGR2RGB)
rgb_img_raw = cv2.resize(rgb_img_raw, (640, 512))

thermal_img_raw = cv2.imread(thermal_path, cv2.IMREAD_GRAYSCALE)
thermal_img_raw = cv2.resize(thermal_img_raw, (640, 512))


#-------------------NORMALİZASYON-------------------------#
if thermal_img_raw.max() > thermal_img_raw.min():
    thermal_img = (thermal_img_raw - thermal_img_raw.min()) / (thermal_img_raw.max() - thermal_img_raw.min())
else:
    thermal_img = thermal_img_raw / 255.0


rgb_img = rgb_img_raw / 255.0
rgb_gray = cv2.cvtColor((rgb_img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY) / 255.0

#------------------FÜZYON RESMİ OLUŞTURUYORUZ------------#
alpha = 0.6
fuzyon_gri = alpha * rgb_gray + (1 - alpha) * thermal_img
fuzyon_gri = (fuzyon_gri - fuzyon_gri.min()) / (fuzyon_gri.max() - fuzyon_gri.min())
fuzyon_renkli = cv2.applyColorMap((fuzyon_gri * 255).astype(np.uint8), cv2.COLORMAP_JET)
fuzyon_renkli = cv2.cvtColor(fuzyon_renkli, cv2.COLOR_BGR2RGB) / 255.0


#-----------------FÜZYON GÖRÜNTÜYÜ GÖRÜNTÜLEME-----------#
plt.figure(figsize=(16, 12))

plt.subplot(2, 2, 1)
plt.imshow(rgb_img)
plt.title("RGB Görüntü")
plt.axis("off")

plt.subplot(2, 2, 2)
plt.imshow(thermal_img, cmap="inferno")
plt.title("Termal Görüntü")
plt.axis("off")

plt.subplot(2, 2, 3)
plt.imshow(fuzyon_gri, cmap="gray")
plt.title("Füzyonlanmış (Gri)")
plt.axis("off")

plt.subplot(2, 2, 4)
plt.imshow(fuzyon_renkli)
plt.title("Füzyonlanmış (Renkli)")
plt.axis("off")

plt.tight_layout()
plt.savefig("fusion_results.png", dpi=300)
plt.show()



#---------------RENKLERİ BGR'DAN RGB'YE ÇEVİRME-----------#

fuzyon_renkli_uint8 = (fuzyon_renkli * 255).astype(np.uint8)
fuzyon_gri_uint8 = (fuzyon_gri * 255).astype(np.uint8)
'''
Bu de normalizasyon işleminin amacı aşağıda yapılacak olan renk değişikliklerine uygun hale getirmek
Diğer türlü hata veriyor

İlk rgb ve thermal görüntüleri direkt raw olarak okuduk ama füzyon görüntüleri normalize edilmiş halleri 
üzerinden yaptığımız için bunlarda bu şekilde bir değişiklik yapmamız gerekti.
'''

#---------------GRİ TONLAMALIYA ÇEVİRME-------------------#
rgb_griton = cv2.cvtColor(rgb_img_raw, cv2.COLOR_RGB2GRAY)
fuzyon_renkli_griton = cv2.cvtColor(fuzyon_renkli_uint8, cv2.COLOR_RGB2GRAY)
#thermal_griton = cv2.cvtColor(thermal_img_raw, cv2.COLOR_RGB2GRAY)
#fuzyon_gri_griton= cv2.cvtColor(fuzyon_gri_uint8, cv2.COLOR_RGB2GRAY)
'''
Yukarıdaki iki değişikliği yapmama sebebimiz bu resimler zaten gri tonlamalı olarak okundu
'''


#--------------PARLAKLIK HARİTALARININ ÇIKARILMASI---------#
rgb_parlaklik_haritasi = brightness_map(rgb_griton)
thermal_parlaklik_haritasi = brightness_map(thermal_img_raw)
fuzyon_gri_parlaklik_haritasi = brightness_map(fuzyon_gri_uint8)
fuzyon_renkli_parlaklik_haritasi = brightness_map(fuzyon_renkli_griton)


#--------- PARLAKLIK ANALİZİ GÖRSELLEŞTİRME-----------------#
colors = ['black', 'blue', 'green', 'red']
cmap = ListedColormap(colors)

plt.figure(figsize=(20, 15))

plt.subplot(4, 3, 1)
plt.imshow(rgb_img)
plt.title('RGB Görüntü')
plt.axis('off')

plt.subplot(4, 3, 2)
plt.imshow(thermal_img)
plt.title('Termal Görüntü')
plt.axis('off')

plt.subplot(4, 3, 3)
plt.imshow(fuzyon_renkli)
plt.title('Renkli Füzyon Görüntü')
plt.axis('off')

plt.subplot(4, 3, 4)
plt.imshow(rgb_griton, cmap='gray')
plt.title('RGB (Gri Tonlama)')
plt.axis('off')

plt.subplot(4, 3, 5)
plt.imshow(thermal_img_raw, cmap='gray')
plt.title('Termal (Gri Tonlama)')
plt.axis('off')

plt.subplot(4, 3, 6)
plt.imshow(fuzyon_gri_uint8, cmap='gray')
plt.title('Gri Füzyon Görüntü')
plt.axis('off')

plt.subplot(4, 3, 7)
plt.imshow(rgb_parlaklik_haritasi, cmap=cmap, vmin=0, vmax=3)
plt.title('RGB Parlaklık Haritası')
plt.axis('off')

plt.subplot(4, 3, 8)
plt.imshow(thermal_parlaklik_haritasi, cmap=cmap, vmin=0, vmax=3)
plt.title('Termal Parlaklık Haritası')
plt.axis('off')

plt.subplot(4, 3, 9)
plt.imshow(fuzyon_renkli_parlaklik_haritasi, cmap=cmap, vmin=0, vmax=3)
plt.title('Renkli Füzyon Parlaklık Haritası')
plt.axis('off')

plt.subplot(4, 3, 10)
plt.hist(rgb_gray.ravel(), 256, [0, 256], alpha=0.7, color='blue')
plt.axvline(x=70, color='green', linestyle='--')
plt.axvline(x=180, color='red', linestyle='--')
plt.title('RGB Parlaklık Histogramı')
plt.xlim([0, 256])

plt.subplot(4, 3, 11)
plt.hist(thermal_img_raw.ravel(), 256, [0, 256], alpha=0.7, color='orange')
plt.axvline(x=70, color='green', linestyle='--')
plt.axvline(x=180, color='red', linestyle='--')
plt.title('Termal Parlaklık Histogramı')
plt.xlim([0, 256])

plt.subplot(4, 3, 12)
plt.hist(fuzyon_renkli_griton.ravel(), 256, [0, 256], alpha=0.7, color='purple')
plt.axvline(x=70, color='green', linestyle='--')
plt.axvline(x=180, color='red', linestyle='--')
plt.title('Füzyon Parlaklık Histogramı')
plt.xlim([0, 256])

labels = ['Arkaplan', 'Düşük Parlaklık', 'Orta Parlaklık', 'Yüksek Parlaklık']
patches = [plt.Rectangle((0, 0), 1, 1, color=colors[i]) for i in range(len(colors))]
plt.figlegend(patches, labels, loc='lower center', ncol=4, frameon=False)

plt.tight_layout()
plt.savefig('fuzyon_analiz_sonucu.png', dpi=300)
plt.show()








