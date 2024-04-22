import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt

def DTC_filter(image_path):
    # Charger l'image en couleur
    img = cv2.imread(image_path)

    # Convertir l'image en espace de couleur YCrCb
    img_ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

    # Initialiser une image pour les anomalies
    anomalies_img = np.zeros_like(img_ycrcb[:, :, 0], dtype=np.float32)

    # Diviser l'image en blocs de 8x8 pixels
    h, w, _ = img.shape
    h_blocks = h // 8
    w_blocks = w // 8

    for i in range(h_blocks):
        for j in range(w_blocks):
            # Extraire le bloc 8x8
            block = img_ycrcb[i*8:(i+1)*8, j*8:(j+1)*8, 0]

            # Appliquer la transformée DCT au bloc
            dct_result = cv2.dct(np.float32(block))

            # Seulement les hautes fréquences (ignorer les basses fréquences)
            dct_high_freq = dct_result[4:, 4:]

            # Ajuster la taille pour correspondre à anomalies_img
            magnitude_spectrum = cv2.resize(np.abs(dct_high_freq), (8, 8))

            # Mettre en évidence les anomalies
            anomalies_img[i*8:(i+1)*8, j*8:(j+1)*8] += magnitude_spectrum

    # Trouver les coefficients DCT élevés en comparant avec un seuil
    anomalies = anomalies_img > 0.6

    # Mettre en évidence les anomalies sur l'image originale
    img_highlighted = img.copy()
    img_highlighted[anomalies] = [0, 0, 255]  # Mettre en rouge les zones anormales
    
    return img_highlighted


assert not (len(sys.argv) != 2), 'Erreur, Usage : dct_filter.py imageIn'

# Exemple d'utilisation
image_path = sys.argv[1]
result = DTC_filter(image_path)

# cv2.imwrite("DCT_out.jpg", result)

# Afficher l'image originale et l'image avec les anomalies mises en évidence
plt.subplot(121), plt.imshow(cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB))
plt.title('Image Originale'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
plt.title('Anomalies DCT Mises en Évidence'), plt.xticks([]), plt.yticks([])

plt.show()