import sys
import cv2

import matplotlib.pyplot as plt

def detect_double_quantization(image_path, quality1, quality2, threshold):
    # Charger l'image originale
    original_image = cv2.imread(image_path)

    # Enregistrer l'image avec la première quantification
    _, buffer1 = cv2.imencode('.jpg', original_image, [int(cv2.IMWRITE_JPEG_QUALITY), quality1])
    compressed_image1 = cv2.imdecode(buffer1, 1)

    # Enregistrer l'image avec la deuxième quantification
    _, buffer2 = cv2.imencode('.jpg', original_image, [int(cv2.IMWRITE_JPEG_QUALITY), quality2])
    compressed_image2 = cv2.imdecode(buffer2, 1)

    # Calculer la différence entre les deux images compressées
    diff_image = cv2.absdiff(compressed_image1, compressed_image2)

    # Convertir l'image en niveaux de gris
    diff_gray = cv2.cvtColor(diff_image, cv2.COLOR_BGR2GRAY)

    # Binariser l'image différence pour mettre en évidence les incohérences
    _, binary_diff = cv2.threshold(diff_gray, threshold, 255, cv2.THRESH_BINARY)

    # Afficher les images
    plt.subplot(131)
    plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    plt.title('Image Originale')
    plt.axis('off')

    plt.subplot(132)
    plt.imshow(cv2.cvtColor(compressed_image1, cv2.COLOR_BGR2RGB))
    plt.title(f'Image Compressée (Qualité {quality1})')
    plt.axis('off')

    plt.subplot(133)
    plt.imshow(binary_diff, cmap='gray')
    plt.title('Carte des Incohérences')
    plt.axis('off')

    plt.show()

assert len(sys.argv) == 5, 'Erreur, Usage : double_quantificazion_detect.py imageIn quality1 quality2 threshold'

# Exemple d'utilisation
image_path = sys.argv[1]
quality1 = int(sys.argv[2])  # Qualité de compression 1
quality2 = int(sys.argv[3])  # Qualité de compression 2
threshold = int(sys.argv[4])  # Seuil de binarisation

detect_double_quantization(image_path, quality1, quality2, threshold)
