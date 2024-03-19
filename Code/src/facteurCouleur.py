import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys

def detecter_falsifications_facteur_couleur(image_path, seuil):
    image = cv2.imread(image_path)

    # on convertit l'image en espace de couleur HSV
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # pour séparer les canaux de couleur
    h, s, v = cv2.split(image_hsv)

    # calcule les gradients entre les canaux de couleur
    gradient_h = cv2.Sobel(h, cv2.CV_64F, 1, 0, ksize=3)
    gradient_s = cv2.Sobel(s, cv2.CV_64F, 1, 0, ksize=3)
    gradient_v = cv2.Sobel(v, cv2.CV_64F, 1, 0, ksize=3)

    # Calcule la magnitude des gradients
    magnitude = np.sqrt(gradient_h**2 + gradient_s**2 + gradient_v**2)

    # on applique un seuil 
    regions_suspectes = np.where(magnitude > seuil)

    # on affiche les régions suspectes sur l'image
    img_color = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_color[regions_suspectes] = [255, 0, 0]  # Rouge

    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB))
    plt.title('Image Originale')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(img_color)
    plt.title('Régions suspectes détectées')
    plt.axis('off')

    plt.show()

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py chemin_image seuil_detection")
        sys.exit(1)

    image_path = sys.argv[1]
    seuil = int(sys.argv[2])
    detecter_falsifications_facteur_couleur(image_path, seuil)