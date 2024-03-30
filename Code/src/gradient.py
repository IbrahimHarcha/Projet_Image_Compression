import cv2
import numpy as np
import sys

import matplotlib.pyplot as plt

def detecter_falsification(image, seuil):
    img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)

    # on calcule le gradient horizontal et vertical de l'image
    gradient_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)

    # magnitude du gradient
    magnitude = np.sqrt(gradient_x**2 + gradient_y**2)

    # on applique le seuil
    regions_suspectes = np.where(magnitude > seuil)

    # on affiche les régions suspectes sur l'image
    img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img_color[regions_suspectes] = [0, 0, 255]  # Rouge

    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2RGB))
    plt.title('Image Falsifiée')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB))
    plt.title('Régions suspectes détectées')
    plt.axis('off')

    plt.show()

image_path = sys.argv[1]
seuil = int(sys.argv[2])
detecter_falsification(image_path, seuil)
