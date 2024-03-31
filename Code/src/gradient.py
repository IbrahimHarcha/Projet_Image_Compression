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

########## EVALUATION DE LA METHODE ##########

# import cv2
# import numpy as np
# import sys
# from sklearn.metrics import roc_curve, auc

# import matplotlib.pyplot as plt

# def detecter_falsification(image, seuil):
#     img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)

#     # on calcule le gradient horizontal et vertical de l'image
#     gradient_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
#     gradient_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)

#     # magnitude du gradient
#     magnitude = np.sqrt(gradient_x**2 + gradient_y**2)

#     # on applique le seuil
#     regions_suspectes = np.where(magnitude > seuil, 1, 0)

#     return regions_suspectes

# assert len(sys.argv) == 3, 'Erreur, Usage : detecter_falsification.py imageIn groundTruthImage'

# # Exemple d'utilisation
# image_path = sys.argv[1]
# ground_truth_path = sys.argv[2]

# # Charger l'image de vérité terrain
# ground_truth = cv2.imread(ground_truth_path, cv2.IMREAD_GRAYSCALE)
# binarized_ground_truth = cv2.threshold(ground_truth, 127, 255, cv2.THRESH_BINARY)[1] / 255

# # Faire varier le seuil
# plt.figure(figsize=(10, 5))
# for i, seuil in enumerate([50,100,150], start=1):  # seuil varie de 0 à 255 avec un pas de 5
#     result = detecter_falsification(image_path, seuil)

#     # Afficher l'image pour le seuil courant
#     plt.subplot(1, 3, i)
#     plt.imshow(result, cmap='gray')
#     plt.title('Seuil = %d' % seuil)

# plt.show()

# # Afficher les courbes ROC
# plt.figure(figsize=(10, 5))
# for seuil in [50,100,150]:
#     result = detecter_falsification(image_path, seuil)

#     # Calculer les points de la courbe ROC et l'AUC
#     fpr, tpr, thresholds = roc_curve(binarized_ground_truth.flatten(), result.flatten())
#     roc_auc = auc(fpr, tpr)

#     # Afficher la courbe ROC
#     plt.plot(fpr, tpr, label='ROC curve (area = %0.2f) pour le seuil = %d' % (roc_auc, seuil))

# plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver Operating Characteristic')
# plt.legend(loc="lower right")
# plt.show()