import sys
import cv2
import matplotlib.pyplot as plt

def extract_noise(image_path, kernel_size=21):
    # Charger l'image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Appliquer le filtre de médiane pour extraire le bruit
    noise = cv2.medianBlur(image, kernel_size)

    # Calculer le bruit en soustrayant l'image filtrée de l'image originale
    extracted_noise = cv2.absdiff(image, noise)

    # Améliorer la luminosité du bruit
    extracted_noise = cv2.convertScaleAbs(extracted_noise, alpha=1.5)

    return extracted_noise

assert not (len(sys.argv) != 2), 'Erreur, Usage : extract_noise.py imageIn'

# Exemple d'utilisation
image_path = sys.argv[1]
result = extract_noise(image_path)

# Afficher l'image originale et le bruit extrait avec Matplotlib
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB))
plt.title('Image Originale')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(result, cmap='gray')
plt.title('Bruit extrait')
plt.axis('off')

plt.show()

########## EVALUATION DE LA METHODE ##########

# import sys
# import cv2
# import matplotlib.pyplot as plt
# from sklearn.metrics import precision_score, recall_score, f1_score
# from sklearn.metrics import roc_curve, roc_auc_score, auc
# import numpy as np

# def extract_noise(image_path, kernel_size=21):
#     # Charger l'image
#     image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

#     # Appliquer le filtre de médiane pour extraire le bruit
#     noise = cv2.medianBlur(image, kernel_size)

#     # Calculer le bruit en soustrayant l'image filtrée de l'image originale
#     extracted_noise = cv2.absdiff(image, noise)

#     # Améliorer la luminosité du bruit
#     extracted_noise = cv2.convertScaleAbs(extracted_noise, alpha=1.5)

#     return extracted_noise

# assert len(sys.argv) == 3, 'Erreur, Usage : extract_noise.py imageIn groundTruthImage'

# # Exemple d'utilisation
# image_path = sys.argv[1]
# ground_truth_path = sys.argv[2]

# # Charger l'image de vérité terrain
# ground_truth = cv2.imread(ground_truth_path, cv2.IMREAD_GRAYSCALE)
# binarized_ground_truth = cv2.threshold(ground_truth, 127, 255, cv2.THRESH_BINARY)[1] / 255

# # Initialiser les listes pour stocker les valeurs de kernel_size et AUC
# kernel_sizes = []
# auc_scores = []

# # Faire varier kernel_size
# for kernel_size in np.arange(3, 30, 2):  # kernel_size varie de 3 à 29 avec un pas de 2
#     result = extract_noise(image_path, kernel_size)

#     # Binariser les images pour la comparaison
#     binarized_result = cv2.threshold(result, 127, 255, cv2.THRESH_BINARY)[1] / 255

#     # Calculer les points de la courbe ROC et l'AUC
#     fpr, tpr, thresholds = roc_curve(binarized_ground_truth.flatten(), binarized_result.flatten())
#     roc_auc = auc(fpr, tpr)

#     # Ajouter les valeurs de kernel_size et AUC aux listes
#     kernel_sizes.append(kernel_size)
#     auc_scores.append(roc_auc)

#     # Afficher les images
#     if kernel_size == 3 or kernel_size == 21 or kernel_size == 29:
#         plt.figure(figsize=(15, 5))
#         plt.subplot(1, 3, 1)
#         plt.imshow(cv2.imread(image_path, cv2.IMREAD_GRAYSCALE), cmap='gray')
#         plt.title('Image originale')
#         plt.subplot(1, 3, 2)
#         plt.imshow(ground_truth, cmap='gray')
#         plt.title('Image de vérité terrain')
#         plt.subplot(1, 3, 3)
#         plt.imshow(result, cmap='gray')
#         plt.title('Bruit extrait, avec kernel_size = '+str(kernel_size))
#         plt.show()

# # Afficher l'AUC en fonction de kernel_size
# plt.figure(figsize=(10, 5))
# plt.plot(kernel_sizes, auc_scores, marker='o')
# plt.xlabel('Kernel Size')
# plt.ylabel('AUC')
# plt.title('AUC en fonction de Kernel Size')
# plt.grid(True)

# plt.show()
