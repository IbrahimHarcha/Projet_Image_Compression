import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt

def jpeg_ghost_detection(image_path):
    """JPEG Ghost Detection"""

    # Charger l'image avec OpenCV
    original_image = cv2.imread(image_path)

    # Convertir l'image en niveaux de gris
    gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

    # Initialiser une liste pour stocker les résultats significatifs
    significant_results = []

    # Calculer le nombre d'images à afficher et ajuster les lignes et les colonnes en conséquence
    num_qualities = 11  # Nombre de qualités JPEG à afficher (50 à 100 avec un pas de 5)
    num_rows = (num_qualities + 4) // 5  # Diviser par 5 avec arrondi supérieur
    num_cols = 5

    # Initialiser la grille pour les sous-graphiques
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(20, 4 * num_rows))

    # Parcourir différentes qualités JPEG
    for i, quality in enumerate(range(50, 101, 5)):
        # Sauvegarder et recharger l'image avec compression JPEG
        _, buffer = cv2.imencode('.jpg', original_image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        recompressed_image = cv2.imdecode(buffer, 1)

        # Convertir l'image recompressée en niveaux de gris
        gray_recompressed_image = cv2.cvtColor(recompressed_image, cv2.COLOR_BGR2GRAY)

        # Soustraire l'image recompressée de l'image originale
        difference = cv2.absdiff(gray_image, gray_recompressed_image)

        # Stocker le résultat si la différence est significative
        if np.max(difference) > 10:  # Valeur arbitraire, à ajuster selon le cas
            significant_results.append((quality, difference))

        # Mettre en évidence les régions anormales en rouge
        anomaly_highlighted = original_image.copy()
        anomaly_highlighted[difference > 10] = [0, 0, 255]

        # Afficher l'image avec les régions anormales mises en évidence dans la grille de sous-graphiques
        ax = axs[i // num_cols, i % num_cols]
        ax.imshow(cv2.cvtColor(anomaly_highlighted, cv2.COLOR_BGR2RGB))
        ax.set_title(f"Qualité {quality}")
        ax.axis('off')

    # Masquer les sous-graphiques restants s'il y a moins de qualités que le nombre maximum de sous-graphiques
    for j in range(num_qualities, num_rows * num_cols):
        axs[j // num_cols, j % num_cols].axis('off')

    # Ajuster l'espacement entre les sous-graphiques
    plt.tight_layout()
    plt.show()


assert not (len(sys.argv) != 2), 'Erreur, Usage : jpeg_ghost_detection.py imageIn'

# Utilisation de la fonction pour détecter les ghosts JPEG
image_path = str(sys.argv[1])
jpeg_ghost_detection(image_path)
