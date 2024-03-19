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
