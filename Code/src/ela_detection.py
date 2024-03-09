import sys
import cv2
import matplotlib.pyplot as plt

def ela_detection_color(path, quality):
    """Error Level Analysis (Color)"""

    # Charger l'image avec OpenCV
    img = cv2.imread(path)

    # Sauvegarder et recharger l'image avec compression JPEG
    _, buffer = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    resaved_img = cv2.imdecode(buffer, 1)

    # Calculer l'image ELA en couleur
    ela_img = cv2.absdiff(img, resaved_img)

    # Améliorer la luminosité de l'image ELA
    scale = 255.0 / ela_img.max()
    ela_img = cv2.convertScaleAbs(ela_img, alpha=scale)

    return ela_img

assert not (len(sys.argv) != 2), 'Erreur, Usage : ela_detection.py imageIn'

# Example usage
image_path = str(sys.argv[1])
quality = 75
result = ela_detection_color(image_path, quality)


# Afficher l'image originale et l'ELA avec Matplotlib
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB))
plt.title('Image Originale')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
plt.title('ELA Image')
plt.axis('off')

plt.show()

