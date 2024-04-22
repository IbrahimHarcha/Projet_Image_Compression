import sys
import cv2
from PIL import Image, ImageChops, ImageEnhance
import tempfile

import matplotlib.pyplot as plt

def ela_detection_color(path, quality=90):
    temp_filename = tempfile.mktemp(suffix='.jpg')
    
    image = Image.open(path).convert('RGB')
    image.save(temp_filename, 'JPEG', quality=quality)
    temp_image = Image.open(temp_filename)
    
    ela_image = ImageChops.difference(image, temp_image)
    
    extrema = ela_image.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    if max_diff == 0:
        max_diff = 1
    scale = 255.0 / max_diff
    
    ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)
    
    return ela_image


assert not (len(sys.argv) != 3), 'Erreur, Usage : ela_detection.py imageIn quality'

# Example usage
image_path = str(sys.argv[1])
quality = int(sys.argv[2])
result = ela_detection_color(image_path, quality)

# result.save("ELA_out.jpg")


# Afficher l'image originale et l'ELA avec Matplotlib
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB))
plt.title('Image Originale')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(result)
plt.title('ELA Image')
plt.axis('off')

plt.show()

