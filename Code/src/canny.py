import cv2
import sys
import matplotlib.pyplot as plt

def detecter_regions_falsifiees(image):
    # Convertir l'image en nvg
    image_gris = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Algo de détection de Canny
    contours = cv2.Canny(image_gris, 100, 200)
    
    # On cherche les régions où les gradients varient de manière significative
    regions_falsifiees = []
    contours, _ = cv2.findContours(contours, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        # calcule l'aire du contour
        aire = cv2.contourArea(contour)
        
        # Si l'aire est supérieure à un certain seuil, on considère la région comme falsifiée
        if aire > 80:
            x, y, w, h = cv2.boundingRect(contour)
            regions_falsifiees.append((x, y, x + w, y + h))
    
    return regions_falsifiees

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py chemin_image")
        sys.exit(1)

    image = cv2.imread(sys.argv[1])
    
    regions_falsifiees = detecter_regions_falsifiees(image)

    # Afficher les régions falsifiées
    for region in regions_falsifiees:
        x1, y1, x2, y2 = region
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

    fig, axes = plt.subplots(1, 2)
    axes[0].imshow(cv2.cvtColor(cv2.imread(sys.argv[1]), cv2.COLOR_BGR2RGB))
    axes[0].set_title('Image Falsifiée')
    axes[0].axis('off')
    
    axes[1].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axes[1].set_title('Régions suspectes détectées')
    axes[1].axis('off')

    plt.show()
