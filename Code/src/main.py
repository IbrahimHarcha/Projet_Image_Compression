from tkinter import ttk, messagebox
import subprocess 
from tkinter import simpledialog
import os
from PIL import Image, ImageChops, ImageEnhance
import tempfile
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from customtkinter import *
import cv2


################## FONCTION POUR EVALUATION ##################
def gradient(image, seuil):
    img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)

    # on calcule le gradient horizontal et vertical de l'image
    gradient_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)

    # magnitude du gradient
    magnitude = np.sqrt(gradient_x**2 + gradient_y**2)

    # on applique le seuil
    regions_suspectes = np.where(magnitude > seuil, 1, 0)

    return regions_suspectes # Renvoie une image binaire !


def extract_noise_eval(image_path, kernel_size=21):
    # Charger l'image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # Appliquer le filtre de médiane pour extraire le bruit
    noise = cv2.medianBlur(image, kernel_size)
    # Calculer le bruit en soustrayant l'image filtrée de l'image originale
    extracted_noise = cv2.absdiff(image, noise)
    # Améliorer la luminosité du bruit
    extracted_noise = cv2.convertScaleAbs(extracted_noise, alpha=1.5)
    return extracted_noise

def jpeg_ghost_detection(image_path, ground_truth_path):
    """JPEG Ghost Detection"""

    # Charger l'image avec OpenCV
    original_image = cv2.imread(image_path)

    # Charger l'image de vérité terrain
    ground_truth = cv2.imread(ground_truth_path, cv2.IMREAD_GRAYSCALE)
    binarized_ground_truth = cv2.threshold(ground_truth, 127, 255, cv2.THRESH_BINARY)[1] / 255

    # Initialiser une liste pour stocker les AUC pour différentes qualités JPEG
    auc_scores = []

    # Nombre de qualités JPEG à évaluer (50 à 100 avec un pas de 5)
    quality_factors = range(50, 101, 5)

    # Parcourir différentes qualités JPEG
    for quality in quality_factors:
        # Sauvegarder et recharger l'image avec compression JPEG
        _, buffer = cv2.imencode('.jpg', original_image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        recompressed_image = cv2.imdecode(buffer, 1)

        # Convertir l'image recompressée en niveaux de gris
        gray_recompressed_image = cv2.cvtColor(recompressed_image, cv2.COLOR_BGR2GRAY)

        # Binariser l'image recompressée
        binarized_recompressed_image = cv2.threshold(gray_recompressed_image, 127, 255, cv2.THRESH_BINARY)[1] / 255

        # Calculer les points de la courbe ROC et l'AUC
        fpr, tpr, _ = roc_curve(binarized_ground_truth.flatten(), binarized_recompressed_image.flatten())
        roc_auc = auc(fpr, tpr)

        # Ajouter l'AUC à la liste
        auc_scores.append(roc_auc)

    # Tracer la courbe AUC en fonction du facteur qualité JPEG
    plt.figure(figsize=(10, 5))
    plt.plot(quality_factors, auc_scores, marker='o')
    plt.xlabel('Facteur Qualité JPEG')
    plt.ylabel('AUC')
    plt.title('AUC en fonction du Facteur Qualité JPEG')
    plt.grid(True)
    plt.show()

############################################################################################################

# IA resources

IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CANAL = 3

def ELA_extraction(path):
    temp_filename = tempfile.mktemp(suffix='.jpg')
    
    image = Image.open(path).convert('RGB')
    image.save(temp_filename, 'JPEG', quality=90)
    temp_image = Image.open(temp_filename)
    
    ela_image = ImageChops.difference(image, temp_image)
    
    extrema = ela_image.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    if max_diff == 0:
        max_diff = 1
    scale = 255.0 / max_diff
    
    ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)
    
    return ela_image

def prepare_image(image_path):
    return np.array(ELA_extraction(image_path).resize((IMG_WIDTH, IMG_HEIGHT))).flatten() / 255.0 # Normalisation

model = load_model('../cnn/fake_image_detector_model4.h5')


# Méthodes de detection

def run_detection(script_name, image_path, root):
    try:
        # Construire la commande de base
        command = ["python", script_name, image_path]

        if script_name == "canny.py" or script_name == "canny2.py":
            seuil = simpledialog.askstring("Input", "Entrez le seuil de détection (ou appuyez sur \"OK\" pour utiliser le seuil par défaut de 80) :", parent=root)
            if seuil is not None and seuil != "":
                command.append(seuil)
            else:
                command.append("80")
        elif script_name == "facteurCouleur.py":
            # Demander le seuil pour le script facteurCouleur.py
            seuil = simpledialog.askstring("Input", "Entrez le seuil de détection :", parent=root)
            if seuil is not None:
                command.append(seuil)
            else:
                messagebox.showinfo("Annulation", "Opération annulée par l'utilisateur.")
                return
        elif script_name == "double_quantificazion_detect.py":
            # Demander les paramètres pour le script double_quantificazion_detect.py
            quality1 = simpledialog.askstring("Input", "Entrez la qualité de compression 1 (ou appuyez sur \"OK\" pour utiliser la valeur par défaut de 90) :", parent=root)
            if quality1 is not None and quality1 != "":
                command.append(quality1)
            else:
                command.append("90")

            quality2 = simpledialog.askstring("Input", "Entrez la qualité de compression 2 (ou appuyez sur \"OK\" pour utiliser la valeur par défaut de 50) :", parent=root)
            if quality2 is not None and quality2 != "":
                command.append(quality2)
            else:
                command.append("50")

            threshold = simpledialog.askstring("Input", "Entrez le seuil de binarisation (ou appuyez sur \"OK\" pour utiliser la valeur par défaut de 20) :", parent=root)
            if threshold is not None and threshold != "":
                command.append(threshold)
            else:
                command.append("20")
        elif script_name == "ela_detection.py":
            # Demander le facteur qualité pour le script ela_detection.py
            quality_factor = simpledialog.askstring("Input", "Entrez le facteur qualité (ou appuyez sur \"OK\" pour utiliser la valeur par défaut (90)) :", parent=root)
            if quality_factor is not None and quality_factor != "":
                command.append(quality_factor)
            else:
                command.append("90")
        elif script_name == "gradient.py":
            # Demander le seuil pour le script gradient.py
            seuil = simpledialog.askstring("Input", "Entrez le seuil de détection (ou appuyez sur \"OK\" pour utiliser le seuil par défaut de 150) :", parent=root)
            if seuil is not None and seuil != "":
                command.append(seuil)
            else:
                command.append("200")

        # on execute le script en tant que processus externe
        result = subprocess.run(command, check=True, capture_output=True, text=True)

        # # tester si le script s'est exécuté avec succès
        # messagebox.showinfo("Succès", "Détection terminée avec succès.")
    except subprocess.CalledProcessError as e:
        messagebox.showerror("Erreur", f"Une erreur est survenue lors de l'exécution du script : {e}")

    # affiche des infos sur la dernière exécution
    image_name = os.path.basename(image_path)
    if len(command) > 2:
        info = f"Précédente exécution : {script_name}  {image_name}  {' '.join(command[3:])}"
    else:
        info = f"Précédente exécution : {script_name}  {image_name}"

    return info


def evaluate_detection(script_name, image_path,root): 
    # Demander à l'utilisateur de sélectionner l'image de vérité terrain
    ground_truth_path = filedialog.askopenfilename(title="Sélectionner l'image de vérité terrain", filetypes=(("Fichiers image", "*.png;*.jpg;*.jpeg"), ("Tous les fichiers", "*.*")))
    if script_name == "extract_noise.py":
        if ground_truth_path:
            # Charger l'image de vérité terrain
            ground_truth = cv2.imread(ground_truth_path, cv2.IMREAD_GRAYSCALE)
            binarized_ground_truth = cv2.threshold(ground_truth, 127, 255, cv2.THRESH_BINARY)[1] / 255

            # Initialiser les listes pour stocker les valeurs de kernel_size et AUC
            kernel_sizes = []
            auc_scores = []

            # Demander à l'utilisateur d'entrer les valeurs minimale et maximale de la taille du noyau
            min_kernel_size = simpledialog.askinteger("Taille du noyau", "Entrez la taille minimale du noyau (entier impair) :", parent=root)
            max_kernel_size = simpledialog.askinteger("Taille du noyau", "Entrez la taille maximale du noyau (entier impair) :", parent=root)

            # Vérifier si les valeurs saisies sont valides
            if min_kernel_size is None or max_kernel_size is None:
                messagebox.showerror("Erreur", "Vous devez entrer des valeurs valides pour la taille du noyau.")
            elif min_kernel_size % 2 == 0 or max_kernel_size % 2 == 0:
                messagebox.showerror("Erreur", "La taille du noyau doit être un entier impair.")
            else:
            # Faire varier kernel_size
                for kernel_size in range(min_kernel_size, max_kernel_size + 1, 2): # kernel_size varie de min_kernel_size à max_kernel_size avec un pas de 2
                    result = extract_noise_eval(image_path, kernel_size)

                    # Binariser les images pour la comparaison
                    binarized_result = cv2.threshold(result, 127, 255, cv2.THRESH_BINARY)[1] / 255

                    # Calculer les points de la courbe ROC et l'AUC
                    fpr, tpr, thresholds = roc_curve(binarized_ground_truth.flatten(), binarized_result.flatten())
                    roc_auc = auc(fpr, tpr)

                    # Ajouter les valeurs de kernel_size et AUC aux listes
                    kernel_sizes.append(kernel_size)
                    auc_scores.append(roc_auc)

            # Afficher l'AUC en fonction de kernel_size
            plt.figure(figsize=(10, 5))
            plt.plot(kernel_sizes, auc_scores, marker='o')
            plt.xlabel('Kernel Size')
            plt.ylabel('AUC')
            plt.title('AUC en fonction de Kernel Size')
            plt.grid(True)
            plt.show()
        else:
            messagebox.showerror("Erreur", "Veuillez sélectionner une image de vérité terrain.")

    elif script_name == "gradient.py":
        if ground_truth_path:
            # Charger l'image de vérité terrain
            ground_truth = cv2.imread(ground_truth_path, cv2.IMREAD_GRAYSCALE)
            binarized_ground_truth = cv2.threshold(ground_truth, 127, 255, cv2.THRESH_BINARY)[1] / 255

            # Demander à l'utilisateur s'il veut afficher la courbe ROC ou la courbe AUC
            choice = simpledialog.askstring("Choix", "Voulez-vous afficher la courbe ROC ou la courbe AUC ? (ROC/AUC)")

            if choice == "ROC":
                # Demander à l'utilisateur d'entrer les valeurs minimale et maximale du seuil
                min_threshold = simpledialog.askinteger("Seuil", "Entrez la valeur minimale du seuil :", parent=root)
                max_threshold = simpledialog.askinteger("Seuil", "Entrez la valeur maximale du seuil :", parent=root)

                # Vérifier si les valeurs saisies sont valides
                if min_threshold is None or max_threshold is None:
                    messagebox.showerror("Erreur", "Vous devez entrer des valeurs valides pour le seuil.")
                else:
                    # Initialiser les listes pour stocker les taux de faux positifs (fpr) et les taux de vrais positifs (tpr)
                    fpr_list = []
                    tpr_list = []

                    # Faire varier le seuil
                    for threshold in range(min_threshold, max_threshold + 1):
                        result = gradient(image_path, threshold)  # Appel de la fonction gradient avec l'image et le seuil

                        # # Comparer les résultats avec l'image de vérité terrain pour calculer TP, FP, TN et FN
                        TP = np.sum((result == 1) & (binarized_ground_truth == 1))
                        FP = np.sum((result == 1) & (binarized_ground_truth == 0))
                        TN = np.sum((result == 0) & (binarized_ground_truth == 0))
                        FN = np.sum((result == 0) & (binarized_ground_truth == 1))

                        # Calculer TPR (True Positive Rate) et FPR (False Positive Rate)
                        TPR = TP / (TP + FN)
                        FPR = FP / (FP + TN)

                        # Ajouter les valeurs de FPR et TPR aux listes
                        fpr_list.append(FPR)
                        tpr_list.append(TPR)

                    # la courbe ROC
                    plt.figure(figsize=(10, 5))
                    # for i in range(len(fpr_list)):
                    #     plt.plot(fpr_list[i], tpr_list[i], color='blue', marker='o')
                    # la diagonale
                    plt.plot([0, 1], [0, 1], color='red', linestyle='--', label='Diagonale')
                    plt.plot(fpr_list, tpr_list, color='blue')
                    plt.xlabel('Taux de faux positifs (FPR)')
                    plt.ylabel('Taux de vrais positifs (TPR)')
                    plt.title('Courbe ROC en fonction du seuil')
                    plt.legend()
                    plt.xlim(0, 1)
                    plt.ylim(0, 1)
                    plt.grid(True)
                    plt.show()

            elif choice == "AUC":
                # Demander à l'utilisateur d'entrer les valeurs minimale et maximale du seuil
                min_threshold = simpledialog.askinteger("Seuil", "Entrez la valeur minimale du seuil :", parent=root)
                max_threshold = simpledialog.askinteger("Seuil", "Entrez la valeur maximale du seuil :", parent=root)

                # Vérifier si les valeurs saisies sont valides
                if min_threshold is None or max_threshold is None:
                    messagebox.showerror("Erreur", "Vous devez entrer des valeurs valides pour le seuil.")
                else:
                    # Initialiser les listes pour stocker les valeurs de seuil et AUC
                    thresholds = []
                    auc_scores = []

                    # Faire varier le seuil
                    for threshold in range(min_threshold, max_threshold + 1):
                        result = gradient(image_path, threshold)  # Appel de la fonction gradient avec l'image et le seuil

                        # Calculer les points de la courbe ROC et l'AUC
                        fpr, tpr, _ = roc_curve(binarized_ground_truth.flatten(), result.flatten())
                        roc_auc = auc(fpr, tpr)

                        # Ajouter les valeurs de seuil et AUC aux listes
                        thresholds.append(threshold)
                        auc_scores.append(roc_auc)

                    # Tracer la courbe AUC en fonction du seuil
                    plt.figure(figsize=(10, 5))
                    plt.plot(thresholds, auc_scores, marker='o')
                    plt.xlabel('Seuil')
                    plt.ylabel('AUC')
                    plt.title('AUC en fonction du seuil')
                    plt.grid(True)
                    plt.show()
        else:
            messagebox.showerror("Erreur", "Veuillez sélectionner une image de vérité terrain.")

    elif script_name == "jpeg_ghost.py":
        if ground_truth_path:
            jpeg_ghost_detection(image_path,ground_truth_path)


def main():
    root = CTk()
    root.title("Détection de Falsification d'Images")

    window_width = 600
    window_height = 500

    position_top = int(root.winfo_screenheight() / 2 - window_height / 2)
    position_right = int(root.winfo_screenwidth() / 2 - window_width / 2)

    root.geometry(f"{window_width}x{window_height}+{position_right}+{position_top}")

    root.configure(bg="#f0f0f0")
    style = ttk.Style(root)
    style.theme_use("clam")

    style.configure("TCombobox", font=("Arial", 12), background="#e0e0e0")
    style.map("TCombobox", fieldbackground=[("readonly", "#e0e0e0")])

    # Style pour les boutons
    # btn_style = ttk.Style()
    # btn_style.configure("TButton", font=("Arial", 12, "bold"), background="#a0a0a0")

    # Cadre pour les widgets
    frame = ttk.Frame(root, padding="20", relief="ridge", style="TFrame")
    frame.pack(expand=True, fill="both")

    # Configure grid
    frame.columnconfigure(0, weight=1)
    frame.columnconfigure(1, weight=1)
    frame.rowconfigure(0, weight=1)
    frame.rowconfigure(1, weight=1)

    # Sélection de script
    global script_name
    script_name = ""
    def on_combobox_change(value):
        global script_name
        script_name = value

    scripts = ["canny.py", "canny2.py", "dct_filter.py", "double_quantificazion_detect.py",
               "ela_detection.py", "extract_noise.py", "facteurCouleur.py",
               "gradient.py", "jpeg_ghost.py"]
    script_select_label = CTkLabel(frame, text="Choisissez un script :", bg_color="#f0f0f0", font=("Arial", 12), text_color="#000000")
    script_select_label.grid(column=0, row=0, sticky="ew")
    script_select = CTkComboBox(frame, command=on_combobox_change, values=scripts, state="readonly")
    script_select.grid(column=1, row=0, pady=5, padx=5, sticky="ew")
    # script_select.current(0)

    # Create a label to display the script info
    script_info_label = CTkLabel(frame, text="", bg_color="#f0f0f0", font=("Arial", 12), text_color="#000000")
    script_info_label.grid(column=0, row=4, columnspan=2, sticky="ew")

    # enregistrement du file_path
    global file_path
    file_path = ""

    # Sélection de l'image
    def choose_image():
        global file_path
        file_path = filedialog.askopenfilename()
        label_file_path.configure(text=file_path)
        
    def run():
        global script_name
        global file_path
        if file_path and script_name:
            info = run_detection(script_name, file_path, root)  # Modifié pour inclure root
            script_info_label.configure(text=info)  # maj du label avec les infos de la dernière exécution

    image_btn = CTkButton(frame, text="Choisir une image", command=choose_image)
    image_btn.grid(column=0, row=1, columnspan=2, pady=10, sticky="ew")

    # label file path
    label_file_path = CTkLabel(frame, text="", bg_color="#f0f0f0", font=("Arial", 12), text_color="#000000")
    label_file_path.grid(column=0, row=2, columnspan=2, sticky="ew")

    # Bouton pour évaluer la méthode de détection
    evaluate_btn = CTkButton(frame, text="Run", command=run)
    evaluate_btn.grid(column=0, row=3, columnspan=2, pady=30, sticky="ew")

    def evaluate_wrapper():
        global script_name
        global file_path
        if file_path and script_name:
            evaluate_detection(script_name, file_path,root)

    # Bouton pour évaluer la méthode de détection
    evaluate_btn = CTkButton(frame, text="Évaluer la méthode de détection", command=evaluate_wrapper)
    evaluate_btn.grid(column=0, row=5, columnspan=2, pady=30, sticky="ew")

    # Label pour le CNN
    label_cnn = CTkLabel(frame, text="", bg_color="#f0f0f0", font=("Arial", 12), text_color="#000000")
    label_cnn.grid(column=0, row=7, columnspan=2, sticky="ew")

    def evaluate_cnn():
        global file_path
        if file_path:
            preprocessed_image = prepare_image(file_path)
            preprocessed_image = preprocessed_image.reshape(-1, IMG_WIDTH, IMG_HEIGHT, IMG_CANAL)
            prediction = model.predict(preprocessed_image)
            class_labels = ['Réel', 'Falsifié']
            predicted_class_index = np.argmax(prediction)
            predicted_class_label = class_labels[predicted_class_index]
            confidence = prediction[0][predicted_class_index]
            label_cnn.configure(text=f"Image {predicted_class_label} avec une confiance de {confidence * 100.0:.2f} %")

    # Bouton pour le model CNN
    cnn_btn = CTkButton(frame, text="Avis de l'IA", command=evaluate_cnn)
    cnn_btn.grid(column=0, row=6, columnspan=2, pady=30, sticky="ew")

    root.mainloop()

if __name__ == "__main__":
    main()
