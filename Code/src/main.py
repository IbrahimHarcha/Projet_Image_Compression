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
            quality_factor = simpledialog.askstring("Input", "Entrez le facteur qualité (ou appuyez sur \"OK\" pour utiliser la valeur par défaut) :", parent=root)
            if quality_factor is not None and quality_factor != "":
                command.append(quality_factor)
            else:
                command.append("75")
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

    # Bouton pour évaluer la méthode de détection
    evaluate_btn = CTkButton(frame, text="Évaluer la méthode de détection", command=evaluate_detection)
    evaluate_btn.grid(column=0, row=5, columnspan=2, pady=30, sticky="ew")

    # Label pour le CNN
    label_cnn = CTkLabel(frame, text="", bg_color="#f0f0f0", font=("Arial", 12), text_color="#000000")
    label_cnn.grid(column=0, row=7, columnspan=2, sticky="ew")

    def evaluate_cnn():
        global file_path
        if file_path:
            model = load_model('../cnn/fake_image_detector_model4.h5')
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


def evaluate_detection(): 
    y_true = [0, 0, 1, 1]
    y_scores = [0.1, 0.4, 0.35, 0.8]
    
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()





if __name__ == "__main__":
    main()