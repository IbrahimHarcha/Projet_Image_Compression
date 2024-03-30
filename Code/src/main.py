import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import subprocess 
from tkinter import simpledialog
import os


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
    root = tk.Tk()
    root.title("Détection de Falsification d'Images")

    window_width = 600
    window_height = 300

    position_top = int(root.winfo_screenheight() / 2 - window_height / 2)
    position_right = int(root.winfo_screenwidth() / 2 - window_width / 2)

    root.geometry(f"{window_width}x{window_height}+{position_right}+{position_top}")

    root.configure(bg="#f0f0f0")
    style = ttk.Style(root)
    style.theme_use("clam")

    style.configure("TCombobox", font=("Arial", 12), background="#e0e0e0")
    style.map("TCombobox", fieldbackground=[("readonly", "#e0e0e0")])

    # Style pour les boutons
    btn_style = ttk.Style()
    btn_style.configure("TButton", font=("Arial", 12, "bold"), background="#a0a0a0")

    # Cadre pour les widgets
    frame = ttk.Frame(root, padding="20", relief="ridge", style="TFrame")
    frame.pack(expand=True, fill="both")

    # Configure grid
    frame.columnconfigure(0, weight=1)
    frame.columnconfigure(1, weight=1)
    frame.rowconfigure(0, weight=1)
    frame.rowconfigure(1, weight=1)

    # Sélection de script
    script_name = tk.StringVar()
    scripts = ["canny.py", "canny2.py", "dct_filter.py", "double_quantificazion_detect.py",
               "ela_detection.py", "extract_noise.py", "facteurCouleur.py",
               "gradient.py", "jpeg_ghost.py"]
    script_select_label = ttk.Label(frame, text="Choisissez un script :", background="#f0f0f0", font=("Arial", 12))
    script_select_label.grid(column=0, row=0, sticky="ew")
    script_select = ttk.Combobox(frame, textvariable=script_name, values=scripts, state="readonly")
    script_select.grid(column=1, row=0, pady=5, padx=5, sticky="ew")
    script_select.current(0)

    # Create a label to display the script info
    script_info_label = ttk.Label(frame, text="", background="#f0f0f0", font=("Arial", 12))
    script_info_label.grid(column=0, row=2, columnspan=2, sticky="ew")

    # Sélection de l'image
    def choose_image():
        file_path = filedialog.askopenfilename()
        if file_path:
            info = run_detection(script_name.get(), file_path, root)  # Modifié pour inclure root
            script_info_label['text'] = info  # maj du label avec les infos de la dernière exécution

    image_btn = ttk.Button(frame, text="Choisir une image", command=choose_image, style="TButton")
    image_btn.grid(column=0, row=1, columnspan=2, pady=10, sticky="ew")

    root.mainloop()

if __name__ == "__main__":
    main()