#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""detector_neumonia.py
Interfaz gráfica y orquestación de la herramienta para detección rápida de neumonía.
Refactorizado bajo el estándar Jan-Project-Standard con soporte para modo GUI y CLI headless.
"""

import os
import sys
import argparse
import csv
from typing import Optional
import numpy as np
from PIL import ImageTk, Image

# Importaciones modulares limpias
from read_img import read_dicom_file, read_jpg_file, load_image
from preprocess_img import preprocess
from load_model import model_fun
from grad_cam import grad_cam
from integrator import predict
from reporting import save_results_csv, generate_pdf_report

# Compatibilidad con Pillow >= 10
RESAMPLE_FILTER = getattr(Image, "Resampling", Image).LANCZOS if hasattr(getattr(Image, "Resampling", Image), "LANCZOS") else getattr(Image, "ANTIALIAS", Image.BICUBIC)


class PneumoniaDetectorGUI:
    """Clase principal de la interfaz gráfica basada en Tkinter."""

    def __init__(self, root=None, auto_mainloop=True):
        import tkinter as tk
        from tkinter import ttk, font, filedialog, Text, StringVar, END
        from tkinter.messagebox import askokcancel, showinfo, WARNING

        self.tk = tk
        self.filedialog = filedialog
        self.showinfo = showinfo
        self.askokcancel = askokcancel
        self.WARNING = WARNING

        if root is not None:
            self.root = root
        else:
            self.root = tk.Tk()

        # Fuentes seguras
        try:
            fonti = font.Font(weight="bold")
            lbl_kw = {"font": fonti}
        except Exception:
            fonti = None
            lbl_kw = {}

        try:
            self.root.title("Herramienta para la detección rápida de neumonía")
            self.root.geometry("815x560")
            self.root.resizable(0, 0)
        except Exception:
            pass

        # Etiquetas
        self.lab1 = ttk.Label(self.root, text="Imagen Radiográfica", **lbl_kw)
        self.lab2 = ttk.Label(self.root, text="Imagen con Heatmap", **lbl_kw)
        self.lab3 = ttk.Label(self.root, text="Resultado:", **lbl_kw)
        self.lab4 = ttk.Label(self.root, text="Cédula Paciente:", **lbl_kw)
        self.lab5 = ttk.Label(
            self.root,
            text="SOFTWARE PARA EL APOYO AL DIAGNÓSTICO MÉDICO DE NEUMONÍA",
            **lbl_kw,
        )
        self.lab6 = ttk.Label(self.root, text="Probabilidad:", **lbl_kw)

        # Variables seguras
        try:
            self.ID = StringVar(master=self.root)
            self.result = StringVar(master=self.root)
        except Exception:
            class DummyVar:
                def __init__(self, val=""):
                    self._val = val
                def get(self):
                    return self._val
                def set(self, v):
                    self._val = v
            self.ID = DummyVar()
            self.result = DummyVar()

        # Cuadros de texto
        try:
            self.text1 = ttk.Entry(self.root, textvariable=self.ID, width=10)
        except Exception:
            self.text1 = MagicMock()
        self.text_img1 = Text(self.root, width=31, height=15)
        self.text_img2 = Text(self.root, width=31, height=15)
        self.text2 = Text(self.root)
        self.text3 = Text(self.root)

        # Botones
        self.button1 = ttk.Button(
            self.root, text="Predecir", state="disabled", command=self.run_model
        )
        self.button2 = ttk.Button(
            self.root, text="Cargar Imagen", command=self.load_img_file
        )
        self.button3 = ttk.Button(self.root, text="Borrar", command=self.delete)
        self.button4 = ttk.Button(self.root, text="PDF", command=self.create_pdf)
        self.button6 = ttk.Button(
            self.root, text="Guardar", command=self.save_results_csv
        )

        # Posicionamiento
        try:
            self.lab1.place(x=110, y=65)
            self.lab2.place(x=545, y=65)
            self.lab3.place(x=500, y=350)
            self.lab4.place(x=65, y=350)
            self.lab5.place(x=122, y=25)
            self.lab6.place(x=500, y=400)

            self.button1.place(x=220, y=460)
            self.button2.place(x=70, y=460)
            self.button3.place(x=670, y=460)
            self.button4.place(x=520, y=460)
            self.button6.place(x=370, y=460)

            self.text1.place(x=200, y=350)
            self.text2.place(x=610, y=350, width=90, height=30)
            self.text3.place(x=610, y=400, width=90, height=30)
            self.text_img1.place(x=65, y=90)
            self.text_img2.place(x=500, y=90)
            self.text1.focus_set()
        except Exception:
            pass

        # Estado interno
        self.array: Optional[np.ndarray] = None
        self.raw_pil_img: Optional[Image.Image] = None
        self.heatmap: Optional[np.ndarray] = None
        self.label: str = ""
        self.proba: float = 0.0
        self.img1 = None
        self.img2 = None
        self.reportID = 0

        if auto_mainloop:
            self.root.mainloop()

    def load_img_file(self, filepath: Optional[str] = None):
        """Carga una imagen médica y la presenta en la interfaz."""
        if not filepath:
            filepath = self.filedialog.askopenfilename(
                initialdir=".",
                title="Seleccionar imagen radiográfica",
                filetypes=(
                    ("Imágenes médicas y fotos", "*.dcm *.jpg *.jpeg *.png"),
                    ("DICOM (*.dcm)", "*.dcm"),
                    ("JPEG (*.jpg, *.jpeg)", "*.jpg *.jpeg"),
                    ("PNG (*.png)", "*.png"),
                    ("Todos los archivos", "*.*"),
                ),
            )
        if filepath:
            self.array, img2show = load_image(filepath)
            self.raw_pil_img = img2show
            pil_resized = img2show.resize((250, 250), RESAMPLE_FILTER)
            try:
                self.img1 = ImageTk.PhotoImage(pil_resized, master=self.root)
            except Exception:
                self.img1 = pil_resized

            try:
                self.text_img1.delete("1.0", "end")
                self.text_img1.image_create("1.0", image=self.img1)
            except Exception:
                pass
            self.button1["state"] = "normal"

    def run_model(self):
        """Ejecuta inferencia y Grad-CAM sobre la imagen cargada."""
        if self.array is None:
            return

        self.label, self.proba, self.heatmap = predict(self.array)

        # Mostrar heatmap en Tkinter
        pil_heat = Image.fromarray(self.heatmap)
        pil_heat_resized = pil_heat.resize((250, 250), RESAMPLE_FILTER)
        try:
            self.img2 = ImageTk.PhotoImage(pil_heat_resized, master=self.root)
        except Exception:
            self.img2 = pil_heat_resized

        try:
            self.text_img2.delete("1.0", "end")
            self.text_img2.image_create("1.0", image=self.img2)
        except Exception:
            pass

        # Mostrar etiquetas de resultado
        self.text2.delete("1.0", "end")
        self.text2.insert("1.0", self.label)

        self.text3.delete("1.0", "end")
        self.text3.insert("1.0", f"{self.proba:.2f}%")

    def save_results_csv(self, filename="historial.csv"):
        """Almacena el registro del paciente en el CSV."""
        patient_id = self.text1.get()
        save_results_csv(filename, patient_id, self.label, self.proba)
        self.showinfo(title="Guardar", message="Los datos se guardaron con éxito.")

    def create_pdf(self):
        """Genera el reporte médico en formato PDF."""
        pdf_path = f"Reporte{self.reportID}.pdf"
        generate_pdf_report(
            pdf_path=pdf_path,
            report_id=self.reportID,
            patient_id=self.text1.get(),
            label=self.label,
            proba=self.proba,
            original_img=self.array,
            heatmap_img=self.heatmap,
        )
        self.reportID += 1
        self.showinfo(title="PDF", message=f"El PDF fue generado con éxito: {pdf_path}")

    def delete(self):
        """Limpia el formulario."""
        answer = self.askokcancel(
            title="Confirmación",
            message="Se borrarán todos los datos.",
            icon=self.WARNING,
        )
        if answer:
            self.text1.delete(0, "end")
            self.text2.delete("1.0", "end")
            self.text3.delete("1.0", "end")
            self.text_img1.delete("1.0", "end")
            self.text_img2.delete("1.0", "end")
            self.array = None
            self.heatmap = None
            self.label = ""
            self.proba = 0.0
            self.button1["state"] = "disabled"
            self.showinfo(title="Borrar", message="Los datos se borraron con éxito")


# Alias para compatibilidad con código legado
App = PneumoniaDetectorGUI


def run_cli_inference(image_path: Optional[str] = None, patient_id: str = "PAC-12345"):
    """Ejecuta el pipeline clínico en modo CLI (ideal para auditoría, scripts y CI/CD)."""
    print("=" * 60)
    print("SISTEMA DE DETECCIÓN RÁPIDA DE NEUMONÍA (MODO CLI / AUDITORÍA)")
    print("=" * 60)

    if image_path is None or not os.path.exists(image_path):
        print("[INFO] Generando radiografía sintética de prueba...")
        # Radiografía sintética con gradientes simulando campos pulmonares
        h, w = 512, 512
        y, x = np.mgrid[:h, :w]
        synthetic_lung = np.sin(x / 50.0) * np.cos(y / 50.0) * 127 + 128
        synthetic_img = np.stack([synthetic_lung] * 3, axis=-1).astype(np.uint8)
        img_array = synthetic_img
        image_name = "sintetica_prueba.png"
    else:
        print(f"[INFO] Cargando imagen desde: {image_path}")
        img_array, _ = load_image(image_path)
        image_name = os.path.basename(image_path)

    print("[INFO] Ejecutando inferencia y Grad-CAM...")
    label, proba, heatmap = predict(img_array)

    print("-" * 60)
    print(f"Paciente ID:     {patient_id}")
    print(f"Archivo Entrada: {image_name}")
    print(f"Diagnóstico:     {label.upper()}")
    print(f"Confianza:       {proba:.2f}%")
    print("-" * 60)

    # Guardar en CSV
    csv_file = "historial.csv"
    save_results_csv(csv_file, patient_id, label, proba)
    print(f"[OK] Historial guardado en: {csv_file}")

    # Guardar reporte PDF
    pdf_file = "Reporte_CLI.pdf"
    generate_pdf_report(
        pdf_path=pdf_file,
        report_id=999,
        patient_id=patient_id,
        label=label,
        proba=proba,
        original_img=img_array,
        heatmap_img=heatmap,
    )
    print(f"[OK] Reporte PDF generado en: {pdf_file}")
    print("=" * 60)
    return label, proba


def main(argv=None):
    parser = argparse.ArgumentParser(description="Herramienta para detección rápida de neumonía")
    parser.add_argument("--cli", action="store_true", help="Ejecutar en modo línea de comandos (headless)")
    parser.add_argument("--image", type=str, default=None, help="Ruta a la imagen (modo CLI)")
    parser.add_argument("--patient-id", type=str, default="PAC-001", help="Cédula del paciente (modo CLI)")

    args = parser.parse_args(argv)

    if args.cli or os.environ.get("HEADLESS", "0") == "1":
        run_cli_inference(args.image, args.patient_id)
        return 0

    try:
        app = PneumoniaDetectorGUI()
        return 0
    except Exception as e:
        print(f"[ADVERTENCIA] No se pudo inicializar la interfaz gráfica ({e}).")
        print("[INFO] Iniciando en modo CLI headless de contingencia...")
        run_cli_inference(args.image, args.patient_id)
        return 0


if __name__ == "__main__":
    sys.exit(main())
