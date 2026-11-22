"""Módulo de generación de reportes (reporting.py)
Exporta historial a formato CSV y genera reportes médicos en PDF.
"""

import os
import csv
import tempfile
from typing import Optional, Union
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import img2pdf


def save_results_csv(
    filepath: Union[str, os.PathLike],
    patient_id: str,
    label: str,
    proba: float,
    delimiter: str = "-",
) -> str:
    """Guarda el resultado del paciente en un archivo CSV con el delimitador '-'."""
    patient_id_clean = str(patient_id).strip() or "SIN_ID"
    proba_str = f"{proba:.2f}%"

    try:
        with open(filepath, "a", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile, delimiter=delimiter)
            writer.writerow([patient_id_clean, label, proba_str])
        return f"Guardado con éxito en {filepath}"
    except (PermissionError, OSError) as e:
        fallback_path = os.path.join(tempfile.gettempdir(), os.path.basename(str(filepath)))
        with open(fallback_path, "a", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile, delimiter=delimiter)
            writer.writerow([patient_id_clean, label, proba_str])
        return f"Guardado en ruta accesible: {fallback_path} (Aviso: {e})"


def create_medical_report_image(
    patient_id: str,
    label: str,
    proba: float,
    original_img: Optional[Union[np.ndarray, Image.Image]] = None,
    heatmap_img: Optional[Union[np.ndarray, Image.Image]] = None,
    canvas_size: tuple = (1000, 750),
) -> Image.Image:
    """Crea una imagen de informe médico de alta calidad para ser convertida a PDF."""
    canvas = Image.new("RGB", canvas_size, color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    # Encabezado
    draw.rectangle([(0, 0), (canvas_size[0], 80)], fill=(24, 76, 120))
    header_text = "SOFTWARE PARA EL APOYO AL DIAGNÓSTICO MÉDICO DE NEUMONÍA"
    draw.text((40, 28), header_text, fill=(255, 255, 255))

    # Información del paciente
    info_y = 110
    draw.text((50, info_y), f"Cédula Paciente: {patient_id or 'No especificada'}", fill=(30, 30, 30))
    draw.text((400, info_y), f"Diagnóstico Sugerido: {label.upper()}", fill=(200, 30, 30) if label != "normal" else (30, 150, 30))
    draw.text((750, info_y), f"Confianza: {proba:.2f}%", fill=(30, 30, 30))

    draw.line([(40, 145), (canvas_size[0] - 40, 145)], fill=(200, 200, 200), width=2)

    # Cuadros para imágenes
    box_w, box_h = 420, 420
    img1_x, img1_y = 50, 170
    img2_x, img2_y = 530, 170

    draw.text((img1_x, img1_y - 20), "Imagen Radiográfica (Original):", fill=(40, 40, 40))
    draw.text((img2_x, img2_y - 20), "Mapeo de Activaciones (Grad-CAM):", fill=(40, 40, 40))

    # Pegar o dibujar imagen 1
    if original_img is not None:
        if isinstance(original_img, np.ndarray):
            pil_1 = Image.fromarray(np.uint8(original_img))
        else:
            pil_1 = original_img
        pil_1_resized = pil_1.resize((box_w, box_h))
        canvas.paste(pil_1_resized, (img1_x, img1_y))
    else:
        draw.rectangle([(img1_x, img1_y), (img1_x + box_w, img1_y + box_h)], fill=(240, 240, 240), outline=(180, 180, 180))
        draw.text((img1_x + 120, img1_y + 190), "[Sin Imagen]", fill=(120, 120, 120))

    # Pegar o dibujar imagen 2
    if heatmap_img is not None:
        if isinstance(heatmap_img, np.ndarray):
            pil_2 = Image.fromarray(np.uint8(heatmap_img))
        else:
            pil_2 = heatmap_img
        pil_2_resized = pil_2.resize((box_w, box_h))
        canvas.paste(pil_2_resized, (img2_x, img2_y))
    else:
        draw.rectangle([(img2_x, img2_y), (img2_x + box_w, img2_y + box_h)], fill=(240, 240, 240), outline=(180, 180, 180))
        draw.text((img2_x + 120, img2_y + 190), "[Sin Heatmap]", fill=(120, 120, 120))

    # Pie de página
    footer_text = "Generado automáticamente por UAO-Neumonia - Herramienta de Soporte Clínico"
    draw.text((50, canvas_size[1] - 40), footer_text, fill=(130, 130, 130))

    return canvas


def generate_pdf_report(
    pdf_path: Union[str, os.PathLike],
    report_id: int,
    patient_id: str,
    label: str,
    proba: float,
    original_img: Optional[Union[np.ndarray, Image.Image]] = None,
    heatmap_img: Optional[Union[np.ndarray, Image.Image]] = None,
) -> str:
    """Genera un reporte PDF con img2pdf a partir del canvas médico."""
    report_image = create_medical_report_image(
        patient_id=patient_id,
        label=label,
        proba=proba,
        original_img=original_img,
        heatmap_img=heatmap_img,
    )

    temp_jpg = os.path.join(tempfile.gettempdir(), f"temp_report_{report_id}.jpg")
    report_image.save(temp_jpg, "JPEG", quality=95)

    target_pdf = str(pdf_path)
    try:
        with open(temp_jpg, "rb") as f_img:
            pdf_bytes = img2pdf.convert(f_img.read())
        try:
            with open(target_pdf, "wb") as f_pdf:
                f_pdf.write(pdf_bytes)
        except (PermissionError, OSError):
            target_pdf = os.path.join(tempfile.gettempdir(), os.path.basename(str(pdf_path)))
            with open(target_pdf, "wb") as f_pdf:
                f_pdf.write(pdf_bytes)
    finally:
        if os.path.exists(temp_jpg):
            try:
                os.remove(temp_jpg)
            except OSError:
                pass

    return target_pdf
