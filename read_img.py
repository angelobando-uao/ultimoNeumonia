"""Módulo de lectura de imágenes (read_img.py)
Soporta lectura de archivos médicos DICOM (.dcm) e imágenes estándar (.jpg, .jpeg, .png).
"""

import os
from typing import Tuple, Union
import numpy as np
from PIL import Image
import cv2

try:
    import pydicom as dicom
except ImportError:
    try:
        import dicom
    except ImportError:
        dicom = None


def read_dicom_file(path: Union[str, os.PathLike]) -> Tuple[np.ndarray, Image.Image]:
    """Lee una imagen médica en formato DICOM (.dcm).
    
    Retorna:
        Tuple[np.ndarray, Image.Image]: 
            - img_RGB: Arreglo numpy en formato RGB uint8 (H, W, 3)
            - img2show: Objeto PIL.Image para visualización en GUI
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Archivo no encontrado: {path}")

    if dicom is None:
        raise ImportError("La librería 'pydicom' no está instalada.")

    try:
        ds = dicom.dcmread(path)
    except Exception as e:
        # Fallback a dicom.read_file para versiones legadas de pydicom
        if hasattr(dicom, "read_file"):
            try:
                ds = dicom.read_file(path)
            except Exception:
                raise ValueError(f"No se pudo decodificar el archivo DICOM: {e}")
        else:
            raise ValueError(f"No se pudo decodificar el archivo DICOM: {e}")

    if not hasattr(ds, "pixel_array"):
        raise ValueError("El archivo DICOM no contiene datos de píxeles válidos.")

    img_array = ds.pixel_array
    return process_raw_pixel_array(img_array)


def read_jpg_file(path: Union[str, os.PathLike]) -> Tuple[np.ndarray, Image.Image]:
    """Lee una imagen en formato estándar (JPG, JPEG, PNG).
    
    Retorna:
        Tuple[np.ndarray, Image.Image]: 
            - img_RGB: Arreglo numpy en formato RGB uint8 (H, W, 3)
            - img2show: Objeto PIL.Image para visualización en GUI
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Archivo no encontrado: {path}")

    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"No se pudo cargar la imagen: {path}")

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img2show = Image.fromarray(img_rgb)
    return img_rgb, img2show


def process_raw_pixel_array(img_array: np.ndarray) -> Tuple[np.ndarray, Image.Image]:
    """Normaliza un pixel_array crudo a uint8 RGB y genera un objeto PIL.Image."""
    if not isinstance(img_array, np.ndarray):
        img_array = np.asarray(img_array)

    if img_array.size == 0:
        raise ValueError("El arreglo de imagen está vacío.")

    img2 = img_array.astype(np.float64)
    max_val = img2.max()
    min_val = img2.min()

    if max_val > min_val:
        img2 = (np.maximum(img2, 0) - min_val) / (max_val - min_val) * 255.0
    else:
        img2 = np.zeros_like(img2)

    img2_uint8 = np.uint8(img2)

    if len(img2_uint8.shape) == 2:
        img_rgb = cv2.cvtColor(img2_uint8, cv2.COLOR_GRAY2RGB)
    elif len(img2_uint8.shape) == 3 and img2_uint8.shape[2] == 1:
        img_rgb = cv2.cvtColor(img2_uint8[:, :, 0], cv2.COLOR_GRAY2RGB)
    elif len(img2_uint8.shape) == 3 and img2_uint8.shape[2] == 3:
        img_rgb = img2_uint8
    else:
        raise ValueError(f"Formato de dimensiones no soportado: {img2_uint8.shape}")

    img2show = Image.fromarray(img_rgb)
    return img_rgb, img2show


def load_image(path: Union[str, os.PathLike]) -> Tuple[np.ndarray, Image.Image]:
    """Carga automáticamente cualquier formato soportado (DICOM, JPG, PNG)."""
    ext = os.path.splitext(str(path))[1].lower()
    if ext in [".dcm", ".dicom"]:
        return read_dicom_file(path)
    elif ext in [".jpg", ".jpeg", ".png", ".bmp"]:
        return read_jpg_file(path)
    else:
        # Intentar primero DICOM, luego OpenCV
        try:
            return read_dicom_file(path)
        except Exception:
            return read_jpg_file(path)
