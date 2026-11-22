"""Módulo de preprocesamiento de imágenes (preprocess_img.py)
Aplica resize a 512x512, conversión a escala de grises, CLAHE y normalización [0, 1].
"""

from typing import Tuple
import numpy as np
import cv2


def to_grayscale(array: np.ndarray) -> np.ndarray:
    """Convierte una imagen de 1 o 3 canales a escala de grises de 1 canal uint8."""
    if array is None or array.size == 0:
        raise ValueError("El arreglo de entrada no puede ser nulo ni vacío.")

    arr = np.asarray(array)

    if arr.dtype != np.uint8:
        # Escalar o convertir a uint8 de manera segura
        if arr.max() <= 1.0 and arr.min() >= 0.0:
            arr = (arr * 255.0).astype(np.uint8)
        else:
            arr = np.clip(arr, 0, 255).astype(np.uint8)

    if len(arr.shape) == 2:
        return arr
    elif len(arr.shape) == 3:
        if arr.shape[2] == 1:
            return arr[:, :, 0]
        elif arr.shape[2] == 3:
            return cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        elif arr.shape[2] == 4:
            return cv2.cvtColor(arr, cv2.COLOR_RGBA2GRAY)

    raise ValueError(f"Formato de dimensiones no soportado para escala de grises: {arr.shape}")


def apply_clahe(
    gray_img: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: Tuple[int, int] = (4, 4),
) -> np.ndarray:
    """Aplica ecualización adaptativa de histograma limitada por contraste (CLAHE)."""
    if gray_img.dtype != np.uint8:
        gray_img = np.clip(gray_img, 0, 255).astype(np.uint8)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(gray_img)


def resize_image(
    image: np.ndarray, target_size: Tuple[int, int] = (512, 512)
) -> np.ndarray:
    """Redimensiona una imagen al tamaño especificado (ancho, alto)."""
    if image is None or image.size == 0:
        raise ValueError("Imagen vacía no se puede redimensionar.")
    return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)


def normalize_image(image: np.ndarray) -> np.ndarray:
    """Normaliza los valores de la imagen a punto flotante en el rango [0.0, 1.0]."""
    img_float = image.astype(np.float32)
    return img_float / 255.0


def preprocess(
    array: np.ndarray,
    target_size: Tuple[int, int] = (512, 512),
    clip_limit: float = 2.0,
    tile_grid_size: Tuple[int, int] = (4, 4),
) -> np.ndarray:
    """Pipeline completo de preprocesamiento:
    1. Resize a (512, 512)
    2. Conversión a escala de grises
    3. CLAHE
    4. Normalización a [0.0, 1.0]
    5. Expansión a tensor batch (1, 512, 512, 1)

    Retorna:
        np.ndarray de forma (1, target_size[0], target_size[1], 1) float32
    """
    if array is None or array.size == 0:
        raise ValueError("El arreglo de entrada no puede ser None ni estar vacío.")

    # 1. Resize inicial
    resized = resize_image(array, target_size)

    # 2. Conversión a escala de grises
    gray = to_grayscale(resized)

    # 3. CLAHE
    clahe_img = apply_clahe(gray, clip_limit=clip_limit, tile_grid_size=tile_grid_size)

    # 4. Normalización
    norm_img = normalize_image(clahe_img)

    # 5. Expansión a formato batch tensor (1, H, W, 1)
    batch_img = np.expand_dims(norm_img, axis=-1)
    batch_img = np.expand_dims(batch_img, axis=0)

    return batch_img.astype(np.float32)
