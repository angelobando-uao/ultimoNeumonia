"""Módulo integrador (integrator.py)
Orquesta el preprocesamiento, inferencia del modelo y generación de Grad-CAM.
Retorna la clase predicha, la probabilidad y la imagen con el mapa de calor.
"""

from typing import Tuple, Optional, Union
import os
import numpy as np
from PIL import Image

from preprocess_img import preprocess
from load_model import model_fun
from grad_cam import grad_cam
from read_img import load_image


CLASS_LABELS = {
    0: "bacteriana",
    1: "normal",
    2: "viral",
}


def predict(
    array: np.ndarray, model=None
) -> Tuple[str, float, np.ndarray]:
    """Ejecuta el flujo completo de predicción para una imagen de entrada.

    Pasos:
        1. Preprocesamiento de la imagen (retorna formato batch tensor)
        2. Inferencia con el modelo CNN
        3. Determinación de la clase y porcentaje de probabilidad
        4. Generación de Grad-CAM con mapa de calor superpuesto

    Retorna:
        Tuple[str, float, np.ndarray]:
            - label: "bacteriana" | "normal" | "viral"
            - proba: Porcentaje de confianza (float entre 0 y 100)
            - heatmap: Imagen RGB uint8 con mapa de calor superpuesto (512x512x3)
    """
    if array is None or array.size == 0:
        raise ValueError("El arreglo de imagen no puede estar vacío.")

    # 1. Preprocesamiento a batch
    batch_array_img = preprocess(array)

    # 2. Cargar modelo
    if model is None:
        model = model_fun()

    # Inferencia
    preds = model.predict(batch_array_img, verbose=0)
    prediction_idx = int(np.argmax(preds[0]))
    proba = float(np.max(preds[0])) * 100.0

    # Mapeo a etiqueta
    label = CLASS_LABELS.get(prediction_idx, "desconocido")

    # 3. Generación de Grad-CAM
    heatmap = grad_cam(array, model=model)

    return label, proba, heatmap


def predict_from_path(
    file_path: Union[str, os.PathLike], model=None
) -> Tuple[str, float, np.ndarray, Image.Image]:
    """Carga una imagen desde disco y retorna el diagnóstico completo y la imagen original."""
    img_rgb, img2show = load_image(file_path)
    label, proba, heatmap = predict(img_rgb, model=model)
    return label, proba, heatmap, img2show
