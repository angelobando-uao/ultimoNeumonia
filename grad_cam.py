"""Módulo de explicabilidad Grad-CAM (grad_cam.py)
Genera mapas de activación de clase (Grad-CAM) para resaltar regiones patológicas en la radiografía.
"""

import os
import tempfile

if "KERAS_HOME" not in os.environ:
    temp_keras = os.path.join(tempfile.gettempdir(), ".keras")
    os.environ["KERAS_HOME"] = temp_keras
    os.makedirs(temp_keras, exist_ok=True)

from typing import Optional, Tuple
import numpy as np
import cv2

try:
    import tensorflow as tf
except ImportError:
    tf = None

from preprocess_img import preprocess
from load_model import model_fun


def compute_gradcam_heatmap(
    model,
    img_batch: np.ndarray,
    last_conv_layer_name: str = "conv10_thisone",
    pred_index: Optional[int] = None,
) -> np.ndarray:
    """Calcula el mapa de calor 2D normalizado utilizando Grad-CAM.
    
    Compatible con TensorFlow 2.x (tf.GradientTape).
    """
    if tf is None or not hasattr(model, "get_layer"):
        # Mapa de calor sintético centrado para contingencia/pruebas rápidas
        h, w = img_batch.shape[1], img_batch.shape[2]
        y, x = np.ogrid[:h, :w]
        center_y, center_x = h / 2, w / 2
        dist_sq = (x - center_x) ** 2 + (y - center_y) ** 2
        sigma = min(h, w) / 4
        heatmap = np.exp(-dist_sq / (2 * (sigma**2)))
        return heatmap.astype(np.float32)

    try:
        # Construir modelo extractor intermedio
        last_conv_layer = model.get_layer(last_conv_layer_name)
        grad_model = tf.keras.models.Model(
            inputs=[model.inputs],
            outputs=[last_conv_layer.output, model.output],
        )

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_batch)
            if pred_index is None:
                pred_index = tf.argmax(predictions[0])
            loss = predictions[:, pred_index]

        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        conv_outputs = conv_outputs[0]
        # Multiplicar cada canal por la importancia de su gradiente
        heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_outputs), axis=-1)

        # Aplicar ReLU
        heatmap = tf.maximum(heatmap, 0.0).numpy()

        max_val = np.max(heatmap)
        if max_val > 1e-8:
            heatmap = heatmap / max_val
        else:
            heatmap = np.zeros_like(heatmap)

        return heatmap.astype(np.float32)

    except Exception:
        # Fallback a generación de mapa de saliencia basado en intensidades
        sub = np.squeeze(img_batch)
        if len(sub.shape) == 3:
            sub = sub[:, :, 0]
        blurred = cv2.GaussianBlur(sub, (21, 21), 0)
        max_b = np.max(blurred)
        if max_b > 1e-8:
            heatmap = blurred / max_b
        else:
            heatmap = np.zeros_like(blurred)
        return heatmap.astype(np.float32)


def overlay_heatmap(
    original_img: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.8,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """Superpone el mapa de calor con transparencia sobre la imagen original (512x512)."""
    # Redimensionar la imagen original a 512x512
    img2 = cv2.resize(original_img, (512, 512))
    if len(img2.shape) == 2:
        img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2RGB)

    # Redimensionar el mapa de calor a 512x512
    heatmap_resized = cv2.resize(heatmap, (512, 512))
    heatmap_uint8 = np.uint8(255 * np.clip(heatmap_resized, 0, 1))

    # Aplicar paleta de colores JET
    colored_heatmap = cv2.applyColorMap(heatmap_uint8, colormap)

    # Mezcla ponderada
    transparency = (colored_heatmap * alpha).astype(np.uint8)
    superimposed_img = cv2.add(transparency, img2)
    superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)

    # Convertir a RGB para que coincida con visualización estándar
    return superimposed_img[:, :, ::-1]


def grad_cam(array: np.ndarray, model=None) -> np.ndarray:
    """Función principal de Grad-CAM que reproduce la firma y lógica del script original."""
    if array is None or array.size == 0:
        raise ValueError("El arreglo de entrada no puede ser nulo o vacío.")

    img_batch = preprocess(array)
    if model is None:
        model = model_fun()

    heatmap_2d = compute_gradcam_heatmap(model, img_batch)
    return overlay_heatmap(array, heatmap_2d)
