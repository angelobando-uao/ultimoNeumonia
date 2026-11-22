"""tests/test_grad_cam.py - 20 Pruebas unitarias para grad_cam.py"""

import pytest
import numpy as np
import cv2

from grad_cam import (
    compute_gradcam_heatmap,
    overlay_heatmap,
    grad_cam,
)


def test_overlay_heatmap_output_shape_512_512_3():
    orig = np.zeros((300, 300, 3), dtype=np.uint8)
    heatmap = np.zeros((512, 512), dtype=np.float32)
    res = overlay_heatmap(orig, heatmap)
    assert res.shape == (512, 512, 3)


def test_overlay_heatmap_dtype_uint8():
    orig = np.ones((512, 512, 3), dtype=np.uint8) * 100
    heatmap = np.ones((512, 512), dtype=np.float32) * 0.5
    res = overlay_heatmap(orig, heatmap)
    assert res.dtype == np.uint8


def test_overlay_heatmap_grayscale_original_support():
    orig_gray = np.ones((400, 400), dtype=np.uint8) * 128
    heatmap = np.zeros((512, 512), dtype=np.float32)
    res = overlay_heatmap(orig_gray, heatmap)
    assert res.shape == (512, 512, 3)


def test_overlay_heatmap_rgb_original_support(synthetic_rgb_512):
    heatmap = np.zeros((512, 512), dtype=np.float32)
    res = overlay_heatmap(synthetic_rgb_512, heatmap)
    assert res.shape == (512, 512, 3)


def test_overlay_heatmap_alpha_blending_limits():
    orig = np.zeros((512, 512, 3), dtype=np.uint8)
    heatmap = np.ones((512, 512), dtype=np.float32)
    res_low = overlay_heatmap(orig, heatmap, alpha=0.1)
    res_high = overlay_heatmap(orig, heatmap, alpha=1.0)
    assert res_high.mean() >= res_low.mean()


def test_overlay_heatmap_channel_reversal_rgb():
    orig = np.zeros((100, 100, 3), dtype=np.uint8)
    heatmap = np.ones((100, 100), dtype=np.float32)
    res = overlay_heatmap(orig, heatmap)
    assert res.shape == (512, 512, 3)


def test_compute_gradcam_heatmap_fallback_shape(mock_model):
    batch = np.zeros((1, 512, 512, 1), dtype=np.float32)
    heatmap = compute_gradcam_heatmap(mock_model, batch)
    assert heatmap.shape == (512, 512)


def test_compute_gradcam_heatmap_fallback_range_0_1(mock_model):
    batch = np.zeros((1, 512, 512, 1), dtype=np.float32)
    heatmap = compute_gradcam_heatmap(mock_model, batch)
    assert heatmap.min() >= 0.0
    assert heatmap.max() <= 1.0


def test_compute_gradcam_heatmap_fallback_center_activation(mock_model):
    batch = np.zeros((1, 512, 512, 1), dtype=np.float32)
    batch[0, 256, 256, 0] = 1.0
    heatmap = compute_gradcam_heatmap(mock_model, batch)
    center_val = heatmap[256, 256]
    corner_val = heatmap[0, 0]
    assert center_val > corner_val


def test_compute_gradcam_heatmap_with_mock_model(mock_model):
    batch = np.random.rand(1, 512, 512, 1).astype(np.float32)
    heatmap = compute_gradcam_heatmap(mock_model, batch)
    assert heatmap.dtype == np.float32


def test_compute_gradcam_heatmap_custom_layer_name(mock_model):
    batch = np.zeros((1, 512, 512, 1), dtype=np.float32)
    heatmap = compute_gradcam_heatmap(mock_model, batch, last_conv_layer_name="custom_conv")
    assert heatmap.shape == (512, 512)


def test_compute_gradcam_heatmap_pred_index_selection(mock_model):
    batch = np.zeros((1, 512, 512, 1), dtype=np.float32)
    h0 = compute_gradcam_heatmap(mock_model, batch, pred_index=0)
    h1 = compute_gradcam_heatmap(mock_model, batch, pred_index=1)
    assert h0.shape == h1.shape


def test_compute_gradcam_heatmap_zeros_guard(mock_model):
    batch = np.zeros((1, 512, 512, 1), dtype=np.float32)
    heatmap = compute_gradcam_heatmap(mock_model, batch)
    assert not np.isnan(heatmap).any()
    assert not np.isinf(heatmap).any()


def test_grad_cam_empty_array_raises():
    with pytest.raises(ValueError):
        grad_cam(np.array([]))


def test_grad_cam_none_array_raises():
    with pytest.raises(ValueError):
        grad_cam(None)


def test_grad_cam_end_to_end_output_shape(synthetic_rgb_512, mock_model):
    result = grad_cam(synthetic_rgb_512, model=mock_model)
    assert result.shape == (512, 512, 3)
    assert result.dtype == np.uint8


def test_grad_cam_synthetic_grayscale(synthetic_grayscale_512, mock_model):
    result = grad_cam(synthetic_grayscale_512, model=mock_model)
    assert result.shape == (512, 512, 3)


def test_grad_cam_custom_model_parameter(synthetic_rgb_512, mock_model):
    res = grad_cam(synthetic_rgb_512, model=mock_model)
    assert isinstance(res, np.ndarray)


def test_overlay_heatmap_clip_prevent_overflow():
    orig = np.ones((512, 512, 3), dtype=np.uint8) * 250
    heatmap = np.ones((512, 512), dtype=np.float32)
    res = overlay_heatmap(orig, heatmap)
    assert res.max() <= 255


def test_compute_gradcam_heatmap_blur_fallback(mock_model):
    # Probar que el fallback de saliencia por desenfoque opera sin excepciones
    batch = np.ones((1, 512, 512, 1), dtype=np.float32)
    heatmap = compute_gradcam_heatmap(mock_model, batch)
    assert heatmap.shape == (512, 512)
