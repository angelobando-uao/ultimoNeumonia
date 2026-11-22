"""tests/test_preprocess_img.py - 26 Pruebas unitarias para preprocess_img.py"""

import pytest
import numpy as np
import cv2

from preprocess_img import (
    to_grayscale,
    apply_clahe,
    resize_image,
    normalize_image,
    preprocess,
)


def test_to_grayscale_from_2d():
    arr = np.zeros((100, 100), dtype=np.uint8)
    res = to_grayscale(arr)
    assert res.shape == (100, 100)
    assert res.dtype == np.uint8


def test_to_grayscale_from_3d_rgb():
    arr = np.ones((100, 100, 3), dtype=np.uint8) * 150
    res = to_grayscale(arr)
    assert res.shape == (100, 100)
    assert res.dtype == np.uint8


def test_to_grayscale_from_3d_single_channel():
    arr = np.ones((100, 100, 1), dtype=np.uint8) * 200
    res = to_grayscale(arr)
    assert res.shape == (100, 100)


def test_to_grayscale_from_4d_rgba():
    arr = np.ones((50, 50, 4), dtype=np.uint8) * 128
    res = to_grayscale(arr)
    assert res.shape == (50, 50)


def test_to_grayscale_none_input_raises():
    with pytest.raises(ValueError):
        to_grayscale(None)


def test_to_grayscale_empty_input_raises():
    with pytest.raises(ValueError):
        to_grayscale(np.array([]))


def test_to_grayscale_float_normalized():
    arr = np.array([[0.0, 0.5], [1.0, 0.2]], dtype=np.float32)
    res = to_grayscale(arr)
    assert res.dtype == np.uint8
    assert res[1, 0] == 255


def test_to_grayscale_clipping_large_numbers():
    arr = np.array([[300, 400], [500, 600]], dtype=np.float32)
    res = to_grayscale(arr)
    assert res.dtype == np.uint8
    assert np.all(res == 255)


def test_to_grayscale_unsupported_shape_raises():
    arr = np.zeros((10, 10, 5), dtype=np.uint8)
    with pytest.raises(ValueError):
        to_grayscale(arr)


def test_apply_clahe_output_shape():
    gray = np.random.randint(0, 256, (200, 200), dtype=np.uint8)
    res = apply_clahe(gray)
    assert res.shape == (200, 200)
    assert res.dtype == np.uint8


def test_apply_clahe_uint8_preservation():
    gray = np.zeros((100, 100), dtype=np.uint8)
    res = apply_clahe(gray)
    assert res.dtype == np.uint8


def test_apply_clahe_clip_limit_variation():
    gray = np.random.randint(50, 200, (100, 100), dtype=np.uint8)
    res1 = apply_clahe(gray, clip_limit=1.0)
    res2 = apply_clahe(gray, clip_limit=5.0)
    assert res1.shape == res2.shape


def test_apply_clahe_tile_grid_variation():
    gray = np.random.randint(50, 200, (128, 128), dtype=np.uint8)
    res = apply_clahe(gray, tile_grid_size=(8, 8))
    assert res.shape == (128, 128)


def test_resize_image_dimensions():
    img = np.zeros((300, 400), dtype=np.uint8)
    res = resize_image(img, (512, 512))
    assert res.shape == (512, 512)


def test_resize_image_arbitrary_target():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    res = resize_image(img, (256, 128))
    assert res.shape == (128, 256, 3)


def test_resize_image_empty_raises():
    with pytest.raises(ValueError):
        resize_image(np.array([]))


def test_resize_image_none_raises():
    with pytest.raises(ValueError):
        resize_image(None)


def test_normalize_image_range():
    arr = np.array([0, 128, 255], dtype=np.uint8)
    res = normalize_image(arr)
    assert res.min() == 0.0
    assert abs(res.max() - 1.0) < 1e-4


def test_normalize_image_float32_dtype():
    arr = np.ones((50, 50), dtype=np.uint8) * 100
    res = normalize_image(arr)
    assert res.dtype == np.float32


def test_normalize_image_zero_matrix():
    arr = np.zeros((10, 10), dtype=np.uint8)
    res = normalize_image(arr)
    assert np.all(res == 0.0)


def test_preprocess_pipeline_output_shape(synthetic_rgb_512):
    batch = preprocess(synthetic_rgb_512)
    assert batch.shape == (1, 512, 512, 1)


def test_preprocess_pipeline_batch_dimension_is_one():
    img = np.zeros((200, 200), dtype=np.uint8)
    batch = preprocess(img)
    assert batch.shape[0] == 1


def test_preprocess_pipeline_value_range(synthetic_rgb_512):
    batch = preprocess(synthetic_rgb_512)
    assert batch.min() >= 0.0
    assert batch.max() <= 1.0


def test_preprocess_none_input_raises():
    with pytest.raises(ValueError):
        preprocess(None)


def test_preprocess_empty_input_raises():
    with pytest.raises(ValueError):
        preprocess(np.array([]))


def test_preprocess_custom_target_size():
    img = np.zeros((100, 100), dtype=np.uint8)
    batch = preprocess(img, target_size=(256, 256))
    assert batch.shape == (1, 256, 256, 1)
