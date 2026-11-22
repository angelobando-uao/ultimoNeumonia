"""tests/test_read_img.py - 22 Pruebas unitarias para read_img.py"""

import os
import pytest
import numpy as np
from PIL import Image
from unittest.mock import MagicMock, patch

from read_img import (
    read_jpg_file,
    read_dicom_file,
    process_raw_pixel_array,
    load_image,
)


def test_read_jpg_file_success(sample_jpg_file):
    img_rgb, img2show = read_jpg_file(sample_jpg_file)
    assert isinstance(img_rgb, np.ndarray)
    assert isinstance(img2show, Image.Image)
    assert img_rgb.shape == (512, 512, 3)
    assert img_rgb.dtype == np.uint8


def test_read_jpg_file_not_found():
    with pytest.raises(FileNotFoundError):
        read_jpg_file("archivo_inexistente_12345.jpg")


def test_read_jpg_file_invalid_file(temp_test_dir):
    corrupt_file = os.path.join(temp_test_dir, "corrupt.jpg")
    with open(corrupt_file, "w") as f:
        f.write("contenido_no_imagen")
    with pytest.raises(ValueError):
        read_jpg_file(corrupt_file)


def test_read_jpg_file_png_format(sample_png_file):
    img_rgb, img2show = read_jpg_file(sample_png_file)
    assert img_rgb.ndim == 3
    assert img_rgb.shape[2] == 3


def test_read_dicom_file_not_found():
    with pytest.raises(FileNotFoundError):
        read_dicom_file("archivo_dicom_inexistente.dcm")


def test_read_dicom_file_missing_pixel_array(temp_test_dir):
    fake_dcm = os.path.join(temp_test_dir, "empty.dcm")
    with open(fake_dcm, "wb") as f:
        f.write(b"DICOM_HEADER_ONLY")
    with pytest.raises(ValueError):
        read_dicom_file(fake_dcm)


def test_read_dicom_mock_valid():
    mock_ds = MagicMock()
    mock_ds.pixel_array = np.random.randint(0, 4000, size=(256, 256), dtype=np.uint16)
    with patch("pydicom.dcmread", return_value=mock_ds):
        with patch("os.path.exists", return_value=True):
            img_rgb, img2show = read_dicom_file("dummy.dcm")
            assert img_rgb.shape == (256, 256, 3)
            assert img_rgb.dtype == np.uint8
            assert isinstance(img2show, Image.Image)


def test_read_dicom_fallback_to_read_file():
    mock_ds = MagicMock()
    mock_ds.pixel_array = np.zeros((128, 128), dtype=np.uint8)
    mock_dicom = MagicMock()
    mock_dicom.dcmread.side_effect = Exception("dcmread error")
    mock_dicom.read_file.return_value = mock_ds
    with patch("read_img.dicom", mock_dicom):
        with patch("os.path.exists", return_value=True):
            img_rgb, img2show = read_dicom_file("fallback.dcm")
            assert img_rgb.shape == (128, 128, 3)


def test_process_raw_pixel_array_2d():
    arr = np.random.randint(0, 256, size=(100, 100), dtype=np.uint8)
    img_rgb, img2show = process_raw_pixel_array(arr)
    assert img_rgb.shape == (100, 100, 3)
    assert img_rgb.dtype == np.uint8


def test_process_raw_pixel_array_3d_single_channel():
    arr = np.random.randint(0, 256, size=(80, 80, 1), dtype=np.uint8)
    img_rgb, img2show = process_raw_pixel_array(arr)
    assert img_rgb.shape == (80, 80, 3)


def test_process_raw_pixel_array_3d_three_channels():
    arr = np.random.randint(0, 256, size=(64, 64, 3), dtype=np.uint8)
    img_rgb, img2show = process_raw_pixel_array(arr)
    assert img_rgb.shape == (64, 64, 3)


def test_process_raw_pixel_array_empty():
    with pytest.raises(ValueError):
        process_raw_pixel_array(np.array([]))


def test_process_raw_pixel_array_uniform_constant():
    arr = np.ones((50, 50), dtype=np.float32) * 42.0
    img_rgb, img2show = process_raw_pixel_array(arr)
    assert img_rgb.shape == (50, 50, 3)
    assert np.all(img_rgb == 0)


def test_process_raw_pixel_array_negative_values():
    arr = np.array([[-100, 0], [100, 200]], dtype=np.float32)
    img_rgb, img2show = process_raw_pixel_array(arr)
    assert img_rgb.shape == (2, 2, 3)
    assert img_rgb.max() <= 255
    assert img_rgb.min() >= 0


def test_process_raw_pixel_array_high_dynamic_range():
    arr = np.array([[0, 2048], [1024, 4096]], dtype=np.uint16)
    img_rgb, img2show = process_raw_pixel_array(arr)
    assert img_rgb.dtype == np.uint8
    assert img_rgb.max() <= 255


def test_process_raw_pixel_array_unsupported_ndim():
    arr = np.zeros((10, 10, 2, 2))
    with pytest.raises(ValueError):
        process_raw_pixel_array(arr)


def test_load_image_auto_detect_jpg(sample_jpg_file):
    img_rgb, img2show = load_image(sample_jpg_file)
    assert img_rgb.shape == (512, 512, 3)


def test_load_image_auto_detect_png(sample_png_file):
    img_rgb, img2show = load_image(sample_png_file)
    assert img_rgb.shape == (512, 512, 3)


def test_load_image_auto_detect_dcm():
    mock_ds = MagicMock()
    mock_ds.pixel_array = np.ones((50, 50), dtype=np.uint8) * 100
    with patch("pydicom.dcmread", return_value=mock_ds):
        with patch("os.path.exists", return_value=True):
            img_rgb, img2show = load_image("paciente_001.dcm")
            assert img_rgb.shape == (50, 50, 3)


def test_load_image_unknown_extension_fallback(sample_png_file, temp_test_dir):
    strange_name = os.path.join(temp_test_dir, "xray.medicalformat")
    with open(sample_png_file, "rb") as f_src, open(strange_name, "wb") as f_dst:
        f_dst.write(f_src.read())
    img_rgb, img2show = load_image(strange_name)
    assert img_rgb.shape == (512, 512, 3)


def test_process_raw_pixel_array_list_input():
    nested_list = [[0, 100], [200, 255]]
    img_rgb, img2show = process_raw_pixel_array(nested_list)
    assert isinstance(img_rgb, np.ndarray)
    assert img_rgb.shape == (2, 2, 3)


def test_read_jpg_returns_pil_and_numpy(sample_jpg_file):
    res = read_jpg_file(sample_jpg_file)
    assert isinstance(res, tuple)
    assert len(res) == 2
    assert isinstance(res[0], np.ndarray)
    assert isinstance(res[1], Image.Image)
