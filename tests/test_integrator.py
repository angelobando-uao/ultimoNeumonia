"""tests/test_integrator.py - 16 Pruebas unitarias para integrator.py"""

import pytest
import numpy as np
from PIL import Image

from integrator import (
    CLASS_LABELS,
    predict,
    predict_from_path,
)
from conftest import DummyMockModel


def test_predict_returns_tuple_of_three(synthetic_rgb_512, mock_model):
    res = predict(synthetic_rgb_512, model=mock_model)
    assert isinstance(res, tuple)
    assert len(res) == 3


def test_predict_label_in_valid_classes(synthetic_rgb_512, mock_model):
    label, proba, heatmap = predict(synthetic_rgb_512, model=mock_model)
    assert label in ["bacteriana", "normal", "viral"]


def test_predict_proba_range_0_100(synthetic_rgb_512, mock_model):
    label, proba, heatmap = predict(synthetic_rgb_512, model=mock_model)
    assert 0.0 <= proba <= 100.0


def test_predict_heatmap_shape_512_512_3(synthetic_rgb_512, mock_model):
    label, proba, heatmap = predict(synthetic_rgb_512, model=mock_model)
    assert heatmap.shape == (512, 512, 3)
    assert heatmap.dtype == np.uint8


def test_predict_none_array_raises():
    with pytest.raises(ValueError):
        predict(None)


def test_predict_empty_array_raises():
    with pytest.raises(ValueError):
        predict(np.array([]))


def test_predict_class_mapping_bacteriana(synthetic_rgb_512):
    bact_model = DummyMockModel(fixed_class_idx=0, confidence=0.95)
    label, proba, heatmap = predict(synthetic_rgb_512, model=bact_model)
    assert label == "bacteriana"
    assert proba > 90.0


def test_predict_class_mapping_normal(synthetic_rgb_512):
    norm_model = DummyMockModel(fixed_class_idx=1, confidence=0.99)
    label, proba, heatmap = predict(synthetic_rgb_512, model=norm_model)
    assert label == "normal"
    assert proba > 95.0


def test_predict_class_mapping_viral(synthetic_rgb_512):
    viral_model = DummyMockModel(fixed_class_idx=2, confidence=0.91)
    label, proba, heatmap = predict(synthetic_rgb_512, model=viral_model)
    assert label == "viral"
    assert proba > 85.0


def test_class_labels_dictionary_content():
    assert CLASS_LABELS[0] == "bacteriana"
    assert CLASS_LABELS[1] == "normal"
    assert CLASS_LABELS[2] == "viral"


def test_predict_from_path_returns_four_elements(sample_jpg_file, mock_model):
    res = predict_from_path(sample_jpg_file, model=mock_model)
    assert len(res) == 4
    label, proba, heatmap, img2show = res
    assert label in ["bacteriana", "normal", "viral"]
    assert isinstance(img2show, Image.Image)


def test_predict_from_path_invalid_file_raises(mock_model):
    with pytest.raises(FileNotFoundError):
        predict_from_path("archivo_inexistente_9999.jpg", model=mock_model)


def test_predict_proba_is_float(synthetic_rgb_512, mock_model):
    label, proba, heatmap = predict(synthetic_rgb_512, model=mock_model)
    assert isinstance(proba, float)


def test_predict_heatmap_is_uint8(synthetic_rgb_512, mock_model):
    label, proba, heatmap = predict(synthetic_rgb_512, model=mock_model)
    assert heatmap.dtype == np.uint8


def test_predict_grayscale_input_support(synthetic_grayscale_512, mock_model):
    label, proba, heatmap = predict(synthetic_grayscale_512, model=mock_model)
    assert label in ["bacteriana", "normal", "viral"]
    assert heatmap.shape == (512, 512, 3)


def test_predict_custom_model_injection(synthetic_rgb_512):
    custom_model = DummyMockModel(fixed_class_idx=0, confidence=0.77)
    label, proba, _ = predict(synthetic_rgb_512, model=custom_model)
    assert label == "bacteriana"
    assert abs(proba - 77.0) < 5.0
