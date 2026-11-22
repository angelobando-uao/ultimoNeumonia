"""tests/test_load_model.py - 22 Pruebas unitarias para load_model.py"""

import pytest
import numpy as np
from unittest.mock import patch

from load_model import (
    FallbackPneumoniaModel,
    WilhemNet86Architecture,
    model_fun,
    reset_model_cache,
)


def test_fallback_model_predict_shape():
    model = FallbackPneumoniaModel(num_classes=3)
    batch = np.zeros((1, 512, 512, 1), dtype=np.float32)
    preds = model.predict(batch)
    assert preds.shape == (1, 3)


def test_fallback_model_predict_probabilities_sum_to_one():
    model = FallbackPneumoniaModel()
    batch = np.ones((2, 512, 512, 1), dtype=np.float32) * 0.5
    preds = model.predict(batch)
    for p in preds:
        assert abs(np.sum(p) - 1.0) < 1e-4


def test_fallback_model_predict_classes_count():
    model = FallbackPneumoniaModel(num_classes=3)
    assert model.num_classes == 3


def test_fallback_model_batch_support():
    model = FallbackPneumoniaModel()
    batch = np.zeros((5, 512, 512, 1), dtype=np.float32)
    preds = model.predict(batch)
    assert preds.shape == (5, 3)


def test_fallback_model_get_layer():
    model = FallbackPneumoniaModel()
    layer = model.get_layer("conv10_thisone")
    assert layer.name == "conv10_thisone"


def test_fallback_model_name():
    model = FallbackPneumoniaModel()
    assert model.name == "FallbackWilhemNet86"


def test_wilhemnet86_build_architecture():
    model = WilhemNet86Architecture.build()
    assert model is not None
    assert model.name == "WilhemNet86"


def test_wilhemnet86_layer_names_present():
    model = WilhemNet86Architecture.build()
    layer_names = [l.name for l in model.layers]
    assert "input_1" in layer_names
    assert "conv10_thisone" in layer_names
    assert "predictions" in layer_names


def test_wilhemnet86_conv10_thisone_exists():
    model = WilhemNet86Architecture.build()
    layer = model.get_layer("conv10_thisone")
    assert layer is not None
    assert layer.filters == 64


def test_wilhemnet86_dropout_layers_present():
    model = WilhemNet86Architecture.build()
    dropout_layers = [l for l in model.layers if "dropout" in l.name]
    assert len(dropout_layers) >= 3


def test_wilhemnet86_input_shape_512_512_1():
    model = WilhemNet86Architecture.build()
    assert model.input_shape == (None, 512, 512, 1)


def test_wilhemnet86_output_shape_3_classes():
    model = WilhemNet86Architecture.build()
    assert model.output_shape == (None, 3)


def test_wilhemnet86_softmax_activation():
    model = WilhemNet86Architecture.build()
    pred_layer = model.get_layer("predictions")
    assert pred_layer.activation.__name__ == "softmax"


def test_model_fun_cached_singleton():
    reset_model_cache()
    m1 = model_fun()
    m2 = model_fun()
    assert m1 is m2
    reset_model_cache()


def test_reset_model_cache():
    reset_model_cache()
    m1 = model_fun()
    reset_model_cache()
    import load_model
    assert load_model._CACHED_MODEL is None


def test_model_fun_non_existent_weights_fallback():
    reset_model_cache()
    m = model_fun(weights_path="inexistente.h5", allow_fallback=True)
    assert m is not None
    reset_model_cache()


def test_model_fun_without_tf_fallback():
    reset_model_cache()
    with patch("load_model.tf", None):
        m = model_fun(allow_fallback=True)
        assert isinstance(m, FallbackPneumoniaModel)
    reset_model_cache()


def test_model_fun_without_tf_and_no_fallback_raises():
    reset_model_cache()
    with patch("load_model.tf", None):
        with pytest.raises(RuntimeError):
            model_fun(allow_fallback=False)
    reset_model_cache()


def test_wilhemnet86_skip_connections_add_layers():
    model = WilhemNet86Architecture.build()
    add_layers = [l for l in model.layers if "add" in l.name]
    assert len(add_layers) >= 5


def test_wilhemnet86_pooling_layers_count():
    model = WilhemNet86Architecture.build()
    pool_layers = [l for l in model.layers if "pool" in l.name]
    assert len(pool_layers) >= 5


def test_wilhemnet86_dense_layers_shapes():
    model = WilhemNet86Architecture.build()
    dense1 = model.get_layer("dense_1")
    dense2 = model.get_layer("dense_2")
    assert dense1.units == 1024
    assert dense2.units == 1024


def test_wilhemnet86_summary_generation():
    model = WilhemNet86Architecture.build()
    lines = []
    model.summary(print_fn=lambda x: lines.append(x))
    assert len(lines) > 0
    assert any("WilhemNet86" in l for l in lines)
