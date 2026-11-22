"""tests/conftest.py - Fixtures compartidas para la suite de pruebas unitarias."""

import os
import sys
import tempfile
import pytest
import numpy as np
from PIL import Image

# Agregar el directorio UAO-Neumonia a sys.path
PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

# Configurar KERAS_HOME seguro
temp_k = os.path.join(tempfile.gettempdir(), ".keras_test")
os.environ["KERAS_HOME"] = temp_k
os.makedirs(temp_k, exist_ok=True)


class DummyMockModel:
    """Mock determinista de modelo CNN para pruebas ultra-rápidas."""

    def __init__(self, fixed_class_idx=1, confidence=0.92):
        self.fixed_class_idx = fixed_class_idx
        self.confidence = confidence
        self.name = "MockCNNModel"

    def predict(self, batch_array, verbose=0):
        rem = max(0.0, (1.0 - self.confidence) / 2.0)
        probs = [rem, rem, rem]
        probs[self.fixed_class_idx] = self.confidence
        return np.array([probs], dtype=np.float32)

    def get_layer(self, name):
        class MockLayer:
            def __init__(self, layer_name):
                self.name = layer_name
                self.output = None
        return MockLayer(name)


@pytest.fixture
def mock_model():
    return DummyMockModel(fixed_class_idx=1, confidence=0.88)


@pytest.fixture
def synthetic_grayscale_512():
    """Genera una imagen 2D sintética en escala de grises de 512x512."""
    h, w = 512, 512
    y, x = np.mgrid[:h, :w]
    pattern = (np.sin(x / 30.0) * np.cos(y / 30.0) * 127 + 128).astype(np.uint8)
    return pattern


@pytest.fixture
def synthetic_rgb_512(synthetic_grayscale_512):
    """Genera una imagen 3D RGB sintética de 512x512x3."""
    return np.stack([synthetic_grayscale_512] * 3, axis=-1)


@pytest.fixture
def temp_test_dir():
    """Directorio temporal aislado para pruebas de archivos."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def sample_png_file(temp_test_dir, synthetic_rgb_512):
    """Crea un archivo PNG temporal para pruebas de lectura."""
    filepath = os.path.join(temp_test_dir, "sample_xray.png")
    img = Image.fromarray(synthetic_rgb_512)
    img.save(filepath)
    return filepath


@pytest.fixture
def sample_jpg_file(temp_test_dir, synthetic_rgb_512):
    """Crea un archivo JPG temporal para pruebas de lectura."""
    filepath = os.path.join(temp_test_dir, "sample_xray.jpg")
    img = Image.fromarray(synthetic_rgb_512)
    img.save(filepath, quality=90)
    return filepath
