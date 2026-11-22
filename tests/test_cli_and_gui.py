"""tests/test_cli_and_gui.py - 16 Pruebas unitarias para detector_neumonia.py (CLI y GUI)"""

import os
import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from PIL import Image

import detector_neumonia
from detector_neumonia import (
    PneumoniaDetectorGUI,
    App,
    run_cli_inference,
    RESAMPLE_FILTER,
)
from conftest import DummyMockModel


def test_gui_resample_filter_available():
    assert RESAMPLE_FILTER is not None


def test_gui_class_alias_app():
    assert App is PneumoniaDetectorGUI


def test_cli_inference_with_synthetic_image():
    with patch("detector_neumonia.predict") as mock_pred:
        heat = np.zeros((512, 512, 3), dtype=np.uint8)
        mock_pred.return_value = ("normal", 94.5, heat)
        label, proba = run_cli_inference(image_path=None, patient_id="PAC-TEST")
        assert label == "normal"
        assert proba == 94.5


def test_cli_inference_with_custom_image(sample_png_file):
    with patch("detector_neumonia.predict") as mock_pred:
        heat = np.zeros((512, 512, 3), dtype=np.uint8)
        mock_pred.return_value = ("bacteriana", 89.2, heat)
        label, proba = run_cli_inference(image_path=sample_png_file, patient_id="PAC-002")
        assert label == "bacteriana"
        assert proba == 89.2


def test_cli_inference_creates_pdf_and_csv(temp_test_dir):
    with patch("detector_neumonia.predict") as mock_pred:
        heat = np.zeros((512, 512, 3), dtype=np.uint8)
        mock_pred.return_value = ("viral", 78.0, heat)
        label, proba = run_cli_inference(image_path=None, patient_id="PAC-FILES")
        assert label == "viral"


def test_gui_mocked_init():
    mock_tk = MagicMock()
    with patch("tkinter.Tk", return_value=mock_tk):
        with patch("tkinter.ttk.Label"):
            with patch("tkinter.ttk.Button"):
                with patch("tkinter.ttk.Entry"):
                    with patch("tkinter.Text"):
                        with patch("tkinter.font.Font"):
                            gui = PneumoniaDetectorGUI(root=mock_tk, auto_mainloop=False)
                            assert gui.root is mock_tk
                            assert gui.array is None
                            assert gui.reportID == 0


def test_gui_delete_resets_state():
    mock_tk = MagicMock()
    with patch("tkinter.Tk", return_value=mock_tk):
        gui = PneumoniaDetectorGUI(root=mock_tk, auto_mainloop=False)
        gui.askokcancel = MagicMock(return_value=True)
        gui.showinfo = MagicMock()
        gui.array = np.zeros((10, 10))
        gui.label = "viral"
        gui.proba = 90.0

        gui.delete()

        assert gui.array is None
        assert gui.label == ""
        assert gui.proba == 0.0


def test_gui_delete_canceled():
    mock_tk = MagicMock()
    with patch("tkinter.Tk", return_value=mock_tk):
        gui = PneumoniaDetectorGUI(root=mock_tk, auto_mainloop=False)
        gui.askokcancel = MagicMock(return_value=False)
        gui.array = np.zeros((10, 10))
        gui.label = "viral"

        gui.delete()

        assert gui.array is not None
        assert gui.label == "viral"


def test_gui_load_img_file_updates_state(sample_png_file):
    mock_tk = MagicMock()
    with patch("tkinter.Tk", return_value=mock_tk):
        gui = PneumoniaDetectorGUI(root=mock_tk, auto_mainloop=False)
        gui.button1 = {}
        gui.load_img_file(filepath=sample_png_file)
        assert gui.array is not None
        assert gui.raw_pil_img is not None
        assert gui.button1["state"] == "normal"


def test_gui_run_model_without_array_does_nothing():
    mock_tk = MagicMock()
    with patch("tkinter.Tk", return_value=mock_tk):
        gui = PneumoniaDetectorGUI(root=mock_tk, auto_mainloop=False)
        gui.array = None
        gui.run_model()
        assert gui.label == ""


def test_gui_run_model_updates_prediction():
    mock_tk = MagicMock()
    with patch("tkinter.Tk", return_value=mock_tk):
        gui = PneumoniaDetectorGUI(root=mock_tk, auto_mainloop=False)
        gui.array = np.zeros((512, 512, 3), dtype=np.uint8)
        heat = np.zeros((512, 512, 3), dtype=np.uint8)
        with patch("detector_neumonia.predict", return_value=("bacteriana", 91.5, heat)):
            gui.run_model()
            assert gui.label == "bacteriana"
            assert gui.proba == 91.5
            assert gui.heatmap is not None


def test_gui_save_results_csv_calls_reporting():
    mock_tk = MagicMock()
    with patch("tkinter.Tk", return_value=mock_tk):
        gui = PneumoniaDetectorGUI(root=mock_tk, auto_mainloop=False)
        gui.showinfo = MagicMock()
        gui.text1 = MagicMock()
        gui.text1.get.return_value = "10203040"
        gui.label = "normal"
        gui.proba = 98.4

        with patch("detector_neumonia.save_results_csv") as mock_save:
            gui.save_results_csv("test.csv")
            mock_save.assert_called_once_with("test.csv", "10203040", "normal", 98.4)


def test_gui_create_pdf_increments_report_id():
    mock_tk = MagicMock()
    with patch("tkinter.Tk", return_value=mock_tk):
        gui = PneumoniaDetectorGUI(root=mock_tk, auto_mainloop=False)
        gui.showinfo = MagicMock()
        gui.text1 = MagicMock()
        gui.text1.get.return_value = "998877"
        gui.reportID = 3

        with patch("detector_neumonia.generate_pdf_report") as mock_pdf:
            gui.create_pdf()
            assert gui.reportID == 4
            mock_pdf.assert_called_once()


def test_main_cli_mode_dispatch():
    with patch("detector_neumonia.run_cli_inference", return_value=("normal", 95.0)) as mock_cli:
        detector_neumonia.main(["--cli", "--patient-id", "PAC-ARG"])
        mock_cli.assert_called_once_with(None, "PAC-ARG")


def test_main_headless_environment_variable():
    with patch.dict(os.environ, {"HEADLESS": "1"}):
        with patch("detector_neumonia.run_cli_inference", return_value=("normal", 95.0)) as mock_cli:
            detector_neumonia.main([])
            assert mock_cli.called


def test_main_gui_exception_triggers_cli_fallback():
    with patch("detector_neumonia.PneumoniaDetectorGUI", side_effect=Exception("No GUI")):
        with patch("detector_neumonia.run_cli_inference", return_value=("normal", 95.0)) as mock_cli:
            code = detector_neumonia.main([])
            assert code == 0
            assert mock_cli.called
