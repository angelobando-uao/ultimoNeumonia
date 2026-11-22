"""tests/test_reporting.py - 16 Pruebas unitarias para reporting.py"""

import os
import csv
import pytest
import numpy as np
from PIL import Image

from reporting import (
    save_results_csv,
    create_medical_report_image,
    generate_pdf_report,
)


def test_save_results_csv_creates_file(temp_test_dir):
    csv_path = os.path.join(temp_test_dir, "test_historial.csv")
    msg = save_results_csv(csv_path, "12345678", "normal", 95.5)
    assert os.path.exists(csv_path)


def test_save_results_csv_appends_row(temp_test_dir):
    csv_path = os.path.join(temp_test_dir, "test_historial_append.csv")
    save_results_csv(csv_path, "ID1", "bacteriana", 88.0)
    save_results_csv(csv_path, "ID2", "viral", 75.2)

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = list(csv.reader(f, delimiter="-"))
        assert len(reader) == 2
        assert reader[0][0] == "ID1"
        assert reader[1][0] == "ID2"


def test_save_results_csv_uses_hyphen_delimiter(temp_test_dir):
    csv_path = os.path.join(temp_test_dir, "test_delim.csv")
    save_results_csv(csv_path, "PAC99", "normal", 99.12)
    with open(csv_path, "r", encoding="utf-8") as f:
        raw_line = f.readline().strip()
        assert "-" in raw_line
        assert raw_line == "PAC99-normal-99.12%"


def test_save_results_csv_patient_id_strip(temp_test_dir):
    csv_path = os.path.join(temp_test_dir, "test_strip.csv")
    save_results_csv(csv_path, "  PAC_ESPACIOS  ", "viral", 60.0)
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = list(csv.reader(f, delimiter="-"))
        assert reader[0][0] == "PAC_ESPACIOS"


def test_save_results_csv_empty_patient_id_fallback(temp_test_dir):
    csv_path = os.path.join(temp_test_dir, "test_empty_id.csv")
    save_results_csv(csv_path, "", "normal", 91.0)
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = list(csv.reader(f, delimiter="-"))
        assert reader[0][0] == "SIN_ID"


def test_save_results_csv_proba_formatting_percentage(temp_test_dir):
    csv_path = os.path.join(temp_test_dir, "test_pct.csv")
    save_results_csv(csv_path, "P1", "viral", 45.6789)
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = list(csv.reader(f, delimiter="-"))
        assert reader[0][2] == "45.68%"


def test_save_results_csv_custom_delimiter(temp_test_dir):
    csv_path = os.path.join(temp_test_dir, "test_comma.csv")
    save_results_csv(csv_path, "P2", "normal", 90.0, delimiter=",")
    with open(csv_path, "r", encoding="utf-8") as f:
        line = f.readline().strip()
        assert line == "P2,normal,90.00%"


def test_create_medical_report_image_dimensions():
    img = create_medical_report_image("123", "normal", 98.0, canvas_size=(800, 600))
    assert img.size == (800, 600)


def test_create_medical_report_image_mode_rgb():
    img = create_medical_report_image("123", "bacteriana", 85.0)
    assert img.mode == "RGB"


def test_create_medical_report_image_with_numpy_images(synthetic_rgb_512):
    heat = np.zeros((512, 512, 3), dtype=np.uint8)
    img = create_medical_report_image("P1", "viral", 70.0, original_img=synthetic_rgb_512, heatmap_img=heat)
    assert isinstance(img, Image.Image)


def test_create_medical_report_image_with_pil_images():
    pil_orig = Image.new("RGB", (200, 200), color="blue")
    pil_heat = Image.new("RGB", (200, 200), color="red")
    img = create_medical_report_image("P2", "normal", 92.0, original_img=pil_orig, heatmap_img=pil_heat)
    assert img.size == (1000, 750)


def test_create_medical_report_image_without_images():
    img = create_medical_report_image("P3", "bacteriana", 99.0, original_img=None, heatmap_img=None)
    assert isinstance(img, Image.Image)


def test_create_medical_report_image_label_colors():
    # Asegura que se renderizan etiquetas normales y patológicas sin excepciones
    img_norm = create_medical_report_image("ID_N", "normal", 95.0)
    img_bact = create_medical_report_image("ID_B", "bacteriana", 95.0)
    assert img_norm.size == img_bact.size


def test_generate_pdf_report_creates_valid_pdf(temp_test_dir, synthetic_rgb_512):
    pdf_out = os.path.join(temp_test_dir, "test_report.pdf")
    heat = np.zeros((512, 512, 3), dtype=np.uint8)
    res_path = generate_pdf_report(pdf_out, 1, "PAC-777", "normal", 96.5, original_img=synthetic_rgb_512, heatmap_img=heat)
    assert os.path.exists(res_path)
    assert os.path.getsize(res_path) > 1000


def test_generate_pdf_report_temp_jpg_cleanup(temp_test_dir):
    pdf_out = os.path.join(temp_test_dir, "clean_test.pdf")
    generate_pdf_report(pdf_out, 42, "PAC-042", "viral", 80.0)
    temp_jpg = os.path.join(temp_test_dir, "temp_report_42.jpg")
    assert not os.path.exists(temp_jpg)


def test_generate_pdf_report_returns_pdf_path(temp_test_dir):
    pdf_out = os.path.join(temp_test_dir, "return_check.pdf")
    path_returned = generate_pdf_report(pdf_out, 5, "PAC-005", "bacteriana", 89.0)
    assert path_returned.endswith(".pdf")
