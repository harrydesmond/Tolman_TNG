import numpy as np

from tolman.image_detection import detect_sources


def test_detect_sources_finds_bright_island():
    image = np.zeros((30, 30), dtype=float)
    image[10:14, 12:16] = 5.0
    detections = detect_sources(image, threshold_sigma=2.0, min_pixels=4)
    assert len(detections) == 1
    assert detections.loc[0, "seg_area_pix"] == 16
    assert 12.0 < detections.loc[0, "x"] < 15.0

