import pytest

from evaluation.visdrone_adapter import (
    build_seqinfo,
    detection_to_gt_row,
    visdrone_bbox_to_pixel_xywh,
)


def test_visdrone_bbox_to_pixel_xywh_converts_normalized_to_pixels():
    # FiftyOne normalize [x,y,w,h] (0-1), 1344x756 gorunut
    x, y, w, h = visdrone_bbox_to_pixel_xywh([0.5, 0.25, 0.1, 0.2], img_width=1344, img_height=756)
    assert x == pytest.approx(672.0) and y == pytest.approx(189.0)
    assert w == pytest.approx(134.4) and h == pytest.approx(151.2)


def test_detection_to_gt_row_converts_pedestrian():
    det = {"label": "pedestrian", "bounding_box": [0.1, 0.1, 0.05, 0.1], "index": 7, "visibility": 1}
    row = detection_to_gt_row(frame_number=42, detection=det, img_width=1000, img_height=1000)
    assert row == "42,7,100.00,100.00,50.00,100.00,1,1,1.00"


def test_detection_to_gt_row_keeps_people_category_too():
    det = {"label": "people", "bounding_box": [0.0, 0.0, 0.1, 0.1], "index": 3, "visibility": 0}
    row = detection_to_gt_row(frame_number=1, detection=det, img_width=100, img_height=100)
    assert row is not None
    assert row.split(",")[-1] == "0.00"  # ikili visibility=0 dogrudan tasindi


def test_detection_to_gt_row_drops_vehicle_classes():
    for label in ("car", "van", "truck", "bus", "motor", "bicycle", "tricycle", "awning-tricycle"):
        det = {"label": label, "bounding_box": [0.0, 0.0, 0.1, 0.1], "index": 1, "visibility": 1}
        assert detection_to_gt_row(1, det, 100, 100) is None


def test_detection_to_gt_row_drops_ignored_region():
    det = {"label": "ignored_region", "bounding_box": [0.0, 0.0, 0.1, 0.1], "index": 1, "visibility": 1}
    assert detection_to_gt_row(1, det, 100, 100) is None


def test_detection_to_gt_row_defaults_visibility_to_one_when_missing():
    det = {"label": "pedestrian", "bounding_box": [0.0, 0.0, 0.1, 0.1], "index": 1}
    row = detection_to_gt_row(1, det, 100, 100)
    assert row.split(",")[-1] == "1.00"


def test_build_seqinfo_matches_mot_metrics_read_seq_fps_expectations():
    ini = build_seqinfo("uav0000305_00000_v", img_width=1344, img_height=756, seq_length=184)
    assert "[Sequence]" in ini
    assert "name=uav0000305_00000_v" in ini
    assert "imDir=img1" in ini
    assert "frameRate=30" in ini
    assert "seqLength=184" in ini
