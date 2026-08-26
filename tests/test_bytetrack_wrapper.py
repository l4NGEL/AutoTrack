from detection.yolo_detector import Detection
from tracking.bytetrack_wrapper import ByteTracker, iou


def _det(xyxy, conf=0.9, cls_name="person"):
    return Detection(xyxy=xyxy, conf=conf, cls_id=0, cls_name=cls_name)


def test_iou_identical_boxes_is_one():
    box = (10, 10, 50, 50)
    assert iou(box, box) == 1.0


def test_iou_disjoint_boxes_is_zero():
    assert iou((0, 0, 10, 10), (100, 100, 110, 110)) == 0.0


def test_iou_partial_overlap_between_zero_and_one():
    val = iou((0, 0, 10, 10), (5, 5, 15, 15))
    assert 0.0 < val < 1.0


def test_new_detection_creates_track_with_incrementing_ids():
    tracker = ByteTracker()
    tracks = tracker.update([_det((0, 0, 10, 10)), _det((100, 100, 110, 110))])
    assert sorted(t.track_id for t in tracks) == [1, 2]


def test_matching_detection_keeps_same_track_id_across_frames():
    tracker = ByteTracker(match_thresh=0.1)
    tracker.update([_det((0, 0, 10, 10))])
    tracks = tracker.update([_det((1, 1, 11, 11))])  # slightly moved, high IOU
    assert len(tracks) == 1
    assert tracks[0].track_id == 1


def test_track_pruned_after_track_buffer_missed_frames():
    tracker = ByteTracker(track_buffer=2)
    tracker.update([_det((0, 0, 10, 10))])
    for _ in range(3):
        tracks = tracker.update([])  # no detections -> track goes unmatched
    assert tracks == []
