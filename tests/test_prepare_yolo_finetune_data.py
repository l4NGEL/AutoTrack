"""evaluation/prepare_yolo_finetune_data.py icin testler -- goruntu/agir
dosya I/O gerektirmeyen SAF donusum mantigi: piksel-kutu -> normalize
YOLO etiketi, ve gt.txt/seqinfo.ini okuma. `build()`'un kendisi (gercek
VisDrone sahne verisi gerektirir) entegrasyon seviyesinde, burada test
edilmiyor -- projenin genel konvansiyonu (bkz. visdrone_adapter.py'nin de
build_sequence() icin birim testi yok, sadece saf yardimci fonksiyonlar test edilir)."""
from evaluation.prepare_yolo_finetune_data import _read_gt_by_frame, _read_seqinfo_size, _write_yolo_label


def test_write_yolo_label_centers_and_normalizes_box(tmp_path):
    dst = tmp_path / "out.txt"
    # 1920x1080 goruntude, (100,100)-(300,400) piksel kutusu (w=200, h=300)
    _write_yolo_label(dst, [(100, 100, 200, 300)], img_w=1920, img_h=1080)
    line = dst.read_text(encoding="utf-8").strip()
    cls, cx, cy, w, h = line.split()
    assert cls == "0"
    assert abs(float(cx) - (200 / 1920)) < 1e-4  # merkez x = 100+200/2 = 200
    assert abs(float(cy) - (250 / 1080)) < 1e-4  # merkez y = 100+300/2 = 250
    assert abs(float(w) - (200 / 1920)) < 1e-4
    assert abs(float(h) - (300 / 1080)) < 1e-4


def test_write_yolo_label_multiple_boxes_multiple_lines(tmp_path):
    dst = tmp_path / "out.txt"
    _write_yolo_label(dst, [(0, 0, 10, 10), (50, 50, 20, 20)], img_w=100, img_h=100)
    lines = dst.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2


def test_write_yolo_label_no_boxes_writes_empty_file(tmp_path):
    dst = tmp_path / "out.txt"
    _write_yolo_label(dst, [], img_w=100, img_h=100)
    assert dst.read_text(encoding="utf-8") == ""


def test_write_yolo_label_skips_zero_size_boxes(tmp_path):
    dst = tmp_path / "out.txt"
    _write_yolo_label(dst, [(10, 10, 0, 0), (10, 10, 5, 5)], img_w=100, img_h=100)
    lines = dst.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1  # sadece gecerli (pozitif boyutlu) kutu kaliyor


def test_write_yolo_label_clips_out_of_bounds_center_to_valid_range(tmp_path):
    dst = tmp_path / "out.txt"
    # merkez goruntu disina tasan bir kutu (kirpilmis GT'de olabilir)
    _write_yolo_label(dst, [(-50, -50, 20, 20)], img_w=100, img_h=100)
    line = dst.read_text(encoding="utf-8").strip()
    _cls, cx, cy, _w, _h = line.split()
    assert 0.0 <= float(cx) <= 1.0
    assert 0.0 <= float(cy) <= 1.0


def test_read_gt_by_frame_parses_pixel_columns(tmp_path):
    seq_dir = tmp_path / "seq"
    (seq_dir / "gt").mkdir(parents=True)
    (seq_dir / "gt" / "gt.txt").write_text(
        "1,5,10.00,20.00,30.00,40.00,1,1,1.00\n"
        "1,6,50.00,60.00,10.00,10.00,1,1,0.00\n"
        "2,5,11.00,21.00,30.00,40.00,1,1,1.00\n",
        encoding="utf-8",
    )
    gt = _read_gt_by_frame(seq_dir)
    assert set(gt.keys()) == {1, 2}
    assert len(gt[1]) == 2
    assert gt[1][0] == (10.0, 20.0, 30.0, 40.0)


def test_read_seqinfo_size_parses_width_height(tmp_path):
    seq_dir = tmp_path / "seq"
    seq_dir.mkdir(parents=True)
    (seq_dir / "seqinfo.ini").write_text(
        "[Sequence]\nname=test\nimDir=img1\nframeRate=30\nseqLength=10\nimWidth=1920\nimHeight=1080\nimExt=.jpg\n",
        encoding="utf-8",
    )
    w, h = _read_seqinfo_size(seq_dir)
    assert (w, h) == (1920, 1080)
