from pathlib import Path

import numpy as np
import pytest

from reid.feature_extractor import OSNetFeatureExtractor, cosine_similarity


@pytest.fixture(scope="module")
def extractor():
    return OSNetFeatureExtractor(model_name="osnet_x1_0")


def test_extract_returns_l2_normalized_embeddings(extractor):
    frame = (np.random.RandomState(0).rand(480, 640, 3) * 255).astype(np.uint8)
    boxes = [(50, 50, 150, 300), (300, 100, 400, 350)]
    feats = extractor.extract(frame, boxes)

    assert feats.shape == (2, extractor._feat_dim)
    norms = np.linalg.norm(feats, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-4)


def test_extract_empty_boxes_returns_empty_array(extractor):
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    feats = extractor.extract(frame, [])
    assert feats.shape == (0, extractor._feat_dim)


def test_cosine_similarity_identical_vector_is_one(extractor):
    frame = (np.random.RandomState(1).rand(480, 640, 3) * 255).astype(np.uint8)
    feats = extractor.extract(frame, [(50, 50, 150, 300)])
    sim = cosine_similarity(feats[0], feats[0])
    assert sim == pytest.approx(1.0, abs=1e-4)


def test_cosine_similarity_bounded():
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    assert cosine_similarity(a, b) == pytest.approx(0.0, abs=1e-6)
    assert -1.0 <= cosine_similarity(a, b) <= 1.0


def test_cosine_similarity_handles_none():
    assert cosine_similarity(None, np.array([1.0])) == 0.0


def test_trained_checkpoint_loads_when_present():
    """checkpoints/osnet_market1501/ Faz 2 egitiminden sonra mevcutsa,
    trusted-load duzeltmesinin (bkz. feature_extractor.py basi) hala
    calistigini dogrula. Checkpoint yoksa (Faz 2 hic calistirilmadiysa)
    atla -- bu testin var olmayan bir dosyaya bagimli olmasi yanlis olur."""
    ckpt = (
        Path(__file__).resolve().parent.parent
        / "checkpoints" / "osnet_market1501" / "model" / "model.pth.tar-15"
    )
    if not ckpt.is_file():
        pytest.skip("Faz 2 checkpoint'i henuz uretilmedi")

    ext = OSNetFeatureExtractor(model_name="osnet_x1_0", checkpoint_path=str(ckpt))
    assert ext.has_trained_weights is True

    frame = (np.random.RandomState(2).rand(480, 640, 3) * 255).astype(np.uint8)
    feats = ext.extract(frame, [(50, 50, 150, 300)])
    assert feats.shape == (1, 512)
    assert np.linalg.norm(feats[0]) == pytest.approx(1.0, abs=1e-4)
