import hashlib
import json
from pathlib import Path

import pytest
import torch

from agentir.cache import MODEL_ID, digest, load_cache
from agentir.evaluation import matched_queries, metrics_from_ranks, query_text
from agentir.retrieval import ranks_for_targets


def make_cache(tmp_path):
    paths = [f'unlabeled2017/{i:012d}.jpg' for i in range(4)]
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps(paths), encoding='utf-8')
    vectors = torch.nn.functional.normalize(torch.arange(1, 1025).float().reshape(4, 256), dim=-1)
    cache = tmp_path / 'vectors.pth'
    torch.save((torch.arange(4), vectors), cache)
    metadata = {'identity': {'model': MODEL_ID, 'mode': 'eval', 'model_revision': 'a' * 40,
                             'transformers': '4.51.3', 'manifest_sha256': digest(manifest)},
                'sha256': digest(cache)}
    cache.with_suffix('.json').write_text(json.dumps(metadata), encoding='utf-8')
    return cache, manifest


def test_cache_uses_cpu_and_preserves_ids(tmp_path):
    path, manifest = make_cache(tmp_path)
    cache = load_cache(path, manifest, expected_count=4)
    assert cache.ids.tolist() == [0, 1, 2, 3]
    assert cache.vectors.device.type == 'cpu'
    assert cache.report['shape'] == [4, 256]


def test_manifest_reordering_cannot_silently_change_images(tmp_path):
    path, manifest = make_cache(tmp_path)
    manifest.write_text(json.dumps(list(reversed(json.loads(manifest.read_text())))))
    with pytest.raises(ValueError, match='manifest does not match'):
        load_cache(path, manifest, expected_count=4)


def test_cache_corruption_detected_before_loading(tmp_path):
    path, manifest = make_cache(tmp_path)
    path.write_bytes(b'bad download')
    with pytest.raises(ValueError, match='checksum'):
        load_cache(path, manifest, expected_count=4)


def test_wrong_image_ids_rejected_even_with_fresh_checksum(tmp_path):
    path, manifest = make_cache(tmp_path)
    _, vectors = torch.load(path, weights_only=True)
    torch.save((torch.tensor([3, 2, 1, 0]), vectors), path)
    meta = json.loads(path.with_suffix('.json').read_text())
    meta['sha256'] = digest(path)
    path.with_suffix('.json').write_text(json.dumps(meta))
    with pytest.raises(ValueError, match='IDs do not match'):
        load_cache(path, manifest, expected_count=4)


def test_non_normalized_vectors_rejected(tmp_path):
    path, manifest = make_cache(tmp_path)
    torch.save((torch.arange(4), torch.ones(4, 256)), path)
    meta = json.loads(path.with_suffix('.json').read_text())
    meta['sha256'] = digest(path)
    path.with_suffix('.json').write_text(json.dumps(meta))
    with pytest.raises(ValueError, match='normalization'):
        load_cache(path, manifest, expected_count=4)


def test_one_based_ranks_and_stable_ties():
    scores = torch.tensor([[.2, .9, .9, .1], [.5, .4, .3, .2]])
    assert ranks_for_targets(scores, [2, 0]).tolist() == [2, 1]
    with pytest.raises(ValueError, match='outside'):
        ranks_for_targets(scores, [4, 0])


def test_recall_can_fall_while_hits_stay_high():
    # Three rounds, two targets; one temporarily drops out after being found.
    ranks = [[1, 20], [11, 5], [2, 30]]
    rows, bri = metrics_from_ranks(ranks, [10])
    assert [r['recall_percent'] for r in rows] == [50., 50., 50.]
    assert [r['hits_percent'] for r in rows] == [50., 100., 100.]
    # Independent hand formula: target1 stays best=1; target2 goes 20 -> 5 -> 5.
    import math
    expected = (((math.log(20) + math.log(5)) / 2) + math.log(5)) / 4
    assert bri == pytest.approx(expected, abs=1e-6)


def test_single_state_has_no_bri_and_hits_equal_recall():
    rows, bri = metrics_from_ranks([[1, 11]], [1, 10, 50])
    assert bri is None
    assert all(r['recall_percent'] == r['hits_percent'] for r in rows)


def test_initial_caption_and_raw_vs_supplied_context():
    raw = {'img': 'x', 'dialog': ['a room', 'is it bright? yes', 'is there a desk? yes']}
    supplied = {'img': 'x', 'dialog': ['a room', 'a bright room', 'a bright room with a desk']}
    pairs = matched_queries([raw], [supplied], ['x'], required_rounds=3)
    assert len(pairs) == 1
    assert query_text(raw, 'raw', 2) == 'a room, is it bright? yes, is there a desk? yes'
    assert query_text(supplied, 'supplied', 2) == 'a bright room with a desk'
    bad = dict(supplied, dialog=['a different room'] * 3)
    with pytest.raises(ValueError, match='initial caption'):
        matched_queries([raw], [bad], ['x'], required_rounds=3)


def test_full_protocol_has_original_colab_first32():
    root = Path(__file__).resolve().parents[1]
    read = lambda name: json.loads((root / name).read_text(encoding='utf-8'))
    paths = read('assets/protocol/manifest_50k.json')
    pairs = matched_queries(read('assets/protocol/baseline_matched.json'),
                            read('assets/protocol/supplied_matched.json'), paths)
    ref = read('assets/colab_reference/run_config.json')
    assert len(paths) == 50_000
    assert len(pairs) == 2064
    assert [p[0]['img'] for p in pairs[:32]] == ref['targets']
    for field, index in [('baseline_sha256', 0), ('plugir_sha256', 1)]:
        payload = json.dumps([p[index] for p in pairs[:32]]).encode()
        assert hashlib.sha256(payload).hexdigest() == ref[field]
