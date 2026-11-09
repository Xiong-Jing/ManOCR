import json
from pathlib import Path
import runpy
import sys

from PIL import Image
import pytest

from manchu_ocr.data.detection_protocol import (resolve_detection_data, validate_detection_inventory,
                                               validate_detection_manifests)
from manchu_ocr.utils.config import load_yaml, save_yaml

ROOT = Path(__file__).resolve().parents[1]


def test_3853_page_config_and_exact_seeded_partitions():
    cfg = load_yaml(ROOT / 'configs/datasets/detection_3853.yaml')
    split = cfg['split']
    assert cfg['expected_valid_pages'] == 3853
    assert split['counts'] == {'train': 3083, 'val': 385, 'test': 385}
    fn = runpy.run_path(str(ROOT / 'tools/prepare_detection_data.py'))['split_items']
    items = [(f'page{i}.png', f'page{i}.json') for i in range(3853)]
    groups = fn(items, 42, .8, .1, .1, split['counts'])
    assert list(map(len, groups)) == [3083, 385, 385]
    assert groups == fn(items, 42, .8, .1, .1, split['counts'])
    assert list(map(len, fn(items, 42, .8, .1, .1))) == [3083, 385, 385]
    assert len({x for group in groups for x in group}) == 3853
    assert items[0] == ('page0.png', 'page0.json')
    with pytest.raises(ValueError, match='3853'):
        fn(items[:-1], 42, .8, .1, .1, split['counts'])


def test_both_paths_configs_use_dataset_profile():
    for name in ['remote_server', 'local_windows']:
        cfg = resolve_detection_data(load_yaml(ROOT / f'configs/paths/{name}.yaml'))
        assert cfg['expected_raw_images'] == cfg['expected_raw_annotations'] == cfg['expected_valid_pages'] == 3853
        assert cfg['split']['counts'] == {'train': 3083, 'val': 385, 'test': 385}


def test_inventory_rejects_missing_or_duplicate_pages():
    with pytest.raises(ValueError, match='requires 3853'):
        validate_detection_inventory([('a', 'a.json')], {'expected_valid_pages': 3853})
    with pytest.raises(ValueError, match='same source image'):
        validate_detection_inventory([('a', 'a.json'), ('a', 'b.json')], {'expected_valid_pages': 2})


def test_preparation_checks_counts_before_replacing_existing_artifacts(tmp_path, monkeypatch):
    image_dir, ann_dir, clean_dir = [tmp_path / s for s in ['images', 'annotations', 'clean']]
    for p in [image_dir, ann_dir, clean_dir]:
        p.mkdir()
    for i in range(5):
        Image.new('RGB', (100, 100), 'white').save(image_dir / f'p{i}.png')
        (ann_dir / f'p{i}.json').write_text(json.dumps({'shapes': [
            {'label': 'manchu', 'shape_type': 'rectangle', 'points': [[20, 20], [30, 60]]}
        ]}))
    sentinel = clean_dir / 'existing.json'
    sentinel.write_text('preserve me')
    train = tmp_path / 'train.txt'
    train.write_text('existing split')
    cfg = {'detection_data': {
        'root': str(tmp_path), 'raw_images': str(image_dir), 'raw_annotations': str(ann_dir),
        'clean_annotations': str(clean_dir), 'train_list': str(train),
        'val_list': str(tmp_path / 'val.txt'), 'test_list': str(tmp_path / 'test.txt'),
        'expected_valid_pages': 6, 'split': {'counts': {'train': 4, 'val': 1, 'test': 1}},
        'cleaning': {'clear_clean_annotations': True, 'remove_side_band': False}}}
    config = tmp_path / 'paths.yaml'
    save_yaml(cfg, config)
    main = runpy.run_path(str(ROOT / 'tools/prepare_detection_data.py'))['main']
    monkeypatch.setattr(sys, 'argv', ['prepare_detection_data.py', '--config', str(config)])
    with pytest.raises(ValueError, match='requires 6'):
        main()
    assert sentinel.read_text() == 'preserve me' and train.read_text() == 'existing split'
    cfg['detection_data']['expected_valid_pages'] = 5
    cfg['detection_data']['split']['counts']['train'] = 3
    save_yaml(cfg, config)
    main()
    assert not sentinel.exists()
    assert validate_detection_manifests(cfg['detection_data']) == {'train': 3, 'val': 1, 'test': 1}
