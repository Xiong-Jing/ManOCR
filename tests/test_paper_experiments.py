import copy
import json
from pathlib import Path
import runpy

import numpy as np
import pytest
import torch
from PIL import Image

from manchu_ocr.data.orthography import build_transition_matrix, RULE_PAIRS
from manchu_ocr.data.page_splits import split_by_page, audit_page_splits
from manchu_ocr.data.label_generators.edge_shrink import edge_asymmetric_shrink
from manchu_ocr.data.label_generators.db_label_generator import DBLabelGenerator
from manchu_ocr.data.transforms.paper_detection_augmentation import elastic_warp
from manchu_ocr.losses.orthographic_transition_loss import OrthographicTransitionLoss
from manchu_ocr.models.detection.modules.vsaa import PaperVSAA
from manchu_ocr.api_eval.page_ocr import load_page_manifest, parse_page_response, aggregate_page_metrics
from manchu_ocr.api_eval.providers import build_openai_vision_payload
from manchu_ocr.utils.config import load_yaml
from manchu_ocr.utils.config import save_yaml
from manchu_ocr.utils.paper_protocol import (PAPER_PROTOCOL, bind_transition_prior,
                                           validate_paper_checkpoint, is_better_checkpoint)

ROOT = Path(__file__).resolve().parents[1]


def script(name):
    return runpy.run_path(str(ROOT / name))


def test_rule_matrix_exact_equation_and_exceptions():
    alphabet = list('aeovwiuqx.')
    labels = ['aewa', 'eov', 'wiva', 'aa', 'ae', 'a e']
    counts = np.ones((len(alphabet), len(alphabet)))
    idx = {c: i for i, c in enumerate(alphabet)}
    for label in labels:
        for word in label.split():
            for a, b in zip(word, word[1:]):
                counts[idx[a], idx[b]] += 1
    rule = np.ones_like(counts)
    for a, b in RULE_PAIRS:
        rule[idx[a], idx[b]] = .05
    expected = counts * rule
    expected /= expected.sum(axis=1, keepdims=True)
    expected = np.maximum(expected, 1e-6)
    expected /= expected.sum(axis=1, keepdims=True)
    got = build_transition_matrix(labels, alphabet)
    np.testing.assert_allclose(got, expected, rtol=1e-6)
    corpus = build_transition_matrix(labels, alphabet, rho=1)
    np.testing.assert_allclose(corpus, counts / counts.sum(axis=1, keepdims=True))
    for pair in [('e', 'o'), ('v', 'e'), ('w', 'a'), ('w', 'e'), ('a', 'a'), ('.', 'a')]:
        assert pair not in RULE_PAIRS
    assert np.all(got > 0)


@pytest.mark.parametrize('rho', [0, -0.1, 1.01, float('nan')])
def test_rule_matrix_rejects_invalid_rho(rho):
    with pytest.raises(ValueError):
        build_transition_matrix(['ae'], ['a', 'e'], rho=rho)


def test_otp_formula_gradient_and_batch_normalization(tmp_path):
    prior = np.array([[.8, .2], [.1, .9]], np.float32)
    path = tmp_path / 'T.npy'
    np.save(path, prior)
    logits = torch.tensor([[[5., 1., 0.], [4., 0., 1.]], [[0., 3., 0.], [0., 0., 3.]]], requires_grad=True)
    criterion = OrthographicTransitionLoss(path)
    actual = criterion(logits)
    q = logits.softmax(-1)
    mass = 1 - q[..., 0]
    p = q[..., 1:] / q[..., 1:].sum(-1, keepdim=True)
    scores = torch.einsum('bti,ij,btj->bt', p[:, :-1], torch.tensor(prior), p[:, 1:])
    weights = (mass[:, :-1] * mass[:, 1:]).detach()
    expected = (-(scores.clamp_min(1e-6).log() * weights).sum(1) / (weights.sum(1) + 1e-8)).mean()
    torch.testing.assert_close(actual, expected)
    grad_actual = torch.autograd.grad(actual, logits, retain_graph=True)[0]
    grad_expected = torch.autograd.grad(expected, logits)[0]
    torch.testing.assert_close(grad_actual, grad_expected)
    assert torch.isfinite(grad_actual).all()
    alone = (criterion(logits[:1]) + criterion(logits[1:])) / 2
    torch.testing.assert_close(actual, alone)


def test_otp_short_sequence_still_has_gradient(tmp_path):
    path = tmp_path / 'T.npy'
    np.save(path, np.ones((2, 2)) / 2)
    x = torch.ones(1, 1, 3, requires_grad=True)
    loss = OrthographicTransitionLoss(path)(x)
    loss.backward()
    assert torch.equal(x.grad, torch.zeros_like(x))


def test_group_split_and_cross_task_leakage(tmp_path):
    samples = [(f'crop{i}_{j}', 'ae') for i in range(20) for j in range(3)]
    mapping = {image: image.split('_')[0] for image, _ in samples}
    groups = split_by_page(samples, mapping)
    sets = [{mapping[path] for path, _ in group} for group in groups]
    assert not sets[0] & sets[1] and not sets[0] & sets[2] and not sets[1] & sets[2]
    assert groups == split_by_page(samples, mapping)
    fixed = {mapping[samples[0][0]]: 'test'}
    assert samples[0] in split_by_page(samples, mapping, assignments=fixed)[2]
    a, b = tmp_path / 'train.txt', tmp_path / 'test.txt'
    a.write_text('crop0_0\tae\n')
    b.write_text('crop0_1\tae\n')
    with pytest.raises(ValueError, match='leakage'):
        audit_page_splits({'det': {'train': a}, 'rec': {'test': b}}, mapping)
    with pytest.raises(ValueError, match='Missing source_page_id'):
        split_by_page(samples, {})


def test_as_offsets_follow_edge_orientation_and_winding():
    box = np.array([[10, 10], [30, 10], [30, 110], [10, 110]], np.float32)
    dx = 2000 * (1 - .4 ** 2) / 240
    for polygon in [box, box[::-1]]:
        result = edge_asymmetric_shrink(polygon)
        np.testing.assert_allclose(result.min(0), [10 + dx, 10 + dx / 3], atol=1e-5)
        np.testing.assert_allclose(result.max(0), [30 - dx, 110 - dx / 3], atol=1e-5)
    generator = DBLabelGenerator(use_asymmetric_shrink=True, as_mode='edge_offset', as_auxiliary=False)
    targets = generator([{'points': box.tolist()}], 128, 64)
    plain = DBLabelGenerator()([{'points': box.tolist()}], 128, 64)
    assert targets['prob_map'].sum() > plain['prob_map'].sum()


def test_paper_vsaa_is_single_spatial_multiplication():
    module = PaperVSAA(4, reduction=2, vertical_kernel=5, horizontal_kernel=3)
    with torch.no_grad():
        module.project.weight.zero_()
        module.project.bias.zero_()
    x = torch.randn(2, 4, 8, 6, requires_grad=True)
    torch.testing.assert_close(module(x), .5 * x)
    module(x).sum().backward()
    assert x.grad is not None


def test_elastic_pixels_and_annotations_share_forward_map():
    arr = np.zeros((32, 32, 3), np.uint8)
    arr[8:12, 8:12] = 255
    box = [{'points': [[8, 8], [11, 8], [11, 11], [8, 11]]}]
    dx, dy = np.full((32, 32), 3., np.float32), np.full((32, 32), 2., np.float32)
    image, items = elastic_warp(Image.fromarray(arr), box, dx, dy)
    assert items[0]['points'][0] == [11., 10.]
    np.testing.assert_array_equal(np.asarray(image)[10:14, 11:15], 255)


def test_page_metrics_count_words_not_exact_pages():
    metrics = aggregate_page_metrics(['ab cd', 'ef'], ['ab xx', ''])
    assert metrics['WA'] == pytest.approx(1 / 3)
    assert metrics['CER'] == pytest.approx(4 / 6)
    assert metrics['num_pages'] == 2
    assert metrics['word_substitutions'] == 1
    assert metrics['word_deletions'] == 1
    assert aggregate_page_metrics(['ab'], ['ab cd'])['word_insertions'] == 1


@pytest.mark.parametrize('text', ['', '```\nab\n```', 'I cannot transcribe this.', '<html>ab</html>', '无法识别'])
def test_page_refusals_and_unparseable_outputs_become_empty(text):
    assert parse_page_response(text) == ''


def make_page(tmp_path):
    image = tmp_path / 'page.png'
    Image.new('RGB', (64, 80), 'white').save(image)
    annotation = tmp_path / 'page.json'
    annotation.write_text(json.dumps({'items': [{'bbox': [4, 4, 12, 24], 'text': 'ab'},
                                               {'bbox': [4, 35, 12, 60], 'text': 'cd'}]}))
    manifest = tmp_path / 'test.txt'
    manifest.write_text(f'{image}\t{annotation}\n', encoding='utf-8')
    return image, annotation, manifest


def test_page_manifest_reads_gt_and_invalidates_changed_images(tmp_path):
    image, annotation, manifest = make_page(tmp_path)
    first = load_page_manifest(manifest)
    assert first[0].label == 'ab cd'
    Image.new('RGB', (64, 80), 'black').save(image)
    assert load_page_manifest(manifest)[0].sample_id != first[0].sample_id
    annotation.write_text(json.dumps({'items': [{'bbox': [4, 4, 12, 24], 'text': 'text'}]}))
    with pytest.raises(ValueError, match='complete'):
        load_page_manifest(manifest)


def test_page_api_payload_has_full_image_options_and_no_short_output_limit():
    cfg = load_yaml(ROOT / 'configs/experiments/lmm_page_zero_shot.yaml')
    model = cfg['models']['qwen2_5_vl_7b']
    payload = build_openai_vision_payload(model_id=model['model_id'], prompt=cfg['protocol']['prompt'],
        image_uri='data:image/png;base64,AAAA', generation=cfg['protocol']['generation'], image_options=model['image_options'])
    assert payload['messages'][0]['content'][0]['max_pixels'] == 1003520
    assert 'max_tokens' not in payload
    assert payload['vl_high_resolution_images'] is False
    assert cfg['models']['paddleocr_vl']['model_id'] == 'PaddleOCR-VL-1.5'


def test_rho_prepare_freezes_priors_and_preserves_fixed_variables(tmp_path):
    ns = script('scripts/run_rec_otp_rho_sensitivity.py')
    plan = load_yaml(ROOT / 'configs/experiments/rec_otp_rho_sensitivity.yaml')
    base = load_yaml(ROOT / plan['base_config'])
    train, charset = tmp_path / 'train.txt', tmp_path / 'charset.txt'
    train.write_text('image\tae\n', encoding='utf-8')
    charset.write_text('a\ne\n', encoding='utf-8')
    paths = {'recognition_data': {'train_list': str(train), 'charset': str(charset)}, 'outputs': {'root': str(tmp_path)}}
    configs = ns['prepare'](plan, paths, base, tmp_path / 'runs')
    assert len(configs) == 7
    for rho, path in configs:
        cfg = load_yaml(path)
        assert cfg['model'] == base['model'] and cfg['data'] == base['data']
        assert cfg['loss']['lambda_ortho'] == .1
        assert cfg['loss']['use_orthographic_loss']
        assert not cfg['loss']['use_alignment_loss'] and not cfg['decode']['use_lexicon']
    ns['prepare'](plan, paths, base, tmp_path / 'runs')
    train.write_text('image\tea\n')
    with pytest.raises(ValueError, match='Frozen'):
        ns['prepare'](plan, paths, base, tmp_path / 'runs')


def test_oracle_uses_same_recognizer_and_accounts_for_missed_gt():
    ns = script('scripts/eval_crop_oracle.py')
    calls = []
    class Pipeline:
        crop_padding = 0
        crop_padding_ratio = 0
        def recognizer(self, image):
            calls.append(image.size)
            return {'text': 'ab'}
        def __call__(self, image):
            return {'uses_ground_truth_crops': False, 'items': []}
    gt = [{'box': [[0, 0], [8, 0], [8, 20], [0, 20]], 'text': 'ab'}]
    record = ns['evaluate_crop_pair'](Pipeline(), Image.new('RGB', (32, 32)), gt)
    assert record['matching']['fn'] == 1 and record['predicted_exact_words'] == 0
    assert record['oracle_items'][0]['prediction'] == 'ab'
    assert len(calls) == 1


def test_error_analysis_requires_complete_manual_review():
    ns = script('tools/analyze_e2e_errors.py')
    page = {'image_path': 'p', 'matches': [{'text_exact': False, 'reference': 'ab', 'prediction': 'aa'}],
            'missed_ground_truth_items': [{'text': 'cd'}], 'false_positive_predictions': []}
    expected = ns['residual_events']([page])
    with pytest.raises(ValueError, match='Unreviewed'):
        ns['summarize_review'](expected, expected)
    reviewed = copy.deepcopy(expected)
    for row in reviewed:
        row.update(reviewed='yes', reviewer='human', category='adjacent_column_merge', instance_id='merged1')
    report = ns['summarize_review'](reviewed, expected)
    assert report['num_residual_events'] == 2 and report['num_reviewed_instances'] == 1
    assert sum(c['share_percent'] for c in report['categories']) == 100
    with pytest.raises(ValueError, match='exactly once'):
        ns['summarize_review'](reviewed[:1], expected)


def test_parameter_count_excludes_buffers_and_deduplicates_shared_layers():
    ns = script('tools/measure_model_cost.py')
    layer = torch.nn.Linear(2, 3)
    model = torch.nn.ModuleDict({'a': layer, 'b': layer})
    model.register_buffer('not_a_parameter', torch.ones(99))
    assert ns['parameter_counts'](model)['parameters'] == 9
    result = ns['benchmark'](layer, (1, 2), torch.device('cpu'), warmup=1, iterations=2)
    assert result['mean_batch_ms'] > 0 and result['peak_allocated_bytes'] is None


def test_page_score_is_word_error_rate_with_reference_denominator():
    from manchu_ocr.metrics.e2e_ocr_metrics import page_transcription_metrics
    # A character-normalized score would be high; one wrong word of two is 50.
    metrics = page_transcription_metrics('aaaaaaaaab cd', 'aaaaaaaaaa cd')
    assert metrics['page_transcription_score'] == 50.0
    assert page_transcription_metrics('ab x y z', 'ab')['page_transcription_score'] == 0
    assert aggregate_page_metrics(['aaaaaaaaaa cd'], ['aaaaaaaaab cd'])['page_transcription_score'] == 50


def test_checkpoint_tie_break_and_legacy_guard():
    assert is_better_checkpoint(.8, .1, .8, .2)
    assert not is_better_checkpoint(.8, .3, .8, .2)
    assert not is_better_checkpoint(.7, .01, .8, .2)
    cfg = {'experiment': {'protocol_version': PAPER_PROTOCOL}, 'loss': {'lambda_ortho': .1}}
    with pytest.raises(ValueError, match='Legacy'):
        validate_paper_checkpoint(cfg, {'config': {}})
    validate_paper_checkpoint(cfg, {'config': copy.deepcopy(cfg)})
    old = copy.deepcopy(cfg)
    old['loss']['lambda_ortho'] = .2
    with pytest.raises(ValueError, match='loss'):
        validate_paper_checkpoint(cfg, {'config': old})


def test_prior_provenance_rejects_changed_train_data(tmp_path):
    from manchu_ocr.data.orthography import write_transition_artifact
    train, charset, prior = tmp_path / 'train.txt', tmp_path / 'charset.txt', tmp_path / 'T.npy'
    train.write_text('image\tae\n')
    charset.write_text('a\ne\n')
    write_transition_artifact(prior, build_transition_matrix(['ae'], ['a', 'e']),
                              manifest=train, charset_path=charset, rho=.05, smoothing=1., epsilon_t=1e-6)
    cfg = {'experiment': {'protocol_version': PAPER_PROTOCOL},
           'loss': {'use_orthographic_loss': True, 'transition_matrix': str(prior)}}
    rec = {'train_list': str(train), 'charset': str(charset)}
    bind_transition_prior(cfg, rec)
    assert len(cfg['experiment']['transition_matrix_sha256']) == 64
    train.write_text('image\tea\n')
    with pytest.raises(ValueError, match='train_manifest'):
        bind_transition_prior(cfg, rec)


def test_whole_page_runner_scores_refusal_and_resumes_without_calls(tmp_path):
    from manchu_ocr.api_eval.providers import ProviderOutput
    image, _, manifest = make_page(tmp_path)
    paths_path, config_path = tmp_path / 'paths.yaml', tmp_path / 'plan.yaml'
    output = tmp_path / 'outputs'
    save_yaml({'e2e_data': {'test_list': str(manifest)}, 'outputs': {'root': str(output)}}, paths_path)
    cfg = load_yaml(ROOT / 'configs/experiments/lmm_page_zero_shot.yaml')
    cfg['experiment']['paths_config'] = str(paths_path)
    cfg['protocol']['retry'] = {'max_attempts': 1}
    cfg['models'] = {'fake': dict(cfg['models']['qwen2_5_vl_7b'], workers=1)}
    save_yaml(cfg, config_path)
    ns = script('scripts/run_lmm_zero_shot.py')
    main = ns['main']
    calls = []
    class Provider:
        def predict(self, path):
            calls.append(path)
            return ProviderOutput('I cannot transcribe this.')
        def close(self):
            pass
    main.__globals__['validate_provider_environment'] = lambda config: []
    main.__globals__['create_provider'] = lambda *a, **kw: Provider()
    args = ['--config', str(config_path), '--splits', 'test', '--run-tag', 'mock']
    assert main(args) == 0
    assert main(args) == 0 and calls == [str(image)]
    result_path = next(output.rglob('eval_test.json'))
    result = json.loads(result_path.read_text(encoding='utf-8'))
    assert result['task'] == 'page_ocr'
    assert result['metrics']['WA'] == 0 and result['metrics']['CER'] == 1
    assert result['metrics']['num_pages'] == 1
    assert result['metrics']['page_transcription_score'] == 0
    assert result['protocol']['input_unit'] == 'full_page'


def test_structured_api_refusal_is_an_empty_successful_prediction():
    from manchu_ocr.api_eval.providers import extract_openai_message_text
    assert extract_openai_message_text({'choices': [{'message': {'content': None, 'refusal': 'refused'}}]}) == ''


def test_page_subset_preserves_annotation_paths_and_content_identity(tmp_path):
    from manchu_ocr.data.page_splits import read_manifest
    rows = []
    for index in range(3):
        folder = tmp_path / str(index)
        folder.mkdir()
        image, annotation, _ = make_page(folder)
        annotation.write_text(json.dumps({'items': [
            {'bbox': [4, 4, 12, 24], 'text': 'ab'},
            {'bbox': [35, 4, 45, 24], 'text': 'cd'}]}))
        rows.append(f'{image}\t{annotation}\n')
    manifest = tmp_path / 'all.txt'
    manifest.write_text(''.join(rows), encoding='utf-8')
    cfg = load_yaml(ROOT / 'configs/experiments/lmm_page_zero_shot.yaml')
    ns = script('scripts/run_lmm_zero_shot.py')
    kwargs = dict(experiment_config=cfg, paths_config={'e2e_data': {'test_list': str(manifest)}},
                  selected_splits=['test'], output_root=tmp_path / 'outputs', run_tag='subset',
                  max_samples=2, skip_image_check=False, overwrite=False)
    first = ns['prepare_splits'](**kwargs)['test']
    assert len(read_manifest(first.evaluation_manifest)) == 2
    reloaded = load_page_manifest(first.evaluation_manifest)
    assert [item.label for item in reloaded] == [item.label for item in first.selected_samples]
    assert all('\n' in item.label for item in reloaded)
    Image.new('RGB', (64, 80), 'black').save(first.selected_samples[0].image_path)
    second = ns['prepare_splits'](**kwargs)['test']
    assert second.selected_sha256 != first.selected_sha256


def test_prior_cli_rejects_malformed_training_rows(tmp_path):
    manifest = tmp_path / 'train.txt'
    manifest.write_text('image\tab\nmalformed\n', encoding='utf-8')
    with pytest.raises(ValueError, match='Malformed'):
        script('tools/build_transition_matrix.py')['read_labels'](manifest)


def test_detector_paper_config_forward_and_target_losses():
    from manchu_ocr.models.detection.builder import build_detection_model
    from manchu_ocr.losses.db_loss import DBLoss
    cfg = load_yaml(ROOT / 'configs/detection/dbnetpp_vsaa_asym_shrink.yaml')
    model = build_detection_model(cfg).eval()
    generator = DBLabelGenerator(use_asymmetric_shrink=True, as_mode='edge_offset', as_auxiliary=False)
    target = generator([{'points': [[15, 3], [30, 3], [30, 60], [15, 60]]}], 64, 64)
    batch = {key: torch.tensor(value)[None, None] for key, value in target.items() if key != 'num_valid_polygons'}
    with torch.no_grad():
        out = model(torch.zeros(1, 3, 64, 64))
        losses = DBLoss(**cfg['loss'])(out, batch)
    assert torch.isfinite(losses['loss']) and float(losses['as_loss']) == 0.0
