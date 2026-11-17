import json
import logging
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from safetensors.torch import load_file

from src.app import entrypoint
from src.app.decider import ClefDecider
from src.eval.sequence import iptc_31 as evaluate
from src.train.sequence import iptc_31
from src.train.sequence.iptc_31 import clef_head


class TextBody(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(16, 4)
        self.dropout = torch.nn.Dropout(.9)
        self.grad_modes = []

    def forward(self, input_ids, attention_mask, use_cache, return_dict):
        self.grad_modes.append(torch.is_grad_enabled())
        assert not self.training and not use_cache
        return SimpleNamespace(last_hidden_state=self.dropout(self.embedding(input_ids)))


class Backbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = TextBody()

    def get_output_embeddings(self):
        return self.model.embedding


class Head(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = torch.nn.Linear(4, 2)

    def forward(self, hidden, input_ids, attention_mask, records, output_embedding_weight):
        return [[self.projection(hidden[row].mean(0)) for question in record.questions]
                for row, record in enumerate(records)]


class API:
    @staticmethod
    def encode_record(tokenizer, sample, max_length, max_state_tokens):
        assert set(sample) == {'id', 'state', 'questions'} and set(sample['state']) == {'article'}
        assert 'label' not in sample['state']
        # Reverse question and option ordering to exercise code/option-ID alignment.
        return SimpleNamespace(record_id=sample['id'], input_ids=(1, 2, 3), questions=[
            SimpleNamespace(question_id=code, option_ids=('false', 'true')) for code in reversed(sample['questions'])
        ])

    @staticmethod
    def collate_records(records, pad_token_id, device):
        ids = torch.tensor([record.input_ids for record in records], device=device)
        return {'input_ids': ids, 'attention_mask': torch.ones_like(ids), 'records': records, 'media': {}}


def test_only_head_gets_gradients_and_backbone_stays_deterministic():
    backbone, head = Backbone(), Head()
    model = clef_head.FrozenBackboneHead(SimpleNamespace(language_model=backbone, head=head), ['a', 'b'])
    model.train()
    record = API.encode_record(None, {'id': 'sl:1', 'state': {'article': 'Article'},
                                      'questions': {'a': {}, 'b': {}}}, 10, 10)
    batch = API.collate_records([record], 0, torch.device('cpu'))
    labels = torch.tensor([[1., 0.]])
    before = {key: value.clone() for key, value in backbone.state_dict().items()}
    optimizer = torch.optim.AdamW(model.head.parameters(), lr=.01)
    first = model(**batch, labels=labels)
    second = model(**batch, labels=labels)
    torch.testing.assert_close(first['logits'], second['logits'])
    expected = torch.nn.functional.cross_entropy(head.projection(backbone.model.embedding(batch['input_ids']).mean(1))
                                                 .expand(2, -1), torch.tensor([1, 0]))
    torch.testing.assert_close(first['loss'], expected)
    first['loss'].backward()
    assert all(parameter.grad is None and not parameter.requires_grad for parameter in backbone.parameters())
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in head.parameters())
    optimizer.step()
    assert not backbone.model.training and model.head.training and not any(backbone.model.grad_modes)
    for key, value in backbone.state_dict().items():
        torch.testing.assert_close(value, before[key])


def test_validation_groups_duplicate_texts_and_is_seeded():
    samples = [clef_head.Sample(f'sl:{row}', f'Article {row}', np.asarray([row % 2, 1.]), 'iptc-31.sl')
               for row in range(20)]
    samples.append(clef_head.Sample('hr:copy', 'Article  1', np.asarray([1., 1.]), 'iptc-31.hr'))
    training, validation = clef_head.validation_split(samples, .2, 2611)
    again = clef_head.validation_split(samples, .2, 2611)
    assert [[sample.sample_id for sample in part] for part in (training, validation)] == [
        [sample.sample_id for sample in part] for part in again
    ]
    assert {sample.sample_id for sample in training}.isdisjoint(sample.sample_id for sample in validation)
    assert ('sl:1' in {sample.sample_id for sample in training}) == ('hr:copy' in {sample.sample_id for sample in training})
    assert len(training) + len(validation) == len(samples)


def setup_run(tmp_path, monkeypatch, seed=2611):
    paths = entrypoint._ensure_dirs(tmp_path, 'train', 'sequence', 'iptc-31')
    prepare = paths.get_script_ctx_path('data', 'prepare')
    prepare.mkdir(parents=True)
    # Invalid test files would fail immediately if accidentally read.
    for language in ('hr', 'sl'):
        samples = [{'id': f'{language}:{row}', 'text': f'{language} Article {row}',
                    'label': ['01000000'] if row % 2 else ['15000000']} for row in range(8)]
        (prepare / f'iptc-31.{language}.train.jsonl').write_text(''.join(json.dumps(sample) + '\n' for sample in samples))
        (prepare / f'iptc-31.{language}.test.jsonl').write_text('INVALID TEST FILE\n')
    merged, _ = entrypoint._load_and_merge_configs(Path('conf'), 'sequence', 'iptc-31', ['clef-flash', 'clef-flash-head'])
    merged = entrypoint._apply_cli_overrides(merged, [
        'train.use_cpu=true', 'train.bf16=false', 'train.num_train_epochs=1',
        'train.gradient_accumulation_steps=2', 'train.disable_tqdm=true', 'train.logging_steps=1',
        f'train.seed={seed}',
    ])
    model_args, data_args, train_args, _ = entrypoint._parse_hf_args(merged)
    checkpoint = tmp_path / 'release'
    checkpoint.mkdir()
    (checkpoint / 'joint_head_config.json').write_text('{}')
    instances = []

    class Predictor(ClefDecider):
        def __init__(self, model_args, parameters):
            assert 'head_checkpoint' not in parameters
            self.model_args, self.max_length = model_args, model_args.max_seq_length
            self.parameters = dict(parameters)
            self.max_state_tokens = parameters.get('max_state_tokens')
            self.head_config = {}
            self.head_metadata = None
            self.device, self.dtype = torch.device('cpu'), torch.float32
            self.api, self.tokenizer, self.checkpoint = API, SimpleNamespace(pad_token_id=0), checkpoint
            self.model = SimpleNamespace(language_model=Backbone(), head=Head())
            self.original_head = {key: value.clone() for key, value in self.model.head.state_dict().items()}
            instances.append(self)

    monkeypatch.setattr(clef_head, 'ClefDecider', Predictor)
    monkeypatch.setattr(iptc_31, 'paths', paths, raising=False)
    monkeypatch.setattr(iptc_31, 'logger', logging.getLogger(__name__), raising=False)
    return paths, model_args, data_args, train_args, instances


def test_training_checkpoints_best_reload_and_resume_without_test_data(tmp_path, monkeypatch):
    paths, model_args, data_args, train_args, instances = setup_run(tmp_path, monkeypatch)
    iptc_31.main(data_args, model_args, train_args)
    output = Path(train_args.output_dir)
    assert output == paths.context / 'iptc-31.clef-flash-head.s2611'
    metadata = ClefDecider.read_head_manifest(output, model_args.model_name_or_path, '22b75507981345858dd152fbd9a6f8c0c4068b0a')
    assert len(metadata['train_ids']) + len(metadata['validation_ids']) == 16
    assert set(metadata['train_ids']).isdisjoint(metadata['validation_ids'])
    assert len(metadata['topic_questions']) == 31 and metadata['use_label_descriptions']
    assert metadata['best_checkpoint'] and metadata['validation_metrics']['eval_micro.f1'] >= 0
    saved = load_file(output / ClefDecider.HEAD_WEIGHTS)
    best = load_file(Path(metadata['best_checkpoint']) / ClefDecider.HEAD_WEIGHTS)
    for key in saved:
        torch.testing.assert_close(saved[key], best[key])
    assert set(saved) == set(instances[0].model.head.state_dict())
    assert not any('backbone' in key or 'embedding' in key for key in saved)
    assert any(not torch.equal(saved[key], instances[0].original_head[key]) for key in saved)
    restored = Head()
    restored.load_state_dict(saved, strict=True)
    with pytest.raises(FileExistsError, match='not empty'):
        iptc_31.main(data_args, model_args, train_args)
    train_args.resume_from_checkpoint = metadata['best_checkpoint']
    previous_step = json.loads((output / 'trainer_state.json').read_text())['global_step']
    train_args.max_steps = previous_step + 2
    iptc_31.main(data_args, model_args, train_args)
    state = json.loads((output / 'trainer_state.json').read_text())
    assert state['global_step'] == previous_step + 2
    assert len(instances) == 2


def test_training_keeps_multiple_seeds_without_overwriting_existing_head(tmp_path, monkeypatch):
    paths, model_args, data_args, train_args, _ = setup_run(tmp_path, monkeypatch)
    iptc_31.main(data_args, model_args, train_args)
    first = Path(train_args.output_dir)
    original = {name: (first / name).read_bytes() for name in (ClefDecider.HEAD_WEIGHTS, ClefDecider.HEAD_MANIFEST)}
    another = replace(train_args, seed=2711, output_dir='trainer_output')
    iptc_31.main(data_args, model_args, another)
    second = Path(another.output_dir)
    assert second == paths.context / 'iptc-31.clef-flash-head.s2711' and second != first
    assert json.loads((second / ClefDecider.HEAD_MANIFEST).read_text())['seed'] == 2711
    for name, content in original.items():
        assert (first / name).read_bytes() == content


@pytest.mark.parametrize('automatic', [True, False])
def test_evaluation_reloads_trained_head_and_reports_shared_supervision(tmp_path, monkeypatch, automatic):
    paths, model_args, data_args, train_args, _ = setup_run(tmp_path, monkeypatch, seed=2711)
    iptc_31.main(data_args, model_args, train_args)
    output = Path(train_args.output_dir)
    eval_paths = entrypoint._ensure_dirs(tmp_path, 'eval', 'sequence', 'iptc-31')
    prepare = paths.get_script_ctx_path('data', 'prepare')
    for language in ('hr', 'sl'):
        (prepare / f'iptc-31.{language}.test.jsonl').write_text(json.dumps({
            'id': f'{language}:test', 'text': 'Test article', 'label': ['01000000'],
        }) + '\n')
        (prepare / f'iptc-31.{language}.labels.csv').write_text('id,name\n01000000,arts\n')
    monkeypatch.setattr(evaluate, 'paths', eval_paths, raising=False)
    monkeypatch.setattr(evaluate, 'logger', logging.getLogger(__name__), raising=False)
    parameters = data_args.attributes['sequence']['parameters']
    parameters['device'] = 'cpu'
    if not automatic:
        parameters['head_checkpoint'] = str(output)
        # An explicit path uses the checkpoint's seed for result naming.
        train_args.seed = 9999
    monkeypatch.setattr(ClefDecider, '_load_helper', staticmethod(lambda path: API))
    monkeypatch.setattr('huggingface_hub.snapshot_download', lambda *args, **kwargs: str(tmp_path / 'release'))
    loaded_models = []

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.head = Head()
            loaded_models.append(self)

        def forward(self, batch):
            hidden = torch.ones(len(batch['records']), 3, 4)
            return self.head(hidden, batch['input_ids'], batch['attention_mask'], batch['records'], None)

    monkeypatch.setattr(ClefDecider, '_load_text_model', staticmethod(lambda *args, **kwargs: (Model(), SimpleNamespace(pad_token_id=0))))
    evaluate.main(data_args, model_args, train_args)
    result_file = eval_paths.context / 'iptc-31.descriptions.clef-flash-head.s2711.clef.metrics.json'
    metadata = json.loads(result_file.read_text())
    trained = json.loads((output / ClefDecider.HEAD_MANIFEST).read_text())
    for key, value in load_file(output / ClefDecider.HEAD_WEIGHTS).items():
        torch.testing.assert_close(loaded_models[0].head.state_dict()[key], value)
    expected_probability = loaded_models[0].head.projection(torch.ones(4)).softmax(-1)[1].item()
    with np.load(result_file.with_name(result_file.name.replace('.metrics.json', '.predictions.npz'))) as predictions:
        np.testing.assert_allclose(predictions['y_score'], expected_probability, atol=1e-7)
    assert metadata['supervision'] == 'supervised_decision_head'
    assert metadata['parameters']['head_checkpoint'] == str(output)
    assert metadata['head_training']['seed'] == 2711
    assert metadata['num_train'] == len(trained['train_ids'])
    assert metadata['num_validation'] == len(trained['validation_ids']) and metadata['num_test'] == 2
    assert 'shared trained decision head' in metadata['aggregation']
    with pytest.raises(ValueError, match='original base model'):
        ClefDecider.read_head_manifest(output, 'wrong-model', trained['revision'])
    parameters['use_label_descriptions'] = False
    with pytest.raises(ValueError, match='same topic questions'):
        evaluate.main(data_args, model_args, train_args)
