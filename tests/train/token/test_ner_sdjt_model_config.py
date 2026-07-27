from types import SimpleNamespace

import pytest
from transformers import DebertaV2Config, ModernBertConfig, XLMRobertaConfig

from src.train.token.ner_sdjt import (
    compute_model_prefix,
    configure_classifier_dropout,
)


def test_compute_model_prefix_includes_optimizer_hyperparameters():
    model_args = SimpleNamespace(short_name="mm-bert", classifier_dropout=0.10)
    data_args = SimpleNamespace(dataset_name="ner-sdjt")
    train_args = SimpleNamespace(
        per_device_train_batch_size=16,
        learning_rate=5e-6,
        warmup_ratio=0.06,
        weight_decay=0.01,
    )

    prefix = compute_model_prefix(model_args, data_args, train_args)

    assert prefix == "ner-sdjt.mm-bert.b16.lr5e-06.cd01.wr06.wd01"


@pytest.mark.parametrize(
    ("config", "attribute"),
    (
        (ModernBertConfig(), "classifier_dropout"),
        (XLMRobertaConfig(), "classifier_dropout"),
        (DebertaV2Config(), "hidden_dropout_prob"),
    ),
)
def test_configure_classifier_dropout_uses_token_classifier_setting(config, attribute):
    configure_classifier_dropout(config, 0.10)

    assert getattr(config, attribute) == 0.10
