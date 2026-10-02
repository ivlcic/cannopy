from pathlib import Path

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from src.app.entrypoint import _load_and_merge_configs
from src.app.ner import NerSample
from src.app.ner_decoder import (
    CausalNerCollator,
    NerGenerationDataset,
    format_generation_answer,
    format_generation_prompt,
    generate_sentence_labels,
    parse_generation,
    split_generation_tokens,
)
from src.eval.token import ner_dec as eval_ner_dec
from src.test.token import ner_dec as test_ner_dec
from src.train.token import ner_dec as train_ner_dec


class ToyTokenizer:
    eos_token_id = 2
    pad_token_id = 0
    padding_side = "left"

    def __init__(self):
        self.texts = []

    def __call__(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        self.texts.append(text)
        return {"input_ids": [3] * len(text.split())}


class ToyChatTokenizer(ToyTokenizer):
    chat_template = "user -> assistant"

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        assert tokenize and add_generation_prompt
        assert messages[0]["role"] == "user"
        self.texts.append(messages[0]["content"])
        return [9] + [3] * len(messages[0]["content"].split()) + [10, 11]


TOKENS = ["Janez", "Novak", "živi", "v", "Ljubljani", "."]
LABELS = ["B-PER", "I-PER", "O", "O", "B-LOC", "O"]
ANSWER = (
    "Janez      B-PER\n"
    "Novak      I-PER\n"
    "živi       O\n"
    "v          O\n"
    "Ljubljani  B-LOC\n"
    ".          O"
)


def test_bio_rows_match_requested_example():
    assert format_generation_prompt(TOKENS) == "BIO-NER:\nJanez Novak živi v Ljubljani.\n"
    assert format_generation_answer(TOKENS, LABELS) == ANSWER


def test_chat_training_masks_prompt_and_supervises_rows_and_eos():
    tokenizer = ToyChatTokenizer()
    dataset = NerGenerationDataset(tokenizer, 128, [NerSample(TOKENS, LABELS)], True)

    item = dataset[0]

    assert len(dataset) == 1
    assert tokenizer.texts[-2:] == [format_generation_prompt(TOKENS), ANSWER]
    prompt_length = 1 + len(format_generation_prompt(TOKENS).split()) + 2
    answer_length = len(ANSWER.split())
    assert item["labels"] == [-100] * prompt_length + [3] * answer_length + [2]
    assert item["input_ids"][-1] == 2


def test_base_training_puts_rows_directly_after_sentence_newline():
    tokenizer = ToyTokenizer()
    item = NerGenerationDataset(tokenizer, 128, [NerSample(TOKENS, LABELS)], False)[0]

    assert tokenizer.texts[-2:] == [format_generation_prompt(TOKENS), ANSWER]
    assert item["labels"].count(-100) == len(format_generation_prompt(TOKENS).split())
    assert item["labels"][-1] == tokenizer.eos_token_id


def test_generated_rows_require_exact_tokens_labels_and_count():
    valid = {"O", "B-PER", "I-PER", "B-LOC"}

    assert parse_generation(ANSWER, TOKENS, valid) == LABELS
    assert parse_generation(ANSWER.replace("Novak", "Marko"), TOKENS, valid) is None
    assert parse_generation(ANSWER.replace(".          O", ".          o"), TOKENS, valid) is None
    assert parse_generation("\n".join(ANSWER.splitlines()[:-1]), TOKENS, valid) is None


def test_long_sentence_chunks_and_reassembles_rows(monkeypatch):
    tokenizer = ToyTokenizer()
    tokens = [f"word{i}" for i in range(12)]
    sample = NerSample(tokens, ["O"] * len(tokens))
    chunks = split_generation_tokens(tokenizer, tokens, 40, False)
    dataset = NerGenerationDataset(tokenizer, 40, [sample], False)

    assert len(chunks) > 1
    assert chunks[0][0] == 0 and chunks[-1][1] == len(tokens)
    assert all(chunks[i][1] == chunks[i + 1][0] for i in range(len(chunks) - 1))
    assert [chunk.tokens for chunk in dataset.samples] == [tokens[a:b] for a, b in chunks]
    assert all(len(dataset[i]["input_ids"]) <= 40 for i in range(len(dataset)))

    calls = []

    def fake_generate(model, tokenizer, chunk, max_seq_length, use_chat_template, valid_labels):
        calls.append(list(chunk))
        return ["O"] * len(chunk), format_generation_answer(chunk, ["O"] * len(chunk))

    monkeypatch.setattr("src.app.ner_decoder.generate_labels", fake_generate)
    labels, _ = generate_sentence_labels(None, tokenizer, tokens, 40, False, {"O"})

    assert labels == ["O"] * len(tokens)
    assert calls == [tokens[a:b] for a, b in chunks]


def test_causal_collator_masks_left_padding():
    batch = CausalNerCollator(ToyTokenizer())([
        {"input_ids": [9, 3, 2], "attention_mask": [1, 1, 1], "labels": [-100, 3, 2]},
        {"input_ids": [9, 2], "attention_mask": [1, 1], "labels": [-100, 2]},
    ])

    assert batch["input_ids"].tolist() == [[9, 3, 2], [0, 9, 2]]
    assert batch["attention_mask"].tolist() == [[1, 1, 1], [0, 1, 1]]
    assert batch["labels"].tolist() == [[-100, 3, 2], [-100, -100, 2]]


def test_qwen_causal_head_trains_on_rows_only():
    config = Qwen3Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=64,
        eos_token_id=2,
        pad_token_id=0,
    )
    model = Qwen3ForCausalLM(config)
    labels = torch.tensor([[-100, -100, -100, 7, 8, 2]])

    output = model(input_ids=torch.tensor([[9, 10, 11, 7, 8, 2]]), labels=labels)
    output.loss.backward()

    assert output.logits.shape == (1, 6, 64)
    assert torch.isfinite(output.loss)
    assert model.lm_head.weight.grad is not None


def test_decoder_configs_only_offer_generative_models():
    conf_dir = Path(__file__).resolve().parents[3] / "conf"
    instruct, _ = _load_and_merge_configs(
        conf_dir, "token", "ner-dec", ["qwen3-4b-instruct-2507"]
    )
    base, _ = _load_and_merge_configs(conf_dir, "token", "ner-dec", ["qwen3-4b-base"])

    assert instruct["data"]["dataset_name"] == "ner-dec"
    assert instruct["model"]["use_chat_template"] is True
    assert instruct["train"]["metric_for_best_model"] == "eval_loss"
    assert base["model"].get("use_chat_template", False) is False
    assert not (conf_dir / "data" / "ner-dec" / "qwen3-ebd-4b.yaml").exists()
    assert callable(train_ner_dec.main)
    assert callable(eval_ner_dec.main)
    assert callable(test_ner_dec.main)
