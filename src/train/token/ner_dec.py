"""Train token/BIO rows per sentence with a causal language model."""

from logging import Logger
from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

from ...app.args.data import DataArguments
from ...app.args.model import ModelArguments
from ...app.args.runtime import Paths
from ...app.dataset import NerSamplesLoader
from ...app.ner_decoder import CausalNerCollator, NerGenerationDataset

logger: Logger
paths: Paths


def init_dirs(p: Paths) -> tuple[Path, Path]:
    data_root = p.base.result.data / "split" / "ner"
    if not data_root.exists():
        raise FileNotFoundError(f"NER split data not found at {data_root}; run ./data split ner first")
    cache_root = p.base.tmp / "cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    return data_root, cache_root


def compute_run_name(model_args: ModelArguments, data_args: DataArguments,
                     train_args: TrainingArguments) -> str:
    return (
        f"{data_args.dataset_name}.{model_args.short_name}.bio-rows."
        f"b{train_args.per_device_train_batch_size}.lr{train_args.learning_rate:.12g}"
    )


def load_samples(data_root: Path, languages: list[str]) -> NerSamplesLoader:
    if not languages:
        languages = sorted({p.name.split(".")[0][4:] for p in data_root.glob("ner-*.train.csv")})
    return NerSamplesLoader(data_root, languages)


def build_generation_dataset(loader, split: str, tokenizer, max_seq_length: int,
                             use_chat_template: bool) -> NerGenerationDataset:
    samples = [
        sample
        for language_samples in loader.samples_by_lang[split].values()
        for sample in language_samples
    ]
    return NerGenerationDataset(tokenizer, max_seq_length, samples, use_chat_template)


def main(data_args: DataArguments, model_args: ModelArguments,
         train_args: TrainingArguments) -> None:
    data_root, cache_root = init_dirs(paths)
    train_args.output_dir = str(paths.context / compute_run_name(model_args, data_args, train_args))
    loader = load_samples(data_root, data_args.subdata_order)
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name or model_args.model_name_or_path,
        cache_dir=cache_root,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=cache_root,
        dtype=model_args.dtype,
        **({"attn_implementation": model_args.attn_implementation}
           if model_args.attn_implementation else {}),
    )
    model.config.ner_labels = sorted(loader.labeler.label2id)
    model_args.validate_training_parameter_dtypes(model)
    train_dataset = build_generation_dataset(
        loader, "train", tokenizer, model_args.max_seq_length, model_args.use_chat_template,
    )
    eval_dataset = build_generation_dataset(
        loader, "eval", tokenizer, model_args.max_seq_length, model_args.use_chat_template,
    )
    logger.info(
        "Training generative NER on %d chunks; evaluating on %d from %s",
        len(train_dataset), len(eval_dataset), data_root,
    )
    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=CausalNerCollator(tokenizer),
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(train_args.output_dir)
    trainer.state.save_to_json(str(Path(train_args.output_dir) / "trainer_state.json"))
    logger.info("Teacher-forced validation: %s", trainer.evaluate())
