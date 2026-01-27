import os
import shutil

import torch
import zipfile

from datetime import datetime
from logging import Logger
from pathlib import Path
from typing import Any, Dict, Optional

from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    TrainingArguments,
)

from ...app.args.model import ModelArguments
from ...app.args.data import DataArguments

logger: Logger
paths: Dict[str, Any]


def load_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        conll_text = file.read()
        sentences = [sentence.split('\n') for sentence in conll_text.strip().split('\n\n')]
        if file_path.startswith('sample_reference'):  # remove NER tag for sample_reference files to test
            new_sentences = []
            for sentence in sentences:
                new_sentence = []
                for token in sentence:
                    new_sentence.append(token.split(None, 1)[0])
                new_sentences.append(new_sentence)
            sentences = new_sentences

        return sentences


def align_subwords_to_words(labels, word_ids):
    aligned_labels = []
    current_word = None
    current_label = None

    for label, word_id in zip(labels, word_ids):
        if word_id is None:  # Special tokens like [CLS] and [SEP]
            continue

        if word_id != current_word:
            # New word
            if current_word is not None:
                aligned_labels.append(current_label)
            current_word = word_id
            current_label = label

    # Add the last word
    if current_word is not None:
        aligned_labels.append(current_label)

    return aligned_labels


def compute_train_dir(m_args: ModelArguments, t_args: TrainingArguments) -> Optional[str]:
    model_name = f'ner.{m_args.short_name}.b{t_args.train_batch_size}.lr{t_args.learning_rate}'
    output_dir = paths['base']['train'] / 'token' / model_name
    if not output_dir.exists():
        return None
    return str(output_dir)


def compute_output(m_args: ModelArguments, d_args: DataArguments, t_args: TrainingArguments) -> Path:
    model_name = f'ner.{m_args.short_name}.b{t_args.train_batch_size}.lr{t_args.learning_rate}'
    output = paths['ner-slobench']['eval'] / model_name
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    return output


# noinspection DuplicatedCode
def main(data_args: DataArguments, model_args: ModelArguments, train_args: TrainingArguments) -> None:
    logger.info('Evaluating NER')

    train_args.output_dir = compute_train_dir(model_args, train_args)
    output_dir = compute_output(model_args, data_args, train_args)

    sub_dir = data_args.attributes.get('use_subdir', 'sample_reference')
    input_dir = paths['base']['data'] / 'download' / data_args.dataset_name / sub_dir

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_kwargs = {}
    autocast = False
    if device == 'cuda':
        if model_args.attn_implementation:
            model_kwargs['attn_implementation'] = model_args.attn_implementation
        if model_args.dtype:
            autocast = True
            model_kwargs['dtype'] = getattr(torch, model_args.dtype)

    tokenizer_name = model_args.tokenizer_name or model_args.model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name
    )
    model_name = train_args.output_dir or model_args.model_name_or_path
    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        **model_kwargs,
    )
    model.to(device)

    # Process each CoNLL file in the input directory
    for filename in os.listdir(input_dir):
        if not filename.endswith('.conll2002'):
            continue

        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)

        sentences = load_from_file(input_path)
        text_labels = []
        model.eval()
        for sentence in sentences:
            inputs = tokenizer(
                sentence, is_split_into_words=True, return_tensors="pt", padding=True, truncation=True
            )
            inputs = inputs.to(device)
            with torch.no_grad():
                if autocast:
                    with torch.autocast(device_type='cuda', dtype=model_kwargs['dtype']):
                        outputs = model(**inputs)
                else:
                    outputs = model(**inputs)

            # Get the predicted labels
            predictions = outputs.logits.argmax(dim=-1).squeeze().tolist()
            labels = [model.config.id2label[pred] for pred in predictions]
            word_ids = inputs.word_ids()[1:-1]  # Exclude [CLS] and [SEP] tokens
            labels = align_subwords_to_words(labels[1:-1], word_ids)
            text_labels.append(labels)

        # Write the updated CoNLL data to the output file
        with open(output_path, 'w', encoding='utf-8') as f:
            for sentence, labels in zip(sentences, text_labels):
                for token, label in zip(sentence, labels):
                    f.write(f'{token} {label}\n')
                f.write('\n')

    # zip output files
    timestamp = datetime.now().strftime('%Y-%m-%dT%H-%M')
    zip_path = output_dir / f'submission-{timestamp}.zip'
    with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as zipf:
        for filename in os.listdir(output_dir):
            if filename.endswith('.conll2002'):
                file_path = output_dir / filename
                zipf.write(file_path, arcname=filename)

    logger.info(f'Classified tokens from {input_dir} to {output_dir}')
