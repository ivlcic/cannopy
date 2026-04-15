import os
import shutil
import zipfile
from datetime import datetime
from logging import Logger
from pathlib import Path
from typing import Optional

from transformers import TrainingArguments

from ...app.args.data import DataArguments
from ...app.args.model import ModelArguments
from ...app.args.runtime import Paths
from ...app.token_classifier import EncoderTokenClassifier

logger: Logger
paths: Paths


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


def compute_train_dir(m_args: ModelArguments, t_args: TrainingArguments) -> Optional[str]:
    model_name = f'ner.{m_args.short_name}.b{t_args.train_batch_size}.lr{t_args.learning_rate}'
    output_dir = paths.get_script_path('train') / model_name
    if not output_dir.exists():
        return None
    return str(output_dir)


def compute_output(m_args: ModelArguments, t_args: TrainingArguments) -> Path:
    model_name = f'ner.{m_args.short_name}.b{t_args.train_batch_size}.lr{t_args.learning_rate}'
    output = paths.context / model_name
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    return output


# noinspection DuplicatedCode
def main(data_args: DataArguments, model_args: ModelArguments, train_args: TrainingArguments) -> None:
    logger.info('Evaluating NER')

    train_args.output_dir = compute_train_dir(model_args, train_args)
    output_dir = compute_output(model_args, train_args)

    sub_dir = data_args.attributes.get('use_subdir', 'sample_reference')
    input_dir = paths.get_script_ctx_path('data', 'download') / sub_dir

    model_name = train_args.output_dir or model_args.model_name_or_path
    tagger = EncoderTokenClassifier(model_name, model_args)

    # Process each CoNLL file in the input directory
    for filename in os.listdir(input_dir):
        if not filename.endswith('.conll2002'):
            continue

        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)

        sentences = load_from_file(input_path)
        text_labels = []
        for sentence in sentences:
            labels = tagger.classify_tokens(sentence)
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
