import re

from logging import Logger
from pathlib import Path
from typing import Any, Dict

from transformers import TrainingArguments, pipeline

from ...app.args.model import ModelArguments
from ...app.args.data import DataArguments

logger: Logger
paths: Dict[str, Any]
WORD_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def compute_train_dir(m_args: ModelArguments, d_args: DataArguments, t_args: TrainingArguments) -> Path:
    model_name = f'{d_args.dataset_name}.{m_args.short_name}.b{t_args.train_batch_size}.lr{t_args.learning_rate}'
    output_dir = paths['base']['train'] / 'token' / model_name
    if not output_dir.exists():
        raise FileNotFoundError(output_dir)
    return output_dir


def reconstruct_entities(text: str, predictions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    entities: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    current_scores: list[float] = []

    for item in predictions:
        entity = item.get('entity', '')
        if not entity or entity == 'O':
            continue

        if '-' in entity:
            prefix, label = entity.split('-', 1)
        else:
            prefix, label = 'B', entity

        start = item.get('start')
        end = item.get('end')
        if not isinstance(start, int) or not isinstance(end, int):
            continue

        score = float(item.get('score', 0.0))
        should_start_new = (
            current is None
            or prefix == 'B'
            or current['entity_group'] != label
            or start > current['end']
        )

        if should_start_new:
            if current is not None:
                current['score'] = sum(current_scores) / len(current_scores)
                current['text'] = text[current['start']:current['end']]
                entities.append(current)
            current = {
                'entity_group': label,
                'start': start,
                'end': end,
            }
            current_scores = [score]
        else:
            current['end'] = end
            current_scores.append(score)

    if current is not None:
        current['score'] = sum(current_scores) / len(current_scores)
        current['text'] = text[current['start']:current['end']]
        entities.append(current)

    return entities


# noinspection DuplicatedCode
def main(data_args: DataArguments, model_args: ModelArguments, train_args: TrainingArguments) -> None:
    logger.info('Testing NER')

    train_args.output_dir = str(compute_train_dir(model_args, data_args, train_args))
    tokenizer_name = model_args.tokenizer_name or model_args.model_name_or_path
    ner = pipeline(
        task="token-classification",
        model=train_args.output_dir,
        tokenizer=tokenizer_name,
        aggregation_strategy="first",
    )
    text = " Janez Novak... Metka Kralj,,. in Boris A. Novak živijo v Ljubljani in delajo za Microsoft."
    #raw_result = ner(text)
    #for item in raw_result:
    #    start = item.get('start')
    #    end = item.get('end')
    #    item['text'] = text[start:end] if isinstance(start, int) and isinstance(end, int) else ''
    #print(raw_result)
    #result = reconstruct_entities(text, raw_result)
    #print(result)

    tokens = re.findall(r"\s+|\w+|[^\w\s]", text, flags=re.UNICODE)
    result = ner(tokens, is_split_into_words=True, delimiter="")

    prev = None
    rewritten = []
    for e in result[0]:
        if prev is not None and prev['end'] == e['start']:
            #rewritten[-1]['entity'] = e['entity'][2:]
            rewritten[-1]['span'] = text[rewritten[-1]['start']:e['end']]
            rewritten[-1]['end'] += e['end']
        else:
            rewritten.append(e.copy())
            rewritten[-1].pop('word')
            #rewritten[-1]['entity'] = e['entity'][2:]
            rewritten[-1]['span'] = text[e['start']:e['end']]
        prev = e
    logger.info('Text:')
    print(text)
    logger.info('Text tokens:')
    print(tokens)
    logger.info('Pipeline result:')
    print(result)
    logger.info('Rewritten result:')
    print(rewritten)

