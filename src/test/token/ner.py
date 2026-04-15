import re
from logging import Logger
from pathlib import Path

from transformers import TrainingArguments, pipeline

from ...app.args.runtime import Paths
from ...app.args.data import DataArguments
from ...app.args.model import ModelArguments

logger: Logger
paths: Paths
WORD_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def compute_train_dir(m_args: ModelArguments, d_args: DataArguments, t_args: TrainingArguments) -> Path:
    model_name = f'{d_args.dataset_name}.{m_args.short_name}.b{t_args.train_batch_size}.lr{t_args.learning_rate}'
    output_dir = paths.get_script_path('train') / model_name
    if not output_dir.exists():
        raise FileNotFoundError(output_dir)
    return output_dir


# noinspection DuplicatedCode
def main(data_args: DataArguments, model_args: ModelArguments, train_args: TrainingArguments) -> None:
    logger.info('Testing NER')

    train_args.output_dir = str(compute_train_dir(model_args, data_args, train_args))
    tokenizer_name = model_args.tokenizer_name or model_args.model_name_or_path
    ner = pipeline(
        task="token-classification",
        model=train_args.output_dir,
        tokenizer=tokenizer_name,
        aggregation_strategy="simple",
    )
    text = " Janez Novak... Metka Kralj,,. in Boris A. Novak živijo v Ljubljani in delajo za Microsoft."
    tokens = re.findall(r"\s+|\w+|[^\w\s]", text, flags=re.UNICODE)
    result = ner(tokens, is_split_into_words=True, delimiter="")

    logger.info('Text:')
    print(text)
    logger.info('Text tokens:')
    print(tokens)
    logger.info('Pipeline split to words result:')
    for r in result[0]:
        # noinspection PyStringFormat
        print(f'[{text[r['start']:r['end']]}]({r["entity_group"]}@{"%.2f"%r["score"]}|{r["start"]}:{r["end"]})')
        r['text'] = text[r['start']:r['end']]
    print(result)
    result = ner(text)
    logger.info('Pipeline result:')
    for r in result:
        print(f'[{text[r['start']:r['end']]}]({r["entity_group"]}@{r["start"]}:{r["end"]})')
        r['text'] = text[r['start']:r['end']]
    print(result)
