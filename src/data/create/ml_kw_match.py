import re
from datetime import timedelta
from logging import Logger
from typing import Dict, Any

from ...app.args.runtime import Paths
from ...app.args.data import DataArguments
from ...app.elastic import ElasticQuery, ElasticWriter, ElasticArticleSanitizer
from ...app.iterators import DateTimeIterator, DateTimeState, RuntimeData

logger: Logger
paths: Paths


def contains_any(text: str, keywords: list[str]) -> bool:
    for kw in keywords:
        if kw.islower():
            # Case-insensitive check for fully lowercase keywords
            pat = re.compile(rf"\b{re.escape(kw)}\b", flags=re.IGNORECASE)
            if pat.search(text):
                return True
        else:
            # Case-sensitive for any keyword with non-lowercase chars
            pat = re.compile(rf"\b{re.escape(kw)}\b")
            if pat.search(text):
                return True
    return False


# noinspection DuplicatedCode
def write(state: DateTimeState):
    # noinspection PyUnresolvedReferences
    data_args: DataArguments = state.data_args
    # noinspection PyUnresolvedReferences
    runtime_data: RuntimeData = state.runtime_data
    data_create_path = paths.context  # paths['create']['data']
    file_name = data_args.dataset_name + f"-{runtime_data.file_num:02d}"
    ElasticWriter.write_to_file(
        runtime_data.items,
        data_create_path,
        file_name
    )
    logger.info("Writing data to %s", data_create_path)
    runtime_data.file_num += 1
    runtime_data.items = []


def load_data(state: DateTimeState):
    # noinspection PyUnresolvedReferences
    data_args: DataArguments = state.data_args
    # noinspection PyUnresolvedReferences
    runtime_data: RuntimeData = state.runtime_data
    req = ElasticQuery(data_args.source.conn.url, data_args.source.conn.username)
    query_desc: Dict[str, Any] = data_args.source.select.query
    items_batch = {}
    for category, keywords in query_desc['keywords'].items():
        query = query_desc['template']
        keywords_str = ",\n".join(f'{{ "match_phrase": {{ "text": "{item}" }} }}' for item in keywords)
        query = query.replace('<should_match>', keywords_str)
        results, total = req.query(query, state.step_start, state.step_end)
        for result in results:
            item = ElasticArticleSanitizer.sanitize_es_result(result, {'categories': [category]})
            if item is None:
                continue
            body = item['body']
            title = item['title']
            if not body.startswith(title):
                text = title + "\n\n" + body
            else:
                text = body

            # validate match (case-sensitive words)
            if not contains_any(text, keywords):
                continue

            # we already hit the other category with the same item
            if result['uuid'] in items_batch:
                items_batch[result['uuid']]['categories'].append(category)
                continue

            # first time hit item
            items_batch[result['uuid']] = item

    for k, item in items_batch.items():
        runtime_data.items.append(item)
        if runtime_data.num_items_per_file == len(runtime_data.items):
            write(state)


# noinspection DuplicatedCode
def main(data_args: DataArguments) -> None:
    logger.info(f"Downloading {data_args.dataset_name}")
    runtime = RuntimeData(num_items_per_file=50000)
    state = None
    for state in DateTimeIterator(
            start=data_args.source.select.start,
            end=data_args.source.select.end,
            step=timedelta(days=10),
            callback=load_data,
            data_args=data_args,
            runtime_data=runtime
    ):
        logger.info(f"Processed {state.progress:.2f} @ step [{state.step_start} <=> {state.step_end}] / {state.end}")
    if state:
        write(state)
