from dataclasses import field, dataclass
from datetime import timedelta
from logging import Logger
from typing import Dict, Any, Tuple

from ...app.args.data import DataArguments
from ...app.args.runtime import Paths
from ...app.elastic import ElasticQuery, ElasticWriter, ElasticArticleSanitizer
from ...app.iterators import DateTimeIterator, DateTimeState, RuntimeData

logger: Logger
paths: Paths


@dataclass
class IptcRuntimeData(RuntimeData):
    tag_map: Dict[str, str] = field(default_factory=dict)
    iptc_map: Dict[str, str] = field(default_factory=dict)


# noinspection DuplicatedCode,PyGlobalUndefined
def write(state: DateTimeState):
    # noinspection PyUnresolvedReferences
    data_args: DataArguments = state.data_args
    # noinspection PyUnresolvedReferences
    runtime_data: IptcRuntimeData = state.runtime_data
    data_create_path = paths.context  # paths['create']['data']
    file_name = data_args.dataset_name + f'-{runtime_data.file_num:02d}'
    ElasticWriter.write_to_file(
        runtime_data.items,
        data_create_path,
        file_name
    )
    logger.info('Writing data to %s', data_create_path)
    runtime_data.file_num += 1
    runtime_data.items = []


def init_item(result, state: DateTimeState) -> Tuple[Dict[str, Any], Dict[str, str], Any]:
    # noinspection PyUnresolvedReferences
    runtime_data: IptcRuntimeData = state.runtime_data
    item = ElasticArticleSanitizer.sanitize_es_result(result)
    if item is None:
        return {}, {}, None

    iptc_tags = {}
    if 'tags' not in result:
        return {}, {}, None

    for tag in result['tags']:
        if tag['uuid'] not in runtime_data.tag_map.keys():
            continue
        iptc_id = runtime_data.iptc_map[tag['uuid']]
        iptc_tags[iptc_id] = runtime_data.tag_map[tag['uuid']]

    body = item['body']
    title = item['title']
    if not body.startswith(title):
        text = title + '\n\n' + body
    else:
        text = body

    return item, iptc_tags, text


# noinspection DuplicatedCode
def load_data(state: DateTimeState):
    # noinspection PyUnresolvedReferences
    data_args: DataArguments = state.data_args
    # noinspection PyUnresolvedReferences
    runtime_data: IptcRuntimeData = state.runtime_data
    req = ElasticQuery(data_args.source.conn.url, data_args.source.conn.username)
    query_desc: Dict[str, Any] = data_args.source.select.query

    items_batch = {}
    query_template = query_desc['template']
    results, total = req.query(query_template, state.step_start, state.step_end)
    for result in results:
        item, iptc_tags, text = init_item(result, state)
        if not item or not iptc_tags:
            continue
        item['iptc_tags'] = iptc_tags
        items_batch[result['uuid']] = item

    for k, item in items_batch.items():
        runtime_data.items.append(item)
        if runtime_data.num_items_per_file == len(runtime_data.items):
            write(state)


def parse_config(runtime: IptcRuntimeData, data_args: DataArguments):
    query_desc: Dict[str, Any] = data_args.source.select.query
    runtime.tag_map = query_desc['tag_map']
    runtime.iptc_map = query_desc['iptc_map']


# noinspection DuplicatedCode
def main(data_args: DataArguments) -> None:
    runtime = IptcRuntimeData()
    parse_config(runtime, data_args)

    logger.info(f'Downloading {data_args.dataset_name}')
    state = None
    for state in DateTimeIterator(
        start=data_args.source.select.start,
        end=data_args.source.select.end,
        step=timedelta(days=10),
        callback=load_data,
        data_args=data_args,
        runtime_data=runtime
    ):
        logger.info(f'Processing {state.progress:.2f} @ step [{state.step_start} <=> {state.step_end}] / {state.end}')
    if state:
        write(state)
