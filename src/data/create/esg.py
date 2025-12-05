import re
from dataclasses import field, dataclass
from datetime import timedelta
from typing import Dict, Any, List, Tuple, Set

from ...app.args.data import DataArguments
from ...app.elastic import ElasticQuery, ElasticWriter, ElasticArticleSanitizer
from ...app.iterators import DateTimeIterator, DateTimeState, RuntimeData


@dataclass
class CategoryKeywords:
    name: str
    keywords: List[str] = field(default_factory=list)


@dataclass
class CategoryExpressions:
    name: str
    expressions: List[str] = field(default_factory=list)
    tokens: List[List[str]] = field(default_factory=list)


@dataclass
class EsgRuntimeData(RuntimeData):
    category_keywords: List[CategoryKeywords] = field(default_factory=list)
    category_expressions: List[CategoryExpressions] = field(default_factory=list)
    industries_map: Dict[str, str] = field(default_factory=dict)
    valid_industry_uuids: Set[str] = field(default_factory=set)
    keyword_should: str = ''
    expression_should: str = ''


def return_keyword_matches(text: str, category_keywords: List[CategoryKeywords]) -> Tuple[List[str], List[str]]:
    matches = []
    categories = []
    for category in category_keywords:
        for keyword in category.keywords:
            flags = 0
            if keyword.lower():
                flags = re.IGNORECASE
            if keyword.endswith('*'):
                keyword = keyword[:-1]
                pat = re.compile(rf'\b{re.escape(keyword)}\w*\b', flags=flags)
            else:
                pat = re.compile(rf'\b{re.escape(keyword)}\b', flags=flags)
            if pat.search(text):
                matches.append(keyword)
                if category.name not in categories:
                    categories.append(category.name)

    return matches, categories


def return_expression_matches(text: str, category_expressions: List[CategoryExpressions]) \
        -> Tuple[List[str], List[str]]:
    matches = []
    categories = []
    for category in category_expressions:
        for idx, expression in enumerate(category.expressions):
            expr_tokens = category.tokens[idx]
            token_regex = ''
            all_lowercase = True
            for token in expr_tokens:
                if not token.islower():
                    all_lowercase = False
                if token_regex:
                    token_regex += '\s+'
                if token.endswith('*'):
                    token = token[:-1]
                    token_regex += rf'\b{re.escape(token)}\w*\b'
                else:
                    token_regex += rf'\b{re.escape(token)}\b'
            flags = 0
            if all_lowercase:
                flags = re.IGNORECASE
            pat = re.compile(token_regex, flags=flags)
            if pat.search(text):
                matches.append(expression)
                if category.name not in categories:
                    categories.append(category.name)

    return matches, categories


# noinspection PyUnresolvedReferences,DuplicatedCode,PyGlobalUndefined
def write(state: DateTimeState):
    global paths
    data_create_path = paths['create']['data']
    file_name = state.data_args.dataset_name + f'-{state.runtime_data.file_num:02d}'
    ElasticWriter.write_to_file(
        state.runtime_data.items,
        data_create_path,
        file_name
    )
    logger.info('Writing data to %s', data_create_path)
    state.runtime_data.file_num += 1
    state.runtime_data.items = []


# noinspection PyUnresolvedReferences
def init_item(result, state: DateTimeState) -> Tuple[Dict[str, Any], Dict[str, str], Any]:
    item = ElasticArticleSanitizer.sanitize_es_result(result)
    if item is None:
        return {}, {}, None

    industries = {}
    if 'tags' not in result:
        return {}, {}, None

    for tag in result['tags']:
        if 'class' not in tag or not tag['class'].endswith('.CustomerTopicGroup'):
            continue
        if tag['uuid'] not in state.runtime_data.industries_map.keys():
            continue
        industries[tag['uuid']] = state.runtime_data.industries_map[tag['uuid']]

    body = item['body']
    title = item['title']
    if not body.startswith(title):
        text = title + '\n\n' + body
    else:
        text = body

    return item, industries, text


# noinspection PyUnresolvedReferences,DuplicatedCode
def load_data(state: DateTimeState):
    req = ElasticQuery(state.data_args.dataset_src_url, state.data_args.dataset_src_user)
    query_desc: Dict[str, Any] = state.data_args.dataset_src_query

    items_batch = {}
    query_template = query_desc['template']
    if state.runtime_data.keyword_should:
        keyword_query = query_template.replace('<should_match>', state.runtime_data.keyword_should)
        results, total = req.query(keyword_query, state.step_start, state.step_end)
        for result in results:
            item, industries, text = init_item(result, state)
            if not item:
                continue
            matches, categories = return_keyword_matches(text, state.runtime_data.category_keywords)
            if not categories:
                continue

            item['monitored'] = []
            item['monitored_uuid'] = []
            for k, v in industries.items():
                item['monitored'].append(v)
                item['monitored_uuid'].append(k)
            item['matched'] = categories
            item['matches'] = matches
            items_batch[result['uuid']] = item

    if state.runtime_data.expression_should:
        expression_query = query_template.replace('<should_match>', state.runtime_data.expression_should)
        results, total = req.query(expression_query, state.step_start, state.step_end)
        for result in results:
            item, industries, text = init_item(result, state)
            if not item:
                continue
            matches, categories = return_expression_matches(text, state.runtime_data.category_expressions)
            if not categories:
                continue

            item['monitored'] = []
            item['monitored_uuid'] = []
            for k, v in industries.items():
                item['monitored'].append(v)
                item['monitored_uuid'].append(k)
            item['matched'] = categories
            item['matches'] = matches

            if result['uuid'] in items_batch:
                monitored_uuid = set(items_batch[result['uuid']]['monitored_uuid'])
                for k, v in industries.items():
                    if k not in monitored_uuid:
                        items_batch[result['uuid']]['monitored'].append(v)
                        items_batch[result['uuid']]['monitored_uuid'].append(k)

                for c in categories:
                    if c not in items_batch[result['uuid']]['matched']:
                        items_batch[result['uuid']]['matched'].append(categories)

                items_batch[result['uuid']]['matches'].extend(matches)
                continue
            items_batch[result['uuid']] = item

    for k, item in items_batch.items():
        state.runtime_data.items.append(item)
        if state.runtime_data.num_items_per_file == len(state.runtime_data.items):
            write(state)


def parse_config(runtime: EsgRuntimeData, data_args: DataArguments):
    query_desc: Dict[str, Any] = data_args.dataset_src_query

    runtime.industries_map = query_desc['industry_map']
    industry_matches = query_desc['industry_match']
    runtime.keyword_should = ''
    runtime.expression_should = ''
    for category_name, industry_match in industry_matches.items():
        if 'keywords' in industry_match:
            category_keywords = CategoryKeywords(category_name, industry_match['keywords'])
            runtime.category_keywords.append(category_keywords)
            for keyword in category_keywords.keywords:
                if runtime.keyword_should:
                    runtime.keyword_should += ',\n'
                if keyword.endswith('*'):
                    runtime.keyword_should += '{"prefix": { "text": "' + keyword[:-1] + '"}}'
                else:
                    runtime.keyword_should += '{"match": { "text": "' + keyword + '"}}'
        if 'expressions' in industry_match:
            category_expression = CategoryExpressions(category_name, industry_match['expressions'])
            runtime.category_expressions.append(category_expression)
            for expression in category_expression.expressions:
                tokens = expression.split()
                category_expression.tokens.append(tokens)
                span_near = ''
                for token in tokens:
                    if span_near:
                        span_near += ',\n'
                    if token.endswith('*'):
                        span_near += '{"span_multi": {"match": {"prefix": {"text": {"value": "' + token[:-1] + '"}}}}}'
                    else:
                        span_near += '{"span_multi": {"match": {"prefix": {"text": {"value": "' + token + '"}}}}}'
                if span_near:
                    if runtime.expression_should:
                        runtime.expression_should += ',\n'
                    runtime.expression_should += (
                            '{"span_near": {"slop": 0, "in_order": true, "clauses": [\n' + span_near + '\n]}}')


# noinspection PyUnresolvedReferences,DuplicatedCode,PyGlobalUndefined
def main(data_args: DataArguments) -> None:
    global logger, paths

    runtime = EsgRuntimeData()
    parse_config(runtime, data_args)

    logger.info(f'Downloading {data_args.dataset_name}')
    state = None
    for state in DateTimeIterator(
        start=data_args.dataset_src_start,
        end=data_args.dataset_src_end,
        step=timedelta(days=10),
        callback=load_data,
        data_args=data_args,
        runtime_data=runtime
    ):
        logger.info(f'Processing {state.progress:.2f} @ step [{state.step_start} <=> {state.step_end}] / {state.end}')
    if state:
        write(state)
