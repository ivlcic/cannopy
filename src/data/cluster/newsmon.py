import json

import numpy as np
import networkx as nx

from collections import defaultdict
from datetime import datetime, timedelta, timezone
from logging import Logger
from typing import Any, Dict, List

from dateutil.relativedelta import relativedelta

from ..embed.newsmon import load_embeddings
from ...app.args.data import DataArguments
from ...app.args.model import ModelArguments

logger: Logger
paths: Dict[str, Any]


def cosine_similarity_matrix(X, eps=1e-12):
    X = np.asarray(X, dtype=float)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    X = X / np.clip(norms, eps, None)
    return X @ X.T


def cluster_louvain(articles: List[Dict[str, Any]], embed_field_name: str, sim_threshold: float = 0.84, seed=None):
    embeddings = []
    [embeddings.append(x[embed_field_name]) for x in articles]
    embeddings = np.array(embeddings)
    x = cosine_similarity_matrix(embeddings)
    similarity_matrix = x > sim_threshold
    G = nx.from_numpy_array(similarity_matrix)
    communities = nx.algorithms.community.louvain_communities(G, resolution=0.1, seed=seed)

    labels = [0] * len(embeddings)
    for community in communities:
        initial_member = min(community)
        for member in community:
            labels[member] = initial_member

    clusters = {}
    for a, lbl in zip(articles, labels):
        if lbl not in clusters:
            clusters[lbl] = [a]
        else:
            clusters[lbl].append(a)
    clusters = dict(sorted(clusters.items(), key=lambda x: -len(x[1])))
    consistent = {}
    for k in clusters.keys():
        c_articles: List[Dict[str, Any]] = clusters[k]
        c_articles.sort(key=lambda article: (article['reach'], article['created']), reverse=True)
        consistent[c_articles[0]['id']] = c_articles
    return consistent


def cluster_prep(clusters: Dict[int, List[Dict[str, Any]]], key: str, start: datetime, end: datetime):
    data: Dict[str, Any] = {'key': key, 'from': start.isoformat(), 'to': end.isoformat(), 'clusters': []}
    for c, k in enumerate(clusters.keys()):
        articles: List[Dict[str, Any]] = clusters[k]
        cl = articles[0]
        size = len(articles)
        cluster = {'id': cl['id'], 'size': size, 'idx': c, 'title': cl['title']['text'], 'articles': []}
        articles.sort(key=lambda article: article["created"])
        data['clusters'].append(cluster)
        for x, a in enumerate(articles):
            cl_article = {
                'id': a['id'],
                'uuid': a['uuid'],
                'published': datetime.isoformat(a['published']),
                'created': datetime.isoformat(a['created']),
                'source_id': a['m_id'],
                'source': a['source'],
                'language': a['lang'],
                'country': a['country'],
                'reach': a['reach'],
                'type': a['type'],
                'url': a['url'],
                'title': a['title']['text'],
                'body': a['body']['text']
            }
            cluster['articles'].append(cl_article)
    return data


def main(data_args: DataArguments, model_args: ModelArguments) -> None:
    logger.info(f'Clustering {data_args.dataset_name}')
    source_dir = paths['base']['data'] / 'embed' / data_args.dataset_name
    if not source_dir.exists():
        logger.error(f'Source [embed] {data_args.dataset_name} directory not found: %s', source_dir)
        return

    target_dir = paths['cluster']['data'] / data_args.dataset_name
    target_dir.mkdir(parents=True, exist_ok=True)

    start = datetime.fromisoformat(data_args.source.select.start)
    if start.tzinfo is None:
        start = start.astimezone()

    end = datetime.fromisoformat(data_args.source.select.end)
    if end.tzinfo is None:
        end = end.astimezone()

    # Validate year == 2023 as Newsmon is 2023 only
    if start.year != 2023 or end.year != 2023:
        raise ValueError(
            f"Dates must be in year 2023; got start={start.date().isoformat()}, end={end.date().isoformat()}"
        )
    if end < start:
        raise ValueError(f"end must be >= start; got start={start}, end={end}")

    # Iterate month-by-month with per-month clipped ranges
    cur = start.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

    # is data only a subset
    subset = ''
    if data_args.source.select.subset:
        subset = f'{data_args.source.select.subset}_'

    seed = None
    if 'seed' in data_args.cluster.attributes:
        seed = data_args.cluster.attributes['seed']

    num_days = data_args.cluster.attributes['num_days']
    sim_threshold = data_args.cluster.attributes['sim_threshold']
    output_excel = False
    if 'output_excel' in data_args.cluster.attributes:
        output_excel = data_args.cluster.attributes['output_excel']
    while cur < end:
        next_month = cur + relativedelta(months=1)
        src_file = source_dir / f'{subset}data_{start.year}_{cur.month:02d}.jsonl'
        if not src_file.exists():
            raise FileNotFoundError(
                f"Data file not found: {src_file}, check data.source.select.start and data.source.select.end"
            )
        src_ebd_file = source_dir / f'{subset}data_{start.year}_{cur.month:02d}-{model_args.short_name}.jsonl'
        if not src_ebd_file.exists():
            raise FileNotFoundError(
                f"Data file not found: {src_file}, check data.source.select.start and data.source.select.end"
            )
        src_ebd = load_embeddings(src_ebd_file)

        collected: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        with (src_file.open('r', encoding='utf-8') as f_in):
            for line_no, line in enumerate(f_in, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    article = json.loads(line)
                    created = datetime.fromisoformat(article['created'].replace('Z', '+00:00')).astimezone()
                    delta = created - cur
                    bucket_start_date = cur + timedelta(days=(delta.days // num_days) * num_days)
                    bucket_key = bucket_start_date.strftime('%Y-%m-%d')

                    article['created'] = created
                    article['date'] = datetime.fromisoformat(article['date'].replace('Z', '+00:00'))
                    article['published'] = datetime.fromisoformat(article['published'].replace('Z', '+00:00'))
                    article['embedding'] = src_ebd[article['id']]

                    collected[bucket_key].append(article)
                except json.JSONDecodeError:
                    logger.warning('Skipping malformed JSON in %s line %d.', src_file.name, line_no)
                    raise
        base_name = f'{subset}data_clusters_{start.year}_{cur.month:02d}-{model_args.short_name}@{sim_threshold}'
        tgt_file = target_dir / f'{base_name}.jsonl'
        clusters: List[Dict[str, Any]] = []
        with (tgt_file.open('w', encoding='utf-8') as f_out):
            for key, articles in collected.items():
                bucket_clusters = cluster_louvain(articles, 'embedding', sim_threshold, seed)
                created_values = [a["created"] for a in articles if a.get("created") is not None]
                min_created = min(created_values)
                max_created = max(created_values)
                logger.info(
                    "Computed [%s] %s days clusters [%s from %s to %s] ",
                    len(bucket_clusters), num_days, key, min_created, max_created
                )
                bucket_clusters = cluster_prep(bucket_clusters, key, min_created, max_created)
                f_out.write(json.dumps(bucket_clusters, ensure_ascii=False) + "\n")
                clusters.append(bucket_clusters)
        if output_excel:
            from .__newsmon_xlsx import ClusterExcel
            tgt_file = target_dir / f'{base_name}.xlsx'
            excel = ClusterExcel(tgt_file)
            excel.write_xlsx(clusters)
        cur = next_month
