import json
import numpy as np
import networkx as nx

from datetime import datetime, timedelta
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


def main(data_args: DataArguments, model_args: ModelArguments) -> None:
    logger.info(f'Clustering {data_args.dataset_name}')
    source_dir = paths['base']['data'] / 'embed' / data_args.dataset_name
    if not source_dir.exists():
        logger.error(f'Source [embed] {data_args.dataset_name} directory not found: %s', source_dir)
        return

    target_dir = paths['cluster']['data'] / data_args.dataset_name
    target_dir.mkdir(parents=True, exist_ok=True)

    start = datetime.fromisoformat(data_args.source.select.start)
    end = datetime.fromisoformat(data_args.source.select.end)
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
    while cur < end:
        next_month = cur + relativedelta(months=1)
        cur = datetime.fromisoformat('2023-03-01T00:00:00.000+00:00')
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

        collected: Dict[str, List[Dict[str, Any]]] = {}
        with (src_file.open('r', encoding='utf-8') as f_in):
            for line_no, line in enumerate(f_in, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    created = datetime.fromisoformat(obj['created'].replace('Z', '+00:00'))
                    delta = created - cur
                    bucket_start_date = cur + timedelta(days=(delta.days // num_days) * num_days)
                    bucket_key = bucket_start_date.strftime('%Y-%m-%d')
                    obj['created'] = created
                    obj['date'] = datetime.fromisoformat(obj['date'].replace('Z', '+00:00'))
                    obj['published'] = datetime.fromisoformat(obj['published'].replace('Z', '+00:00'))
                    obj['embedding'] = src_ebd[obj['id']]
                    if bucket_key not in collected:
                        collected[bucket_key] = []
                    collected[bucket_key].append(obj)
                except json.JSONDecodeError:
                    logger.warning('Skipping malformed JSON in %s line %d.', src_file.name, line_no)
                    raise

        for key, articles in collected.items():
            daily_clusters = cluster_louvain(articles, 'embedding', sim_threshold, seed)
            logger.info(
                "Computed [%s] %s days clusters [%s from %s::%s] ",
                len(daily_clusters), num_days, key, cur, end
            )

        # tgt_file = target_dir / f'{subset}data_{start.year}_{cur.month:02d}.jsonl'
        cur = next_month
