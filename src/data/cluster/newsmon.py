import json

import networkx as nx
import numpy as np

from collections import defaultdict
from datetime import datetime, timedelta
from logging import Logger
from typing import Any, Dict, List

from ..embed.newsmon import load_embeddings
from ...app.args.data import DataArguments
from ...app.args.model import ModelArguments

logger: Logger
paths: Dict[str, Any]


def _get_subset_name(data_args: DataArguments) -> str:
    subset = data_args.source.select.subset
    if subset:
        return subset
    return data_args.dataset_name


def _parse_datetime(value: str) -> datetime:
    return datetime.fromisoformat(value.replace('Z', '+00:00')).astimezone()


def cosine_similarity_matrix(x, eps=1e-12):
    x = np.asarray(x, dtype=float)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    x = x / np.clip(norms, eps, None)
    return x @ x.T


def cluster_louvain(articles: List[Dict[str, Any]], embed_field_name: str, sim_threshold: float = 0.84, seed=None):
    embeddings = []
    [embeddings.append(x[embed_field_name]) for x in articles]
    embeddings = np.array(embeddings)
    x = cosine_similarity_matrix(embeddings)

    similarity_matrix = x > sim_threshold
    np.fill_diagonal(similarity_matrix, False)

    graph = nx.from_numpy_array(similarity_matrix)
    communities = nx.algorithms.community.louvain_communities(graph, resolution=0.1, seed=seed)

    labels = [0] * len(embeddings)
    for community in communities:
        initial_member = min(community)
        for member in community:
            labels[member] = initial_member

    clusters = {}
    for article, lbl in zip(articles, labels):
        if lbl not in clusters:
            clusters[lbl] = [article]
        else:
            clusters[lbl].append(article)
    clusters = dict(sorted(clusters.items(), key=lambda x: -len(x[1])))

    consistent = {}
    for key in clusters.keys():
        c_articles: List[Dict[str, Any]] = clusters[key]
        c_articles.sort(key=lambda article: (article['reach'], article['created']), reverse=True)
        consistent[c_articles[0]['id']] = c_articles
    return consistent


def cluster_prep(clusters: Dict[int, List[Dict[str, Any]]], key: str, start: datetime, end: datetime):
    data: Dict[str, Any] = {'key': key, 'from': start.isoformat(), 'to': end.isoformat(), 'clusters': []}
    for idx, cluster_key in enumerate(clusters.keys()):
        articles: List[Dict[str, Any]] = clusters[cluster_key]
        lead_article = articles[0]
        cluster = {
            'id': lead_article['id'],
            'size': len(articles),
            'idx': idx,
            'title': lead_article['title']['text'],
            'articles': []
        }
        articles.sort(key=lambda article: article['created'])
        data['clusters'].append(cluster)
        for article in articles:
            cluster['articles'].append({
                'id': article['id'],
                'uuid': article['uuid'],
                'published': article['published'].isoformat(),
                'created': article['created'].isoformat(),
                'source_id': article['m_id'],
                'source': article['source'],
                'language': article['lang'],
                'country': article['country'],
                'reach': article['reach'],
                'type': article['type'],
                'url': article['url'],
                'title': article['title']['text'],
                'body': article['body']['text']
            })
    return data


def main(data_args: DataArguments, model_args: ModelArguments) -> None:
    logger.info('Clustering %s', data_args.dataset_name)

    subset = _get_subset_name(data_args)
    prepared_dir = paths['prepare']['data'] / data_args.dataset_name
    embed_dir = paths['embed']['data'] / data_args.dataset_name
    if not prepared_dir.exists():
        logger.error('Source [prepare] %s directory not found: %s', data_args.dataset_name, prepared_dir)
        return
    if not embed_dir.exists():
        logger.error('Source [embed] %s directory not found: %s', data_args.dataset_name, embed_dir)
        return

    src_file = prepared_dir / f'{subset}.jsonl'
    if not src_file.exists():
        raise FileNotFoundError(f'Prepared data file not found: {src_file}')

    src_ebd_file = embed_dir / f'{subset}.{model_args.short_name}.jsonl'
    if not src_ebd_file.exists():
        raise FileNotFoundError(f'Embedding file not found: {src_ebd_file}')

    target_dir = paths['cluster']['data'] / data_args.dataset_name
    target_dir.mkdir(parents=True, exist_ok=True)

    seed = data_args.cluster.attributes.get('seed')
    num_days = data_args.cluster.attributes['num_days']
    sim_threshold = data_args.cluster.attributes['sim_threshold']
    output_excel = data_args.cluster.attributes.get('output_excel', False)

    src_ebd = load_embeddings(src_ebd_file)
    collected: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    min_created = None
    max_created = None

    with src_file.open('r', encoding='utf-8') as f_in:
        for line_no, line in enumerate(f_in, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                article = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f'Malformed JSON in {src_file} line {line_no}') from exc

            article_id = article.get('id')
            if not article_id:
                logger.warning('Missing id in %s line %d.', src_file, line_no)
                continue
            if article_id not in src_ebd:
                logger.warning('Missing embedding for %s in %s.', article_id, src_ebd_file)
                continue

            created = _parse_datetime(article['created'])
            published = _parse_datetime(article['published'])
            article['created'] = created
            article['published'] = published
            article['date'] = _parse_datetime(article['date'])
            article['embedding'] = src_ebd[article_id]

            if min_created is None or created < min_created:
                min_created = created
            if max_created is None or created > max_created:
                max_created = created

            collected[created.strftime('%Y-%m-01')].append(article)

    if min_created is None or max_created is None:
        raise ValueError(f'No clusterable articles found in {src_file}')

    bucketed: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    anchor = min_created.replace(hour=0, minute=0, second=0, microsecond=0)
    for articles in collected.values():
        for article in articles:
            delta = article['created'] - anchor
            bucket_start_date = anchor + timedelta(days=(delta.days // num_days) * num_days)
            bucket_key = bucket_start_date.strftime('%Y-%m-%d')
            bucketed[bucket_key].append(article)

    base_name = f'{subset}_clusters-{model_args.short_name}@{sim_threshold}'
    tgt_file = target_dir / f'{base_name}.jsonl'
    clusters: List[Dict[str, Any]] = []
    with tgt_file.open('w', encoding='utf-8') as f_out:
        for key, articles in sorted(bucketed.items()):
            bucket_clusters = cluster_louvain(articles, 'embedding', sim_threshold, seed)
            created_values = [article['created'] for article in articles]
            bucket_min_created = min(created_values)
            bucket_max_created = max(created_values)
            logger.info(
                'Computed [%s] %s day clusters [%s from %s to %s]',
                len(bucket_clusters), num_days, key, bucket_min_created, bucket_max_created
            )
            prepared_cluster = cluster_prep(bucket_clusters, key, bucket_min_created, bucket_max_created)
            f_out.write(json.dumps(prepared_cluster, ensure_ascii=False) + '\n')
            clusters.append(prepared_cluster)

    if output_excel:
        from .__newsmon_xlsx import ClusterExcel
        xlsx_file = target_dir / f'{base_name}.xlsx'
        excel = ClusterExcel(xlsx_file)
        excel.write_xlsx(clusters)
