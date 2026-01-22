import json
import logging

import numpy as np
import networkx as nx

from collections import defaultdict
from pathlib import Path
from datetime import datetime, timedelta
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


def load_embeddings(file_name) -> Dict[str, List[float]]:
    embeddings: Dict[str, List[float]] = {}
    with file_name.open('r', encoding='utf-8') as f_in:
        for line_no, line in enumerate(f_in, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if 'id' not in obj:
                    logger.warning('Missing id in %s line %d.', file_name, line_no)
                    continue
                # accept both embeddings/embedding keys
                vec = obj.get('embeddings', obj.get('embedding'))
                if vec is None:
                    logger.warning('Missing embeddings in %s line %d.', file_name, line_no)
                    continue
                embeddings[obj['id']] = vec
            except json.JSONDecodeError:
                logger.warning('Skipping malformed JSON in %s line %d.', file_name.name, line_no)
                raise
    return embeddings


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
    # Remove self-loops (diagonal True values)
    np.fill_diagonal(similarity_matrix, False)

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


def main():
    work_dir = Path.cwd()
    subset = 'stories_'
    short_name = 'bge-m3'
    seed = 2611
    num_days = 5
    sim_threshold = 0.88  # 0.88 if bge, 0.96 id ada_002
    year = 2023
    month = 3

    src_file = work_dir / f'{subset}data_{year}_{month:02d}.jsonl'
    if not src_file.exists():
        raise FileNotFoundError(
            f"Data file not found: {src_file}"
        )
    src_ebd_file = work_dir / f'{subset}data_{year}_{month:02d}-{short_name}.jsonl'
    if not src_ebd_file.exists():
        raise FileNotFoundError(
            f"Data file not found: {src_file}"
        )
    src_ebd = load_embeddings(src_ebd_file)

    # the 1st of march in CET
    cur = datetime.fromisoformat('2023-02-28T23:00:00.000Z'.replace('Z', '+00:00')).astimezone()
    buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    # read samples and move them to num_days time range buckets starting with 1st of march in CET
    with (src_file.open('r', encoding='utf-8') as f_in):
        for line_no, line in enumerate(f_in, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                article = json.loads(line)
                # to local time
                created = datetime.fromisoformat(article['created'].replace('Z', '+00:00')).astimezone()
                delta = created - cur
                bucket_start_date = cur + timedelta(days=(delta.days // num_days) * num_days)
                bucket_key = bucket_start_date.strftime('%Y-%m-%d')

                article['created'] = created
                article['date'] = datetime.fromisoformat(article['date'].replace('Z', '+00:00'))
                article['published'] = datetime.fromisoformat(article['published'].replace('Z', '+00:00'))
                article['embedding'] = src_ebd[article['id']]

                buckets[bucket_key].append(article)
            except json.JSONDecodeError:
                logger.warning('Skipping malformed JSON in %s line %d.', src_file.name, line_no)
                raise

    base_name = f'{subset}data_clusters_{year}_{month:02d}-{short_name}@{sim_threshold}'
    tgt_file = work_dir / f'{base_name}.jsonl'
    clusters: List[Dict[str, Any]] = []
    with (tgt_file.open('w', encoding='utf-8') as f_out):
        for key, articles in buckets.items():
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


if __name__=="__main__":
    main()
