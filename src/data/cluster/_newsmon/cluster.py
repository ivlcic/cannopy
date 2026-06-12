from datetime import datetime
from logging import Logger
from typing import Any, Dict, List

import networkx as nx
import numpy as np



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
    # noinspection PyTypeChecker
    np.fill_diagonal(similarity_matrix, False)
    # noinspection PyTypeChecker
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
        c_articles.sort(key=lambda a: (a['reach'], a['created']), reverse=True)
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
        articles.sort(key=lambda a: a['created'])
        data['clusters'].append(cluster)
        for article in articles:
            # noinspection PyUnresolvedReferences
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


def compute_clusters(bucketed: Dict[str, List[Dict[str, Any]]],
                     sim_threshold: float,
                     seed: int,
                     logger: Logger) -> List[Dict[str, Any]]:
    clusters: List[Dict[str, Any]] = []
    for key, articles in sorted(bucketed.items()):
        bucket_clusters = cluster_louvain(articles, 'embedding', sim_threshold, seed)
        created_values = [article['created'] for article in articles]
        bucket_min_created = min(created_values)
        bucket_max_created = max(created_values)
        logger.info(
            'Computed [%s] clusters [%s from %s to %s]',
            len(bucket_clusters), key, bucket_min_created, bucket_max_created
        )
        prepared_cluster = cluster_prep(bucket_clusters, key, bucket_min_created, bucket_max_created)
        clusters.append(prepared_cluster)
    return clusters