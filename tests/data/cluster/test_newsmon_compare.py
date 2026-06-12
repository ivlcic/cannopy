from datetime import datetime, timezone
from math import isclose

from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score

from src.data.cluster._newsmon.compare import compare_thresholds


class _LoggerStub:
    def info(self, *_args, **_kwargs) -> None:
        return None

    def warning(self, *_args, **_kwargs) -> None:
        return None


def _make_article(article_id: str, embedding: list[float]) -> dict:
    timestamp = datetime(2023, 3, 1, tzinfo=timezone.utc)
    return {
        'id': article_id,
        'uuid': article_id,
        'published': timestamp,
        'created': timestamp,
        'm_id': article_id,
        'source': 'source',
        'lang': 'en',
        'country': 'SI',
        'reach': 1,
        'type': 'news',
        'url': f'https://example.com/{article_id}',
        'title': {'text': article_id},
        'body': {'text': article_id},
        'embedding': embedding,
    }


def test_compare_thresholds_computes_label_invariant_scores() -> None:
    baseline_clusters = [
        {
            'key': '2023-03-01',
            'from': '2023-03-01T00:00:00+00:00',
            'to': '2023-03-05T23:59:59+00:00',
            'clusters': [
                {
                    'id': 'article_1',
                    'size': 3,
                    'idx': 0,
                    'title': 'article_1',
                    'articles': [
                        {'id': 'article_1'},
                        {'id': 'article_2'},
                        {'id': 'article_5'},
                    ],
                },
                {
                    'id': 'article_3',
                    'size': 2,
                    'idx': 1,
                    'title': 'article_3',
                    'articles': [
                        {'id': 'article_3'},
                        {'id': 'article_4'},
                    ],
                },
            ],
        }
    ]
    bucketed = {
        '2023-03-01': [
            _make_article('article_1', [1.0, 0.0]),
            _make_article('article_2', [1.0, 0.0]),
            _make_article('article_3', [0.0, 1.0]),
            _make_article('article_4', [0.0, 1.0]),
            _make_article('article_5', [0.0, 1.0]),
        ]
    }

    detail_rows, aggregate_rows = compare_thresholds(
        baseline_clusters=baseline_clusters,
        bucketed=bucketed,
        model_name='model-x',
        thresholds=[0.5],
        seed=2611,
        logger=_LoggerStub(),
    )

    assert len(detail_rows) == 1
    assert len(aggregate_rows) == 1

    detail_row = detail_rows[0]
    expected_true = [0, 0, 1, 1, 0]
    expected_pred = [0, 0, 1, 1, 1]

    assert detail_row['Model'] == 'model-x'
    assert detail_row['Span'] == '2023-03-01'
    assert detail_row['# Articles'] == 5
    assert detail_row['# Baseline clusters'] == 2
    assert detail_row['# Model clusters'] == 2
    assert isclose(detail_row['ARI'], adjusted_rand_score(expected_true, expected_pred))
    assert isclose(detail_row['AMI'], adjusted_mutual_info_score(expected_true, expected_pred))
    assert isclose(detail_row['Pairwise Precision'], 0.5)
    assert isclose(detail_row['Pairwise Recall'], 0.5)
    assert isclose(detail_row['Pairwise F1'], 0.5)

    aggregate_row = aggregate_rows[0]
    assert aggregate_row['Threshold'] == 0.5
    assert isclose(aggregate_row['Mean ARI'], detail_row['ARI'])
    assert isclose(aggregate_row['Mean AMI'], detail_row['AMI'])
    assert isclose(aggregate_row['Mean V-measure'], detail_row['V-measure'])
    assert isclose(aggregate_row['Mean Pairwise F1'], detail_row['Pairwise F1'])
