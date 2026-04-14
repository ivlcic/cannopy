import json
from collections import Counter
from logging import Logger
from typing import Any, Dict

from .__common import _load_jsonl, _compute_stats, _load_split_stats, _render_label_histogram_svg
from ...app.args.data import DataArguments

logger: Logger
paths: Dict[str, Any]


# noinspection DuplicatedCode
def main(data_args: DataArguments) -> None:
    output_dir = paths['analyze']['data'] / 'eurlex'
    output_dir.mkdir(parents=True, exist_ok=True)

    base_dir = paths['base']['data'] / 'prepare' / 'eurlex'
    split_dir = paths['base']['data'] / 'split' / 'eurlex'
    source_file = base_dir / f'{data_args.dataset_name}.jsonl'
    if not source_file.exists():
        raise FileNotFoundError(f'Prepared subset file not found: {source_file}')

    base_stats = _compute_stats(_load_jsonl(source_file))
    split_stats = _load_split_stats(split_dir, data_args.dataset_name)

    report = {
        'dataset': data_args.dataset_name,
        'prepared': base_stats,
        'splits': split_stats,
    }

    report_file = output_dir / f'{data_args.dataset_name}.stats.json'
    with report_file.open('w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    prepared_rows = _load_jsonl(source_file)
    prepared_label_counts: Counter = Counter()
    for row in prepared_rows:
        prepared_label_counts.update(row.get('label', []))
    _render_label_histogram_svg(output_dir / f'{data_args.dataset_name}.label_histogram.svg', prepared_label_counts)

    logger.info(
        'Analyzed %s: samples=%s labels=%s avg_labels=%.3f density=%.6f diversity=%s',
        data_args.dataset_name,
        base_stats['num_samples'],
        base_stats['num_labels'],
        base_stats['avg_labels_per_sample'],
        base_stats['label_density'],
        base_stats['label_diversity'],
    )
    if split_stats:
        logger.info(
            'Split sample counts train/eval/test: %s/%s/%s',
            split_stats.get('train', {}).get('num_samples', 0),
            split_stats.get('eval', {}).get('num_samples', 0),
            split_stats.get('test', {}).get('num_samples', 0),
        )
