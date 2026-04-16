import json
from collections import Counter
from logging import Logger

from .newsmon import load_jsonl, compute_stats, load_split_stats, render_label_histogram_svg
from ...app.args.data import DataArguments
from ...app.args.runtime import Paths

logger: Logger
paths: Paths


# noinspection DuplicatedCode
def main(data_args: DataArguments) -> None:
    base_dir = paths.get_ctx_path('prepare')
    split_dir = paths.get_ctx_path('split')
    source_file = base_dir / f'{data_args.dataset_name}.jsonl'
    if not source_file.exists():
        raise FileNotFoundError(f'Prepared subset file not found: {source_file}')

    base_stats = compute_stats(load_jsonl(source_file))
    split_stats = load_split_stats(split_dir, data_args.dataset_name)

    report = {
        'dataset': data_args.dataset_name,
        'prepared': base_stats,
        'splits': split_stats,
    }

    report_file = paths.context / f'{data_args.dataset_name}.stats.json'
    with report_file.open('w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    prepared_rows = load_jsonl(source_file)
    prepared_label_counts: Counter = Counter()
    for row in prepared_rows:
        prepared_label_counts.update(row.get('label', []))
    render_label_histogram_svg(paths.context / f'{data_args.dataset_name}.label_histogram.svg', prepared_label_counts)

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
