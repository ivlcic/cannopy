import json

from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _load_jsonl(file_path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with file_path.open('r', encoding='utf-8') as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f'Malformed JSON in {file_path} line {line_no}') from exc
    return rows


def _label_histogram(label_counts: Counter) -> Dict[str, int]:
    bins = {
        'le_1': 0,
        'le_5': 0,
        'le_10': 0,
        'le_50': 0,
        'le_100': 0,
        'le_500': 0,
        'gt_500': 0,
    }
    for count in label_counts.values():
        if count <= 1:
            bins['le_1'] += 1
        elif count <= 5:
            bins['le_5'] += 1
        elif count <= 10:
            bins['le_10'] += 1
        elif count <= 50:
            bins['le_50'] += 1
        elif count <= 100:
            bins['le_100'] += 1
        elif count <= 500:
            bins['le_500'] += 1
        else:
            bins['gt_500'] += 1
    return bins


def _label_histogram_bins(label_counts: Counter, bin_size: int = 5, max_count: int = 500) -> Tuple[List[str], List[int]]:
    num_bins = max_count // bin_size
    labels = [f'{i * bin_size + 1}-{(i + 1) * bin_size}' for i in range(num_bins)]
    labels.append(f'>{max_count}')
    counts = [0 for _ in labels]
    for count in label_counts.values():
        if count > max_count:
            counts[-1] += 1
            continue
        idx = max(0, (count - 1) // bin_size)
        counts[idx] += 1
    return labels, counts


def _svg_text(x: int, y: int, text: str, size: int = 12, anchor: str = 'start', weight: str = 'normal') -> str:
    safe = (
        text.replace('&', '&amp;')
        .replace('<', '&lt;')
        .replace('>', '&gt;')
    )
    return (
        f'<text x="{x}" y="{y}" font-size="{size}" text-anchor="{anchor}" '
        f'font-family="sans-serif" font-weight="{weight}">{safe}</text>'
    )


def _render_label_histogram_svg(output_file: Path, label_counts: Counter) -> None:
    bin_size = 5
    max_count = 500
    bin_labels, bin_counts = _label_histogram_bins(label_counts, bin_size=bin_size, max_count=max_count)

    width = 800
    height = 675
    margin_left = 90
    margin_right = 30
    margin_top = 40
    margin_bottom = 120
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    max_y = 1200
    bar_width = max(2, plot_width / max(1, len(bin_counts)))

    def x_pos(index: int) -> float:
        return margin_left + index * bar_width

    def y_pos(value: int) -> float:
        return margin_top + plot_height - (value / max_y) * plot_height

    def threshold_index(threshold: int) -> int:
        return max(0, (threshold - 1) // bin_size)

    ten_idx = threshold_index(10)
    fifty_idx = threshold_index(50)
    five_hundred_idx = threshold_index(500)

    sum_10 = sum(1 for count in label_counts.values() if count <= 10)
    sum_50 = sum(1 for count in label_counts.values() if count <= 50)
    sum_500 = sum(1 for count in label_counts.values() if count > 500)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<line x1="{margin_left}" y1="{margin_top + plot_height}" x2="{width - margin_right}" y2="{margin_top + plot_height}" stroke="#333" stroke-width="1"/>',
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_height}" stroke="#333" stroke-width="1"/>',
    ]

    y_ticks = 5
    for i in range(y_ticks + 1):
        value = round((max_y / y_ticks) * i)
        y = y_pos(value)
        parts.append(f'<line x1="{margin_left}" y1="{y}" x2="{width - margin_right}" y2="{y}" stroke="#e5e5e5" stroke-width="1"/>')
        parts.append(_svg_text(margin_left - 10, int(y) + 4, str(value), size=11, anchor='end'))

    for idx, count in enumerate(bin_counts):
        x = x_pos(idx)
        y = y_pos(count)
        rect_height = margin_top + plot_height - y
        parts.append(
            f'<rect x="{x + 0.5}" y="{y}" width="{max(1, bar_width - 1)}" height="{rect_height}" '
            f'fill="#4a7dbb"/>'
        )

    for idx, color in [(ten_idx, '#c62828'), (fifty_idx, '#c62828'), (five_hundred_idx, '#c62828')]:
        x = x_pos(idx + 1)
        parts.append(
            f'<line x1="{x}" y1="{margin_top}" x2="{x}" y2="{margin_top + plot_height}" '
            f'stroke="{color}" stroke-width="2" stroke-dasharray="4 4"/>'
        )

    tick_step = 10
    for idx in range(0, len(bin_labels), tick_step):
        x = x_pos(idx)
        label_value = str(idx * bin_size)
        parts.append(f'<line x1="{x}" y1="{margin_top + plot_height}" x2="{x}" y2="{margin_top + plot_height + 6}" stroke="#333" stroke-width="1"/>')
        parts.append(_svg_text(int(x), margin_top + plot_height + 22, label_value, size=10, anchor='middle'))

    parts.append(_svg_text(width // 2, height - 30, 'Number of occurrences in documents', size=14, anchor='middle'))
    parts.append(
        f'<g transform="translate(20,{height // 2}) rotate(-90)">' +
        _svg_text(0, 0, 'Number of labels', size=14, anchor='middle') +
        '</g>'
    )

    annotation_specs = [
        (int(x_pos(ten_idx) + 20), margin_top + 70, f'{sum_10} labels', '(≤ 10 occurrences)'),
        (int(x_pos(fifty_idx) + 20), margin_top + 240, f'{sum_50} labels', '(≤ 50 occurrences)'),
        (max(margin_left + 20, min(int(x_pos(five_hundred_idx) - 170), width - 200)), margin_top + 360, f'{sum_500} labels', '(> 500 occurrences)'),
    ]
    balloon_width = 165
    balloon_half_width = balloon_width // 2
    for x, y, line1, line2 in annotation_specs:
        parts.append(
            f'<rect x="{x}" y="{y - 28}" width="{balloon_width}" height="52" rx="10" ry="10" fill="#f6d365" opacity="0.95" stroke="#c9a227"/>'
        )
        parts.append(_svg_text(x + balloon_half_width, y - 6, line1, size=13, anchor='middle', weight='bold'))
        parts.append(_svg_text(x + balloon_half_width, y + 12, line2, size=12, anchor='middle'))

    parts.append('</svg>')
    output_file.write_text('\n'.join(parts), encoding='utf-8')


def _compute_stats(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    label_counts: Counter = Counter()
    language_counts: Counter = Counter()
    country_counts: Counter = Counter()
    labels_per_sample: List[int] = []
    label_combo_counts: Counter = Counter()
    article_ids = set()

    for row in rows:
        labels = row.get('label', [])
        label_counts.update(labels)
        language = row.get('lang')
        country = row.get('country')
        article_id = row.get('a_id') or row.get('id')
        if language:
            language_counts[language] += 1
        if country:
            country_counts[country] += 1
        if article_id:
            article_ids.add(article_id)
        labels_per_sample.append(len(labels))
        label_combo_counts[str(sorted(set(labels)))] += 1

    avg_labels = 0.0
    if labels_per_sample:
        avg_labels = sum(labels_per_sample) / len(labels_per_sample)
    label_density = 0.0
    if rows and label_counts:
        label_density = sum((len(row.get('label', [])) / len(label_counts)) for row in rows) / len(rows)
    label_cardinality = avg_labels
    label_diversity = len(label_combo_counts)
    label_count_stddev = 0.0
    if len(labels_per_sample) > 1:
        mean = avg_labels
        label_count_stddev = (sum((value - mean) ** 2 for value in labels_per_sample) / len(labels_per_sample)) ** 0.5

    top_labels = [
        {'label': label_id, 'count': count}
        for label_id, count in label_counts.most_common(50)
    ]

    return {
        'num_samples': len(rows),
        'num_articles': len(article_ids),
        'num_labels': len(label_counts),
        'avg_labels_per_sample': avg_labels,
        'label_density': label_density,
        'label_cardinality': label_cardinality,
        'label_diversity': label_diversity,
        'label_count_stddev': label_count_stddev,
        'languages': dict(sorted(language_counts.items())),
        'countries': dict(sorted(country_counts.items())),
        'label_histogram': _label_histogram(label_counts),
        'top_labels': top_labels,
    }


def _load_split_stats(split_dir: Path, subset: str) -> Dict[str, Dict[str, Any]]:
    split_stats: Dict[str, Dict[str, Any]] = {}
    for split_name in ['train', 'eval', 'test']:
        split_file = split_dir / f'{subset}_{split_name}.jsonl'
        if not split_file.exists():
            continue
        split_stats[split_name] = _compute_stats(_load_jsonl(split_file))
    return split_stats