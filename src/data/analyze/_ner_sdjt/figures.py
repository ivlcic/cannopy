from __future__ import annotations

import csv
from html import escape
from pathlib import Path
from typing import Any, Dict, List

from ....app.ner import NerSample
from ...resample.ner_sdjt import AUX_LANGUAGES, MAIN_LANGUAGES


def _language_sort_key(lang: str) -> tuple[int, int | str]:
    if lang in MAIN_LANGUAGES:
        return 0, MAIN_LANGUAGES.index(lang)
    if lang in AUX_LANGUAGES:
        return 1, AUX_LANGUAGES.index(lang)
    return 2, lang


def _read_base_ner_stats(stats_file: Path) -> List[Dict[str, Any]]:
    if not stats_file.exists():
        raise FileNotFoundError(f"Base NER stats file not found at {stats_file}. Run `./data analyze ner` first.")

    rows: List[Dict[str, Any]] = []
    with stats_file.open("r", encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            language = str(row.get("language", "")).strip()
            if language not in MAIN_LANGUAGES and language not in AUX_LANGUAGES:
                continue
            entity_counts = {entity_type: 0 for entity_type in NerSample.CORE_ENTITY_TYPES}
            for key, raw_value in row.items():
                if not key or "-" not in key:
                    continue
                prefix, entity_type = key.split("-", 1)
                if prefix not in {"B", "I"} or entity_type not in entity_counts:
                    continue
                entity_counts[entity_type] += int(raw_value or 0)
            rows.append({
                "language": language,
                "tokens": int(row.get("tokens", 0) or 0),
                "entity_counts": entity_counts,
                "entity_total": sum(entity_counts.values()),
                "is_aux": language in AUX_LANGUAGES,
            })
    rows.sort(key=lambda row: _language_sort_key(row["language"]))
    if not rows:
        raise ValueError(f"No rows found in {stats_file}.")
    return rows


def _build_language_slots(stats_rows: List[Dict[str, Any]], plot_width: float) -> Dict[str, float]:
    gap_units = 0.1
    step_units = 0.68
    units: List[float] = []
    cursor = 0.0
    previous_is_aux: bool | None = None
    for row in stats_rows:
        is_aux = bool(row["is_aux"])
        if previous_is_aux is not None:
            cursor += step_units
            if is_aux != previous_is_aux:
                cursor += gap_units
        units.append(cursor)
        previous_is_aux = is_aux
    total_units = (units[-1] + step_units) if units else 1.0
    scale = plot_width / total_units
    return {
        row["language"]: (unit + step_units / 2.0) * scale
        for row, unit in zip(stats_rows, units)
    }


def _compute_aux_region(stats_rows: List[Dict[str, Any]], language_slots: Dict[str, float]) -> tuple[float, float, float] | None:
    main_centers = [language_slots[row["language"]] for row in stats_rows if not row["is_aux"]]
    aux_centers = [language_slots[row["language"]] for row in stats_rows if row["is_aux"]]
    if not main_centers or not aux_centers:
        return None
    separator_x = (max(main_centers) + min(aux_centers)) / 2.0
    return min(aux_centers), max(aux_centers), separator_x


def _build_value_axis(max_value: float, n_ticks: int = 5) -> tuple[float, float]:
    if max_value <= 0:
        return 1.0, 0.2
    rough_step = max_value / n_ticks
    if rough_step <= 1:
        step = 0.2
    elif rough_step <= 2:
        step = 0.5
    elif rough_step <= 5:
        step = 1.0
    elif rough_step <= 10:
        step = 2.0
    elif rough_step <= 20:
        step = 5.0
    elif rough_step <= 50:
        step = 10.0
    else:
        step = 25.0
    axis_max = step
    while axis_max < max_value:
        axis_max += step
    return axis_max, step


def _write_entity_density_svg(output_file: Path, stats_rows: List[Dict[str, Any]]) -> None:
    width = 640
    height = 640
    margin_left = 64.0
    margin_right = 18.0
    margin_top = 96.0
    margin_bottom = 64.0
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    language_slots = _build_language_slots(stats_rows, plot_width)
    aux_region = _compute_aux_region(stats_rows, language_slots)
    bar_width = min(22.0, plot_width / max(len(stats_rows) * 2.8, 1.0))

    density_rows: List[Dict[str, Any]] = []
    max_density = 0.0
    for row in stats_rows:
        density = (row["entity_total"] * 1000.0 / row["tokens"]) if row["tokens"] else 0.0
        density_rows.append({
            "language": row["language"],
            "density": density,
            "is_aux": row["is_aux"],
        })
        max_density = max(max_density, density)
    axis_max, tick_step = _build_value_axis(max_density)

    svg_lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<defs>",
        '<pattern id="auxHatch" patternUnits="userSpaceOnUse" width="8" height="8" patternTransform="rotate(45)">',
        '<rect width="8" height="8" fill="#E68613"/>',
        '<line x1="0" y1="0" x2="0" y2="8" stroke="#FFFFFF" stroke-width="3" stroke-opacity="0.8"/>',
        "</pattern>",
        "</defs>",
        f'<rect width="{width}" height="{height}" fill="#FFFFFF"/>',
        f'<line x1="{margin_left}" y1="{margin_top + plot_height}" x2="{margin_left + plot_width}" y2="{margin_top + plot_height}" '
        'stroke="#111827" stroke-width="1.2"/>',
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_height}" '
        'stroke="#111827" stroke-width="1.2"/>',
    ]

    if aux_region is not None:
        aux_start, aux_end, separator_x = aux_region
        svg_lines.append(
            f'<line x1="{margin_left + separator_x:.2f}" y1="{margin_top - 18:.2f}" '
            f'x2="{margin_left + separator_x:.2f}" y2="{margin_top + plot_height:.2f}" '
            'stroke="#6B7280" stroke-width="1.5" stroke-dasharray="6 6"/>'
        )
        svg_lines.append(
            f'<text x="{margin_left + (aux_start + aux_end) / 2:.2f}" y="{margin_top - 28:.2f}" text-anchor="middle" '
            'font-size="18" font-family="Arial, sans-serif" fill="#374151">Auxiliary</text>'
        )

    tick_value = 0.0
    while tick_value <= axis_max + 1e-9:
        y = margin_top + plot_height - (tick_value / axis_max * plot_height if axis_max else 0.0)
        svg_lines.append(
            f'<line x1="{margin_left}" y1="{y:.2f}" x2="{margin_left + plot_width}" y2="{y:.2f}" '
            'stroke="#E5E7EB" stroke-width="1"/>'
        )
        svg_lines.append(
            f'<text x="{margin_left - 10}" y="{y + 5:.2f}" text-anchor="end" font-size="13" '
            'font-family="Arial, sans-serif" fill="#374151">'
            f'{tick_value:.0f}</text>'
        )
        tick_value += tick_step

    for row in density_rows:
        x = margin_left + language_slots[row["language"]] - bar_width / 2
        bar_height = (row["density"] / axis_max * plot_height) if axis_max else 0.0
        y = margin_top + plot_height - bar_height
        fill = "url(#auxHatch)" if row["is_aux"] else "#4C78A8"
        svg_lines.append(
            f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_width:.2f}" height="{bar_height:.2f}" '
            f'fill="{fill}" stroke="#1F2937" stroke-width="1"/>'
        )
        svg_lines.append(
            f'<text x="{x + bar_width / 2:.2f}" y="{margin_top + plot_height + 18:.2f}" text-anchor="middle" '
            'font-size="15" font-family="Arial, sans-serif" fill="#111827">'
            f'{escape(row["language"])}</text>'
        )
    svg_lines.extend([
        f'<text x="{margin_left + plot_width / 2:.2f}" y="{height - 16:.2f}" text-anchor="middle" font-size="16" '
        'font-family="Arial, sans-serif" fill="#111827">Language</text>',
        f'<text x="18" y="{margin_top + plot_height / 2:.2f}" text-anchor="middle" font-size="16" '
        'font-family="Arial, sans-serif" fill="#111827" transform="rotate(-90 18 '
        f'{margin_top + plot_height / 2:.2f})">Entity-labelled tokens per 1,000 tokens</text>',
        "</svg>",
    ])

    output_file.write_text("\n".join(svg_lines), encoding="utf-8")


def _write_label_composition_svg(output_file: Path, stats_rows: List[Dict[str, Any]]) -> None:
    width = 640
    height = 640
    margin_left = 64.0
    margin_right = 18.0
    margin_top = 96.0
    margin_bottom = 64.0
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    language_slots = _build_language_slots(stats_rows, plot_width)
    aux_region = _compute_aux_region(stats_rows, language_slots)
    bar_width = min(22.0, plot_width / max(len(stats_rows) * 2.8, 1.0))
    colors = {
        "PER": "#54A24B",
        "ORG": "#E45756",
        "LOC": "#4C78A8",
    }

    svg_lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect width="{width}" height="{height}" fill="#FFFFFF"/>',
        '<rect x="76" y="56" width="16" height="16" fill="#54A24B" stroke="#1F2937" stroke-width="1"/>',
        '<text x="98" y="70" font-size="18" font-family="Arial, sans-serif" fill="#111827">PER</text>',
        '<rect x="162" y="56" width="16" height="16" fill="#E45756" stroke="#1F2937" stroke-width="1"/>',
        '<text x="184" y="70" font-size="18" font-family="Arial, sans-serif" fill="#111827">ORG</text>',
        '<rect x="250" y="56" width="16" height="16" fill="#4C78A8" stroke="#1F2937" stroke-width="1"/>',
        '<text x="272" y="70" font-size="18" font-family="Arial, sans-serif" fill="#111827">LOC</text>',
        f'<line x1="{margin_left}" y1="{margin_top + plot_height}" x2="{margin_left + plot_width}" y2="{margin_top + plot_height}" '
        'stroke="#111827" stroke-width="1.2"/>',
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_height}" '
        'stroke="#111827" stroke-width="1.2"/>',
    ]

    if aux_region is not None:
        aux_start, aux_end, separator_x = aux_region
        svg_lines.append(
            f'<line x1="{margin_left + separator_x:.2f}" y1="{margin_top - 18:.2f}" '
            f'x2="{margin_left + separator_x:.2f}" y2="{margin_top + plot_height:.2f}" '
            'stroke="#6B7280" stroke-width="1.5" stroke-dasharray="6 6"/>'
        )
        svg_lines.append(
            f'<text x="{margin_left + (aux_start + aux_end) / 2:.2f}" y="{margin_top - 28:.2f}" text-anchor="middle" '
            'font-size="18" font-family="Arial, sans-serif" fill="#374151">Auxiliary</text>'
        )

    for tick_idx in range(6):
        proportion = tick_idx / 5.0
        y = margin_top + plot_height - proportion * plot_height
        svg_lines.append(
            f'<line x1="{margin_left}" y1="{y:.2f}" x2="{margin_left + plot_width}" y2="{y:.2f}" '
            'stroke="#E5E7EB" stroke-width="1"/>'
        )
        svg_lines.append(
            f'<text x="{margin_left - 10}" y="{y + 5:.2f}" text-anchor="end" font-size="13" '
            'font-family="Arial, sans-serif" fill="#374151">'
            f'{proportion * 100:.0f}%</text>'
        )

    for row in stats_rows:
        total = float(row["entity_total"])
        x = margin_left + language_slots[row["language"]] - bar_width / 2
        y_cursor = margin_top + plot_height
        for entity_type in NerSample.CORE_ENTITY_TYPES:
            proportion = (row["entity_counts"][entity_type] / total) if total else 0.0
            if proportion <= 0:
                continue
            bar_height = proportion * plot_height
            y_cursor -= bar_height
            svg_lines.append(
                f'<rect x="{x:.2f}" y="{y_cursor:.2f}" width="{bar_width:.2f}" height="{bar_height:.2f}" '
                f'fill="{colors[entity_type]}" stroke="#FFFFFF" stroke-width="0.8"/>'
            )
        svg_lines.append(
            f'<rect x="{x:.2f}" y="{margin_top:.2f}" width="{bar_width:.2f}" height="{plot_height:.2f}" '
            'fill="none" stroke="#1F2937" stroke-width="0.8"/>'
        )
        svg_lines.append(
            f'<text x="{x + bar_width / 2:.2f}" y="{margin_top + plot_height + 18:.2f}" text-anchor="middle" '
            'font-size="15" font-family="Arial, sans-serif" fill="#111827">'
            f'{escape(row["language"])}</text>'
        )

    svg_lines.extend([
        f'<text x="{margin_left + plot_width / 2:.2f}" y="{height - 16:.2f}" text-anchor="middle" font-size="16" '
        'font-family="Arial, sans-serif" fill="#111827">Language</text>',
        f'<text x="18" y="{margin_top + plot_height / 2:.2f}" text-anchor="middle" font-size="16" '
        'font-family="Arial, sans-serif" fill="#111827" transform="rotate(-90 18 '
        f'{margin_top + plot_height / 2:.2f})">Proportion of entity-labelled tokens</text>',
        "</svg>",
    ])

    output_file.write_text("\n".join(svg_lines), encoding="utf-8")


def write_dataset_shift_figures(stats_file: Path, density_figure: Path, composition_figure: Path) -> None:
    stats_rows = _read_base_ner_stats(stats_file)
    _write_entity_density_svg(density_figure, stats_rows)
    _write_label_composition_svg(composition_figure, stats_rows)
