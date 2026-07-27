import pytest

from src.data.analyze._ner_sdjt.results import compute_croatian_source_ablation


def _result_row(pool_name: str, language: str, f1: float) -> dict[str, float | int | str]:
    return {
        'run_name': pool_name,
        'pool_name': pool_name,
        'budget_pct': 100,
        'language': language,
        'p': f1,
        'p_std': 0.01,
        'r': f1,
        'r_std': 0.01,
        'f1': f1,
        'f1_std': 0.01,
        'acc': f1,
        'acc_std': 0.01,
    }


def test_croatian_source_ablation_computes_matched_deltas() -> None:
    pd = pytest.importorskip('pandas')
    frame = pd.DataFrame.from_records([
        _result_row('multi7-no-hr', 'bg', 0.80),
        _result_row('multi7-plus-hr500k', 'bg', 0.83),
        _result_row('multi7-plus-hr-wikiann', 'bg', 0.78),
        _result_row('multi7-no-hr', 'sl', 0.90),
        _result_row('multi7-plus-hr500k', 'sl', 0.90),
        _result_row('multi7-plus-hr-wikiann', 'sl', 0.89),
    ])

    result = compute_croatian_source_ablation(frame)

    assert list(result['language']) == ['bg', 'sl']
    assert result['ablation_complete'].all()
    assert result['manual_nonnegative_wikiann_negative'].all()

    bg = result[result['language'] == 'bg'].iloc[0]
    assert bg['delta_hr500k_minus_base'] == pytest.approx(0.03)
    assert bg['delta_hr_wikiann_minus_base'] == pytest.approx(-0.02)
    assert bg['delta_hr500k_minus_hr_wikiann'] == pytest.approx(0.05)
    assert bg['best_model'] == 'multi7_plus_hr500k'
