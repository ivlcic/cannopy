import csv
import json
from statistics import mean
from logging import Logger
from pathlib import Path
from typing import Any, Dict, Optional, List

from app.args.data import TranslateConfig
from app.pip import Pip
from ...app.args.data import DataArguments

logger: Logger
paths: Dict[str, Any]
_EVAL_DEPS_READY = False


def _parse_sample(line: str) -> Optional[Dict[str, Any]]:
    line = line.strip()
    if not line:
        return None

    return {'text': line}


def _ensure_eval_deps() -> None:
    global _EVAL_DEPS_READY
    if _EVAL_DEPS_READY:
        return
    logger.info('Installing evaluation packages')
    Pip.install_packages('nltk', '3.9.3')
    Pip.install_packages('sacrebleu', '2.6.0')
    Pip.install_packages('bert_score', '0.3.13')
    logger.info('Installed evaluation packages')
    _EVAL_DEPS_READY = True


def _read_lines(path: Path) -> List[str]:
    rows: List[str] = []
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            sample = _parse_sample(line)
            if sample is not None:
                rows.append(sample['text'])
    return rows


def eval_ref(source_file: Path, trans_file: Path, ref_file: Path, t_cfg: TranslateConfig) -> Dict[str, Any]:
    _ensure_eval_deps()
    # noinspection PyPackageRequirements
    import nltk
    from nltk.translate.meteor_score import meteor_score
    import sacrebleu
    from bert_score import score as bert_score

    # METEOR may require corpora; download if missing.
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)

    source = _read_lines(source_file)
    hyp = _read_lines(trans_file)
    ref = _read_lines(ref_file)
    n = min(len(hyp), len(ref))
    if n == 0:
        raise ValueError(f'No samples to evaluate for {trans_file}')
    if len(hyp) != len(ref):
        logger.warning(
            'Length mismatch (hyp=%d, ref=%d) for %s. Truncating to %d samples.',
            len(hyp), len(ref), trans_file.name, n
        )
    hyp = hyp[:n]
    ref = ref[:n]
    source = source[:n]

    bleu_avg = mean(sacrebleu.sentence_bleu(h, [r]).score / 100.0 for h, r in zip(hyp, ref))
    meteor_avg = mean(meteor_score([r.split()], h.split()) for h, r in zip(hyp, ref))
    chrf_avg = mean(sacrebleu.sentence_chrf(h, [r]).score / 100.0 for h, r in zip(hyp, ref))
    bleu_corpus = sacrebleu.corpus_bleu(hyp, [ref]).score / 100.0
    chrf_corpus = sacrebleu.corpus_chrf(hyp, [ref]).score / 100.0

    # Let bert_score auto-handle multilingual data.
    _, _, f1 = bert_score(hyp, ref, lang=t_cfg.tgt_code, verbose=False)
    bert_f1 = float(f1.mean().item())

    return {
        'file': trans_file.name,
        'samples': n,
        'source_file': source_file.name,
        'reference_file': ref_file.name,
        "BLEU (avg)": float(bleu_avg),
        "METEOR (avg)": float(meteor_avg),
        "CHRF (avg)": float(chrf_avg),
        "BLEU (corpus)": float(bleu_corpus),
        "CHRF (corpus)": float(chrf_corpus),
        "BERT score": bert_f1,
    }


def eval_ref_free(source_file: Path, trans_file: Path) -> Dict[str, Any]:
    return {'file': trans_file.name}


def main(data_args: DataArguments) -> None:
    logger.info('Evaluating MT Slobench translations')
    ds_name = 'mt-slobench'
    translate_ds_dir = paths['base']['data'] / 'translate' / ds_name
    download_ds_dir = paths['base']['data'] / 'download' / ds_name

    t_cfg = data_args.translate
    translation_dir = translate_ds_dir / f'slobench_ensl.{t_cfg.src_code}'
    if not translation_dir.exists() or not translation_dir.is_dir():
        logger.error(
            f'Translation [translate] {ds_name} directory not found: %s', translation_dir
        )
        return

    ref_file = download_ds_dir / f'slobench_ensl.{t_cfg.tgt_code}' / f'slobench_ensl.{t_cfg.tgt_code}.txt'
    if not ref_file.exists() or not ref_file.is_file():
        logger.error(
            f'Reference [translate] {ds_name} directory not found: %s', ref_file
        )
        return

    src_file = download_ds_dir / f'slobench_ensl.{t_cfg.src_code}' / f'slobench_ensl.{t_cfg.src_code}.txt'
    if not src_file.exists() or not src_file.is_file():
        logger.error(
            f'Source [translate] {ds_name} directory not found: %s', src_file
        )
        return

    test_prefix = None  # eval specific model only
    if hasattr(t_cfg, "model"):
        test_prefix = t_cfg.get_base_name()

    target_dir = paths[ds_name]['eval']
    target_dir.mkdir(parents=True, exist_ok=True)
    files: List[Path] = []
    for child in translation_dir.iterdir():
        if not child.is_file() or child.suffix != '.txt':
            continue
        if test_prefix is None:
            files.append(child)
        elif child.name.startswith(test_prefix):
            files.append(child)

    eval_rows: List[Dict[str, Any]] = []
    for trans_file in files:
        logger.info('Evaluating translations from %s...', trans_file)
        ref_scores = eval_ref(src_file, trans_file, ref_file, t_cfg)
        eval_ref_free(src_file, trans_file)
        eval_rows.append(ref_scores)
        logger.info('Evaluated translations from %s...', trans_file)

    if not eval_rows:
        logger.warning('No translation files selected for evaluation in %s', translation_dir)
        return

    model_short_name = t_cfg.model.short_name if hasattr(t_cfg, "model") else "all"
    base_name = f'{ds_name}.{t_cfg.tgt_code}.{model_short_name}'
    out_jsonl = target_dir / f'{base_name}.jsonl'
    out_csv = target_dir / f'{base_name}.csv'

    with out_jsonl.open('w', encoding='utf-8') as f:
        for row in eval_rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')

    csv_columns = [
        'file',
        'samples',
        'source_file',
        'reference_file',
        "BLEU (avg)",
        "METEOR (avg)",
        "CHRF (avg)",
        "BLEU (corpus)",
        "CHRF (corpus)",
        "BERT score",
    ]
    with out_csv.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=csv_columns)
        writer.writeheader()
        writer.writerows(eval_rows)

    logger.info('Wrote evaluation aggregate JSONL: %s', out_jsonl)
    logger.info('Wrote evaluation aggregate CSV: %s', out_csv)
