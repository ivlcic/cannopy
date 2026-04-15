import hashlib
import json
import re
import time
from collections import defaultdict
from logging import Logger
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from ...app.args.runtime import Paths
from ...app.args.data import DataArguments, SamplingConfig
from ...app.args.model import ModelArguments
from ...app.helpers import JsonIRHelper
from ...app.token_classifier import EncoderTokenClassifier


logger: Logger
paths: Paths

_len_re = re.compile(r"_len-(\d+)-(\d+|inf)\.jsonl$")
_num_re = re.compile(r"\b\d[\d,./:-]*\b")  # years, dates, decimals, IDs etc.
_url_re = re.compile(r"(https?://\S+|www\.\S+)")
_email_re = re.compile(r"\b[\w.-]+@[\w.-]+\.\w+\b")
_acronym_re = re.compile(r"\b[A-Z]{2,}\b")


def _file_len_bucket(file_name: str) -> str:
    """
    Uses the file naming scheme: *_len-<lo>-<hi>.jsonl, where hi may be 'inf'.
    """
    m = _len_re.search(file_name)
    if not m:
        return "len-unknown"
    lo = int(m.group(1))
    hi = m.group(2)
    return f"len-{lo}-{hi}"


def _query_len_bucket(q: str) -> str:
    # Tokenize cheaply by whitespace. This is fine for stratification.
    n = len(q.strip().split())
    if n <= 3:
        return "q_len-1_3"
    if n <= 7:
        return "q_len-4_7"
    if n <= 15:
        return "q_len-8_15"
    return "q_len-16_plus"


def _flags_for_text(t: str) -> Dict[str, bool]:
    return {
        "has_number": bool(_num_re.search(t)),
        "has_url": bool(_url_re.search(t)),
        "has_email": bool(_email_re.search(t)),
        "has_acronym": bool(_acronym_re.search(t)),
        "has_question_mark": "?" in t,
    }


def _stable_hash_to_u64(s: str, seed: int) -> int:
    # Deterministic across runs and machines.
    h = hashlib.blake2b(digest_size=8)
    h.update(str(seed).encode("utf-8"))
    h.update(b"|")
    h.update(s.encode("utf-8", errors="ignore"))
    return int.from_bytes(h.digest(), "big")


# ----------------------------
# Stratification
# ----------------------------
def _stratum_key(
    dataset_dir: str,
    file_len_bucket: str,
    q: str,
    q_flags: Dict[str, bool],
    ner_has_entity: Optional[bool] = None,
) -> str:
    # Keep stratum names readable and stable.
    parts = [
        f"ds={dataset_dir}",
        file_len_bucket,
        _query_len_bucket(q),
        "q_num=1" if q_flags["has_number"] else "q_num=0",
        "q_ent=1" if (ner_has_entity is True) else "q_ent=0",
        "q_url=1" if q_flags["has_url"] else "q_url=0",
        "q_acr=1" if q_flags["has_acronym"] else "q_acr=0",
        "q_qm=1" if q_flags["has_question_mark"] else "q_qm=0",
    ]
    return "|".join(parts)


def _iter_jsonl(path: Path) -> Iterable[Tuple[int, Dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            obj = JsonIRHelper.read_ir_sample(line, i, path)
            if obj is None:
                continue
            yield i, obj


def _make_sample_id(obj: Dict[str, Any], fallback: str) -> str:
    """
    Prefer stable IDs if present; otherwise hash the query+pos snippet.
    """
    for k in ("id", "qid", "sample_id", "query_id"):
        if k in obj and obj[k]:
            return str(obj[k])
    q = str(obj.get("query", ""))
    pos = obj.get("pos", "")
    # pos could be list/dict/string; keep short & deterministic
    pos_s = json.dumps(pos, ensure_ascii=False)[:200]
    base = f"{q}\n{pos_s}\n{fallback}"
    return hashlib.blake2b(base.encode("utf-8", errors="ignore"), digest_size=16).hexdigest()


def _select_top_k_by_hash(items: List[Tuple[int, str, Dict[str, Any]]], k: int) \
        -> List[Tuple[int, str, Dict[str, Any]]]:
    """
    Deterministically pick the K lowest-hash items (stable sampling).
    Each item: (hash_u64, provenance_str, obj)
    """
    if k <= 0:
        return []
    items.sort(key=lambda x: x[0])
    return items[:k]


def _flush_batch(
    buffer: List[Tuple[str, str, Dict[str, Any], Dict[str, bool]]],
    ner: EncoderTokenClassifier,
    *,
    dataset_dir: str,
    file_len_bucket: str,
    seed: int,
    sample_per_stratum: int,
    stratum_candidates: Dict[str, List[Tuple[int, str, Dict[str, Any]]]],
    stratum_counts: Dict[str, int],
    max_strata: int,
) -> None:
    # NER: determine which queries have at least one entity
    queries = [b[2].get("query", "") for b in buffer]
    ner_flags = [False] * len(queries)
    if ner is not None:
        ner_flags = []
        for query in queries:
            ner_flags.append(True if ner.count_labels(query, none_label="O") > 0 else False)

    for (sid, prov, obj, q_flags), has_ent in zip(buffer, ner_flags):
        sk = _stratum_key(dataset_dir, file_len_bucket, str(obj.get("query", "")), q_flags, has_ent)

        # Optional safety cap to avoid exploding strata due to overly granular keys
        if 0 < max_strata <= len(stratum_candidates) and sk not in stratum_candidates:
            continue

        stratum_counts[sk] += 1

        # Compute stable hash for deterministic top-K selection per stratum
        h = _stable_hash_to_u64(f"{sk}|{sid}", seed=seed)

        # Keep a bounded list: at most sample_per_stratum * 2 candidates,
        # then trim to sample_per_stratum (lowest hashes) for memory safety.
        cand_list = stratum_candidates[sk]
        out_obj = dict(obj)

        cand_list.append((h, prov, out_obj))
        # Periodic trim
        if len(cand_list) > sample_per_stratum * 2:
            cand_list.sort(key=lambda x: x[0])
            del cand_list[sample_per_stratum:]

    buffer.clear()


# ----------------------------
# Main entry
# ----------------------------
# noinspection SpellCheckingInspection
def main(data_args: DataArguments) -> None:
    dataset_name = data_args.dataset_name
    sampling_cfg: SamplingConfig = data_args.sampling
    seed = sampling_cfg.seed
    sample_per_stratum = sampling_cfg.stratification.sample_per_stratum
    max_strata = sampling_cfg.stratification.max_strata
    include_ner = sampling_cfg.stratification.attributes.get("include_ner", True)

    # ner_model_name = "ivlcic/sour-sarma"
    # ner_tagger = EncoderTokenClassifier(
    #    ner_model_name,
    #    ModelArguments(attn_implementation="flash_attention_2", dtype="float16")
    # )

    ner_model_name = "Jean-Baptiste/roberta-large-ner-english"
    ner_tagger = EncoderTokenClassifier(ner_model_name, ModelArguments())

    source_dir = paths.get_ctx_path('prepare')  # paths["base"]["data"] / "prepare" / dataset_name
    if not source_dir.exists():
        logger.error("Source [prepare] %s directory not found: %s", dataset_name, source_dir)
        return

    target_dir = paths.context
    target_dir.mkdir(parents=True, exist_ok=True)

    # Collect all the JSONL files in the 11 directories
    jsonl_files: List[Path] = []
    for ds_dir in sorted([p for p in source_dir.iterdir() if p.is_dir()]):
        for f in sorted(ds_dir.iterdir()):
            if f.is_file() and f.suffix == ".jsonl":
                jsonl_files.append(f)

    if not jsonl_files:
        logger.error("No .jsonl files found under %s", source_dir)
        return

    logger.info("Found %d jsonl files under %s", len(jsonl_files), source_dir)

    # For each stratum, keep a small list of best (lowest-hash) candidates.
    # This is memory-light: at most sample_per_stratum items stored per stratum.
    stratum_candidates: Dict[str, List[Tuple[int, str, Dict[str, Any]]]] = defaultdict(list)
    stratum_counts: Dict[str, int] = defaultdict(int)

    progress_interval = sampling_cfg.attributes.get("progress_interval", 10000)
    total_lines = 0
    total_start = time.monotonic()

    # For NER batching: collect queries in small batches per file.
    for file_path in jsonl_files:
        dataset_dir = file_path.parent.name  # e.g., "MSMARCO", "MIRACL"
        flb = _file_len_bucket(file_path.name)

        logger.info("Scanning %s/%s ...", dataset_dir, file_path.name)
        file_start = time.monotonic()
        file_lines = 0

        buffer: List[Tuple[str, str, Dict[str, Any], Dict[str, bool]]] = []
        # buffer entry: (sample_id, provenance, obj, q_flags)

        for line_no, obj in _iter_jsonl(file_path):
            total_lines += 1
            file_lines += 1
            q = str(obj.get("query", "")).strip()
            if not q:
                continue

            q_flags = _flags_for_text(q)
            sid = _make_sample_id(obj, fallback=f"{file_path.name}:{line_no}")
            prov = f"{dataset_dir}/{file_path.name}:{line_no}"

            buffer.append((sid, prov, obj, q_flags))

            # Process in batches for NER
            if len(buffer) >= max(sampling_cfg.batch_size, 64):
                _flush_batch(
                    buffer,
                    ner_tagger,
                    dataset_dir=dataset_dir,
                    file_len_bucket=flb,
                    seed=seed,
                    sample_per_stratum=sample_per_stratum,
                    stratum_candidates=stratum_candidates,
                    stratum_counts=stratum_counts,
                    max_strata=max_strata,
                )

            if progress_interval and total_lines % progress_interval == 0:
                elapsed = time.monotonic() - total_start
                rate = total_lines / elapsed if elapsed > 0 else 0.0
                logger.info(
                    "Progress: %d lines processed (%.1f lines/s), %d strata",
                    total_lines,
                    rate,
                    len(stratum_candidates),
                )

        if buffer:
            _flush_batch(
                buffer,
                ner_tagger,
                dataset_dir=dataset_dir,
                file_len_bucket=flb,
                seed=seed,
                sample_per_stratum=sample_per_stratum,
                stratum_candidates=stratum_candidates,
                stratum_counts=stratum_counts,
                max_strata=max_strata,
            )
        file_elapsed = time.monotonic() - file_start
        logger.info(
            "Finished %s/%s: %d lines in %.1fs",
            dataset_dir,
            file_path.name,
            file_lines,
            file_elapsed,
        )

    # Materialize final sample and write outputs
    final_rows: List[Dict[str, Any]] = []
    final_meta: List[Dict[str, Any]] = []

    for sk, cand in sorted(stratum_candidates.items(), key=lambda x: x[0]):
        picked = _select_top_k_by_hash(cand, sample_per_stratum)
        for h, prov, obj in picked:
            out = dict(obj)
            out["_stratum"] = sk
            out["_provenance"] = prov
            out["_hash_u64"] = h
            final_rows.append(out)

        final_meta.append(
            {
                "stratum": sk,
                "seen": stratum_counts.get(sk, 0),
                "kept": min(sample_per_stratum, len(cand)),
            }
        )

    out_name = getattr(data_args, "output_name", None)
    if not out_name:
        out_name = f"stratified_sample_seed{seed}_k{sample_per_stratum}.jsonl"

    out_jsonl = target_dir / out_name
    out_stats = target_dir / out_name.replace(".jsonl", ".stats.json")

    with out_jsonl.open("w", encoding="utf-8") as w:
        for row in final_rows:
            w.write(json.dumps(row, ensure_ascii=False) + "\n")

    stats = {
        "dataset_name": dataset_name,
        "seed": seed,
        "sample_per_stratum": sample_per_stratum,
        "use_ner": include_ner,
        "ner_model": ner_model_name if include_ner else None,
        "num_files": len(jsonl_files),
        "num_strata": len(stratum_candidates),
        "total_output_rows": len(final_rows),
        "strata": final_meta,
    }
    with out_stats.open("w", encoding="utf-8") as w:
        json.dump(stats, w, ensure_ascii=False, indent=2)

    total_elapsed = time.monotonic() - total_start
    logger.info(
        "Wrote stratified sample: %s (%d rows). Total lines=%d in %.1fs",
        out_jsonl,
        len(final_rows),
        total_lines,
        total_elapsed,
    )
    logger.info("Wrote stats: %s", out_stats)
