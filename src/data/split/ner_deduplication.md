# NER sentence deduplication

## Procedure

Deduplication is enabled by `data.split.dedup: true` during the `ner` splitting
step. It runs immediately after the random train-validation-test split and
before the split files are written. The later `ner-sdjt` resampling step uses
these already deduplicated files to construct its experimental pools and
token-budgeted training sets.

For every sample, the duplicate key is computed as:

```python
text = " ".join(sample.tokens)
key = unicodedata.normalize("NFKC", text).casefold()
```

The operations have the following effects:

1. The prepared tokens are joined with a single space.
2. Unicode NFKC normalization makes canonically or compatibly equivalent
   representations comparable. This includes composed versus decomposed
   characters, full-width forms, and some compatibility characters.
3. Unicode `casefold()` removes case distinctions more comprehensively than
   `lower()`.

The procedure does not remove punctuation or diacritics, perform stemming or
lemmatization, transliterate between scripts, resolve typographic variants not
covered by NFKC, or perform fuzzy matching. Corpus name, document ID, sentence
ID, and NER labels are not part of the duplicate key.

Duplicate detection is performed independently within each language key.
Consequently, identical sentences in two different languages are not compared.
The manually curated Croatian data (`hr`) and Croatian WikiANN data
(`hr-wikiann`) are also intentionally treated as separate source keys and are
not deduplicated against each other. Corpora combined under the same language,
such as the four Slovenian corpora, are deduplicated jointly.

Samples are examined in the following priority order:

```text
test -> validation -> training
```

The first occurrence of a normalized sentence is retained and all subsequent
occurrences are removed. Therefore, a test occurrence takes precedence over
the same sentence in validation or training, and a validation occurrence takes
precedence over training. Repetitions within the same split are also removed,
with the first sample in that split's existing order retained.

Labels do not determine whether samples are duplicates. After a textual match
is found, the original label sequences are compared without applying an
experiment-specific label mapping. If they differ, the lower-priority sample
is still removed and the case is additionally counted as a label conflict.
Thus, `label_conflicts` is a subset of `duplicates_removed`. Label
harmonization for the SDJT experiments is performed later by the `ner-sdjt`
resampling task.

The implementation is part of
[`src/data/split/ner.py`](src/data/split/ner.py). Aggregate counts are recorded
in [`ner-dedup-stats.csv`](result/data/analyze/ner/ner-dedup-stats.csv), and
each removed-to-retained pair is recorded in
[`ner-duplicates.csv`](result/data/analyze/ner/ner-duplicates.csv).

## Overall results

| Split | Before | Duplicates removed | After | Duplicate rate | Label conflicts |
|---|---:|---:|---:|---:|---:|
| Test | 33,160 | 1,595 | 31,565 | 4.81% | 18 |
| Validation | 33,145 | 3,033 | 30,112 | 9.15% | 30 |
| Training | 265,193 | 43,597 | 221,596 | 16.44% | 812 |
| **Total** | **331,498** | **48,225** | **283,273** | **14.55%** | **860** |

The duplicate rate is calculated as:

```text
duplicates_removed / before * 100
```

The 860 label conflicts constitute 1.78% of removed duplicate occurrences.

## Duplicate rate by language

The following values combine all source corpora belonging to each language
key.

| Language key | Before | Duplicates removed | After | Duplicate rate | Label conflicts |
|---|---:|---:|---:|---:|---:|
| Bulgarian (`bg`) | 18,333 | 2,559 | 15,774 | 13.96% | 103 |
| Bosnian (`bs`) | 18,810 | 4,369 | 14,441 | 23.23% | 0 |
| Czech (`cs`) | 20,864 | 1,749 | 19,115 | 8.38% | 37 |
| Croatian (`hr`) | 24,794 | 432 | 24,362 | 1.74% | 7 |
| Croatian WikiANN (`hr-wikiann`) | 48,885 | 8,977 | 39,908 | 18.36% | 62 |
| Macedonian (`mk`) | 16,227 | 4,904 | 11,323 | 30.22% | 0 |
| Polish (`pl`) | 20,423 | 2,043 | 18,380 | 10.00% | 63 |
| Russian (`ru`) | 25,141 | 3,627 | 21,514 | 14.43% | 408 |
| Slovak (`sk`) | 50,907 | 12,009 | 38,898 | 23.59% | 43 |
| Slovenian (`sl`) | 48,701 | 1,891 | 46,810 | 3.88% | 108 |
| Albanian (`sq`) | 10,098 | 4,848 | 5,250 | 48.01% | 1 |
| Serbian (`sr`) | 3,891 | 3 | 3,888 | 0.08% | 0 |
| Ukrainian (`uk`) | 24,424 | 814 | 23,610 | 3.33% | 28 |

The highest duplicate rate is found in Albanian WANN (48.01%), followed by
Macedonian WANN (30.22%). Bulgarian BSNLP is high at 13.96%, but comparable to
Czech BSNLP (14.15%) and Russian BSNLP (14.43%).

## Duplicate rate by language and source corpus

Each row aggregates training, validation, and test samples for one
language-source combination. A duplicate is attributed to the corpus of the
removed occurrence; its retained counterpart can belong to the same or another
corpus under the same language key.

| Language key | Source corpus | Before | Duplicates removed | After | Duplicate rate | Label conflicts |
|---|---|---:|---:|---:|---:|---:|
| `bg` | BSNLP | 18,333 | 2,559 | 15,774 | 13.96% | 103 |
| `bs` | WANN | 18,810 | 4,369 | 14,441 | 23.23% | 0 |
| `cs` | BSNLP | 11,947 | 1,690 | 10,257 | 14.15% | 20 |
| `cs` | CNEC | 8,917 | 59 | 8,858 | 0.66% | 17 |
| `hr` | HR500K | 24,794 | 432 | 24,362 | 1.74% | 7 |
| `hr-wikiann` | WikiANN-HR | 48,885 | 8,977 | 39,908 | 18.36% | 62 |
| `mk` | WANN | 16,227 | 4,904 | 11,323 | 30.22% | 0 |
| `pl` | BSNLP | 20,423 | 2,043 | 18,380 | 10.00% | 63 |
| `ru` | BSNLP | 25,141 | 3,627 | 21,514 | 14.43% | 408 |
| `sk` | WANN | 50,907 | 12,009 | 38,898 | 23.59% | 43 |
| `sl` | BSNLP | 17,124 | 1,483 | 15,641 | 8.66% | 100 |
| `sl` | ELEXIS-WSD | 2,024 | 0 | 2,024 | 0.00% | 0 |
| `sl` | SentiCoref | 18,142 | 354 | 17,788 | 1.95% | 3 |
| `sl` | ssj500k | 11,411 | 54 | 11,357 | 0.47% | 5 |
| `sq` | WANN | 10,098 | 4,848 | 5,250 | 48.01% | 1 |
| `sr` | SETimes | 3,891 | 3 | 3,888 | 0.08% | 0 |
| `uk` | BSNLP | 6,085 | 563 | 5,522 | 9.25% | 20 |
| `uk` | NER-UK | 18,339 | 251 | 18,088 | 1.37% | 8 |

## Duplicate rate by source corpus

These are sample-weighted aggregates across all languages using a given source
corpus. Deduplication itself remains language-specific.

| Source corpus | Before | Duplicates removed | After | Duplicate rate | Label conflicts |
|---|---:|---:|---:|---:|---:|
| BSNLP | 99,053 | 11,965 | 87,088 | 12.08% | 714 |
| CNEC | 8,917 | 59 | 8,858 | 0.66% | 17 |
| ELEXIS-WSD | 2,024 | 0 | 2,024 | 0.00% | 0 |
| HR500K | 24,794 | 432 | 24,362 | 1.74% | 7 |
| NER-UK | 18,339 | 251 | 18,088 | 1.37% | 8 |
| SentiCoref | 18,142 | 354 | 17,788 | 1.95% | 3 |
| SETimes | 3,891 | 3 | 3,888 | 0.08% | 0 |
| ssj500k | 11,411 | 54 | 11,357 | 0.47% | 5 |
| WANN | 96,042 | 26,130 | 69,912 | 27.21% | 44 |
| WikiANN-HR | 48,885 | 8,977 | 39,908 | 18.36% | 62 |

## Interpretation

The rates show that duplication is strongly source-dependent. Manually
curated corpora such as CNEC, HR500K, NER-UK, SentiCoref, SETimes, and ssj500k
contain comparatively little exact repetition. BSNLP contains substantial
repetition, particularly in the Bulgarian, Czech, and Russian event-centred
news collections. WANN and Croatian WikiANN exhibit the greatest aggregate
redundancy.

The Bulgarian audit confirms that its high rate is not an artefact of Unicode
normalization: 2,540 of its 2,559 removals are already exact, case-sensitive
sentence matches. NFKC introduces no additional Bulgarian matches, while case
folding adds only 19. The repeated items include syndicated news sentences,
datelines, publisher notices, translator credits, and short location
sentences. Of the Bulgarian removals, 1,286 cross split boundaries and 1,273
occur within a split.

Exact sentence deduplication prevents identical train-validation-test leakage,
but it does not remove paraphrases, near-duplicate passages, or different
sentences from the same source document. The resulting scores should therefore
still be described as evaluation on a random sentence-level split of the
pooled source material, rather than as cross-document or cross-domain
generalization.
