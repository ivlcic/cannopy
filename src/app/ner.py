from dataclasses import dataclass
from typing import ClassVar, Dict, List, Tuple


@dataclass(frozen=True)
class NerSample:
    NER_CSV_COLUMNS: ClassVar[List[str]] = [
        'sentence',
        'labels',
        'corpus_name',
        'doc_id',
        'sent_id',
    ]
    CORE_ENTITY_TYPES: ClassVar[Tuple[str, ...]] = ("PER", "ORG", "LOC")

    tokens: List[str]
    labels: List[str]
    corpus_name: str = ''
    doc_id: str = ''
    sent_id: str = ''

    @classmethod
    def from_csv_row(cls, row: Dict[str, str | None]) -> 'NerSample':
        sentence = row.get('sentence') or ''
        labels = row.get('labels') or ''
        return cls(
            tokens=sentence.split(' ') if sentence else [],
            labels=labels.split(' ') if labels else [],
            corpus_name=row.get('corpus_name') or '',
            doc_id=row.get('doc_id') or '',
            sent_id=row.get('sent_id') or '',
        )

    @classmethod
    def harmonize_label(cls, label: str) -> str:
        value = label.strip()
        if not value or value.upper() == "O":
            return "O"
        if "-" not in value:
            entity = value.upper()
            return f"B-{entity}" if entity in cls.CORE_ENTITY_TYPES else "O"
        prefix, entity = value.split("-", 1)
        prefix = prefix.upper()
        entity = entity.upper()
        if prefix in {"S", "U"}:
            prefix = "B"
        elif prefix in {"E", "L"}:
            prefix = "I"
        if prefix not in {"B", "I"}:
            return "O"
        if entity not in cls.CORE_ENTITY_TYPES:
            return "O"
        return f"{prefix}-{entity}"

    def to_csv_row(self) -> Dict[str, str]:
        values = [
            ' '.join(self.tokens),
            ' '.join(self.labels),
            self.corpus_name,
            self.doc_id,
            self.sent_id,
        ]
        return dict(zip(self.NER_CSV_COLUMNS, values, strict=True))
