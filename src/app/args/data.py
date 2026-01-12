from dataclasses import dataclass, field
from typing import Dict, Any, List


@dataclass
class TranslateModelConfig:
    provider: str = ''
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TranslateModelsConfig:
    default: TranslateModelConfig = field(default_factory=TranslateModelConfig)
    fallback: TranslateModelConfig = field(default_factory=TranslateModelConfig)


@dataclass
class TranslateConfig:
    src_lang: str = 'en'
    lang: str = ''
    prompt: str = ''
    max_payload_threads: int = 5
    max_chars_per_payload: int = 2000
    batch_size: int = 5
    max_batch_threads: int = 5
    models: TranslateModelsConfig = field(default_factory=TranslateModelsConfig)


@dataclass
class ConnectionConfig:
    url: str = ''
    username: str = ''
    password: str = ''


@dataclass
class SourceSelectConfig:
    start: str = ''
    end: str = ''
    subset: str = ''
    query: Dict[str, Any] = field(default_factory=dict)
    filter: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SourceConfigLink:
    url: str = ''
    lang: str = ''


@dataclass
class SourceConfig:
    select: SourceSelectConfig = field(default_factory=SourceSelectConfig)
    conn: ConnectionConfig = field(default_factory=ConnectionConfig)
    lang: str = ''
    links: List[SourceConfigLink] = field(default_factory=list)


@dataclass
class ClusterConfig:
    attributes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SplitConfig:
    train: float = 0.8
    eval: float = 0.1
    test: float = 0.1
    seed: int = 42


@dataclass
class DataArguments:
    dataset_name: str = ''
    version: str = ''
    lang: str = ''
    label_remap: Dict[str, Dict[Any, Any]] = field(default_factory=dict)
    overwrite_cache: bool = False
    preprocessing_num_workers: int = 4
    split: SplitConfig = field(default_factory=SplitConfig)
    subdata_order: List[str] = field(default_factory=list)
    translate: TranslateConfig = field(default_factory=TranslateConfig)
    source: SourceConfig = field(default_factory=SourceConfig)
    cluster: ClusterConfig = field(default_factory=ClusterConfig)
