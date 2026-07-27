from dataclasses import dataclass, field
from typing import Dict, Any, List


@dataclass
class TranslateModelConfig:
    short_name: str = ''
    provider: str = ''
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TranslateConfig:
    src_code: str = 'en'
    tgt_code: str = ''
    src_lang: str = 'English'
    tgt_lang: str = ''
    prompt: str = ''
    max_payload_threads: int = 5
    max_chars_per_payload: int = 2000
    batch_size: int = 5
    max_batch_threads: int = 5
    model: TranslateModelConfig = field(default_factory=TranslateModelConfig)
    attributes: Dict[str, Any] = field(default_factory=dict)

    def get_base_name(self):
        parameters = self.model.parameters
        model_name = self.model.short_name
        if model_name:
            model_name = f'.{model_name}'
        temp = ''
        if 'temperature' in parameters:
            temp = f'{parameters["temperature"]:.2f}'.replace('.', '_')
            temp = f'.t={temp}'
        top_p = ''
        if 'top_p' in parameters:
            top_p = f'{parameters["top_p"]:.2f}'.replace('.', '_')
            top_p = f'.p={top_p}'
        return f'{self.tgt_code}{model_name}{temp}{top_p}'


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
    attributes: Dict[str, Any] = field(default_factory=dict)


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
class SamplingStratificationConfig:
    sample_per_stratum: int = 100
    max_strata: int = 0  # 0 => no cap
    attributes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SamplingConfig:
    seed: int = 2611
    dedup: bool = False
    batch_size: int = 64
    stratification: SamplingStratificationConfig = field(default_factory=SamplingStratificationConfig)
    attributes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DataArguments:
    dataset_name: str = ''
    version: str = ''
    lang: str = ''
    label_remap: Dict[str, Dict[Any, Any]] = field(default_factory=dict)
    label_remap_exact: Dict[str, Dict[Any, bool]] = field(default_factory=dict)
    overwrite_cache: bool = False
    preprocessing_num_workers: int = 4
    split: SplitConfig = field(default_factory=SplitConfig)
    subdata_order: List[str] = field(default_factory=list)
    translate: TranslateConfig = field(default_factory=TranslateConfig)
    source: SourceConfig = field(default_factory=SourceConfig)
    cluster: ClusterConfig = field(default_factory=ClusterConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    attributes: Dict[str, Any] = field(default_factory=dict)
