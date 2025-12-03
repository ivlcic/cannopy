from dataclasses import dataclass, field
from typing import Dict, Any, List


@dataclass
class DataArguments:
    dataset_name: str = ""
    dataset_urls: List[str] = field(default_factory=list)
    dataset_config_name: str = ""
    label_remap: Dict[str, Dict[Any, Any]] = field(default_factory=dict)
    max_seq_length: int = 512
    overwrite_cache: bool = False
    preprocessing_num_workers: int = 4
    dataset_src_url: str = ""
    dataset_src_start: str = ""
    dataset_src_end: str = ""
    dataset_src_user: str = ""
    dataset_src_query: Dict[str, Any] = field(default_factory=dict)
