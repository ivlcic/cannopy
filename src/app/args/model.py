from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelArguments:
    batch_size: int = 32
    model_name_or_path: str = None
    config_name: str = None
    tokenizer_name: str = None
    cache_dir: str = None
    short_name: str = None
    max_seq_length: int = 512
    truncate_dim: Optional[int] = None
    use_auth_token: bool = False
    attn_implementation: str = None
    dtype: str = None
    classifier_dropout: Optional[float] = None
