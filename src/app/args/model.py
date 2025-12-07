from dataclasses import dataclass

@dataclass
class ModelArguments:
    model_name_or_path: str = ''
    config_name: str = ''
    tokenizer_name: str = ''
    cache_dir: str = ''
    short_name: str = ''
    max_seq_length: int = 512
    use_auth_token: bool = False