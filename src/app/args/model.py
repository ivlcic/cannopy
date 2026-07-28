from dataclasses import dataclass
from typing import Optional

import torch
from torch.nn import Module


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

    def validate_training_parameter_dtypes(self, model: Module) -> None:
        trainable_dtypes = {
            parameter.dtype
            for parameter in model.parameters()
            if parameter.requires_grad and parameter.is_floating_point()
        }
        if torch.float16 in trainable_dtypes:
            raise ValueError(
                "Raw float16 trainable parameters are unsafe with AdamW. "
                "Load the model with model.dtype=float32 and enable mixed precision "
                "through train.fp16 or train.bf16 instead."
            )
