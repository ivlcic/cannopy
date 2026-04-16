import ast
from pathlib import Path
from types import SimpleNamespace
from typing import Optional


def _load_newsmon_name_helpers():
    source_path = Path('src/data/prepare/newsmon.py')
    module = ast.parse(source_path.read_text(encoding='utf-8'), filename=str(source_path))
    selected_nodes = [
        node for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name in {'get_subset_name', 'get_sidecar_name'}
    ]
    helper_module = ast.Module(body=selected_nodes, type_ignores=[])
    namespace = {
        'DataArguments': object,
        'ModelArguments': object,
        'Optional': Optional,
    }
    exec(compile(helper_module, filename=str(source_path), mode='exec'), namespace)
    return namespace['get_sidecar_name']


def test_get_sidecar_name_includes_split_suffix() -> None:
    get_sidecar_name = _load_newsmon_name_helpers()
    data_args = SimpleNamespace(
        dataset_name='newsmon',
        source=SimpleNamespace(select=SimpleNamespace(subset='newsmon_sl')),
    )
    model_args = SimpleNamespace(short_name='bge-m3')

    assert get_sidecar_name(data_args, model_args) == 'newsmon_sl.bge-m3.npz'
    assert get_sidecar_name(data_args, model_args, 'train') == 'newsmon_sl.bge-m3.train.npz'
