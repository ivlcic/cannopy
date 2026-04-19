from __future__ import annotations

import argparse
import importlib
import importlib.util
import inspect
import logging
import logging.config
from dataclasses import fields, is_dataclass, replace
from logging import Logger
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union, get_args, get_origin

import yaml
from transformers import HfArgumentParser, TrainingArguments

from app.args.data import DataArguments
from app.args.model import ModelArguments
from app.args.runtime import Paths, PathSet, ResultPathSet, Runtime


# ---------------------------
# Discovery utilities
# ---------------------------

def _project_paths() -> Dict[str, Path]:
    # src/app/entrypoint.py -> src -> repo
    src_dir = Path(__file__).resolve().parents[1]
    repo = src_dir.parent
    return {
        "repo": repo,
        "src": src_dir,
        "conf": repo / "conf",
        "tmp": repo / "tmp",
        "result": repo / "result",
        "data": repo / "result" / "data"
    }


def _is_pkg_dir(p: Path) -> bool:
    return p.is_dir() and (p / "__init__.py").exists()


def _list_subactions(src: Path, script: str) -> List[str]:
    base = src / script
    if not base.exists():
        return []
    items = []
    for child in base.iterdir():
        if child.is_dir() and any(child.glob("*.py")):
            items.append(child.name)
        else:
            for sub_child in child.iterdir():
                if _is_pkg_dir(sub_child):
                    items.append(child.name)
                    break
    return sorted(items)


def list_names(src: Path, script: str, sub_action: str) -> List[str]:
    base = src / script / sub_action
    if not base.exists():
        return []
    names = []
    for child in base.iterdir():
        if child.is_file() and child.suffix == ".py" and child.stem != "__init__":
            names.append(child.stem)
        elif _is_pkg_dir(child):
            names.append(child.name)
    return sorted(names)


def _module_exists(module_path: str, package: Union[str, None] = None) -> bool:
    return importlib.util.find_spec(module_path, package) is not None


# ---------------------------
# YAML loading and merge
# ---------------------------

def _deep_merge(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(a)
    for k, v in b.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _load_yaml_if_exists(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML at {path} must be a mapping at top level")
    return data


def _resolve_config_stack(conf_dir: Path, task: str, context: str, extra_confs: List[str]) \
        -> List[Path]:
    # Follow the exact order:
    # Accept either bare names or .yaml filenames in -c
    def norm(x: str) -> str:
        return x if x.endswith(".yaml") else f"{x}.yaml"

    ordered = [norm(context)] + [norm(x) for x in extra_confs]

    paths: List[Path] = []

    # load model confs
    for fn in ordered:
        paths.append(conf_dir / 'model' / fn)

    for fn in ordered:
        paths.append(conf_dir / 'model' / task / fn)

    for fn in ordered:
        paths.append(conf_dir / 'model' / task / context / fn)

    for fn in ordered:
        paths.append(conf_dir / 'model' / context / fn)

    for fn in ordered:
        paths.append(conf_dir / 'model' / context / task / fn)

    # load data confs
    for fn in ordered:
        paths.append(conf_dir / 'data' / context / fn)

    for fn in ordered:
        paths.append(conf_dir / 'data' / context / task / fn)

    # load task confs
    for fn in ordered:
        paths.append(conf_dir / 'task' / task / fn)

    for fn in ordered:
        paths.append(conf_dir / 'task' / task / context / fn)

    return paths


def _load_and_merge_configs(conf_dir: Path, task: str, context: str, extra_confs: List[str]) \
        -> Tuple[Dict[str, Any], List[Path]]:
    context = context or task  # allow missing context; but keep task as a default context if needed
    files = _resolve_config_stack(conf_dir, task, context, extra_confs)
    merged: Dict[str, Any] = {}
    loaded_files: List[Path] = []
    for p in files:
        d = _load_yaml_if_exists(p)
        if d:
            merged = _deep_merge(merged, d)
            loaded_files.append(p)
    return merged, loaded_files


def _apply_cli_overrides(merged: Dict[str, Any], overrides: List[str]) -> Dict[str, Any]:
    updated = dict(merged)
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Invalid override {override!r}. Expected dotted.path=value.")
        dotted_path, raw_value = override.split("=", 1)
        keys = [part.strip() for part in dotted_path.split(".") if part.strip()]
        if not keys:
            raise ValueError(f"Invalid override {override!r}. Expected dotted.path=value.")
        value = yaml.safe_load(raw_value)
        cursor = updated
        for key in keys[:-1]:
            next_value = cursor.get(key)
            if not isinstance(next_value, dict):
                next_value = {}
                cursor[key] = next_value
            cursor = next_value
        cursor[keys[-1]] = value
    return updated


# ---------------------------
# Hugging Face args parsing
# ---------------------------

def _split_for_dataclasses(merged: Dict[str, Any]) \
        -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    # Try nested keys first; else fall back by pulling known fields to each dataclass
    model_keys = {f.name for f in fields(ModelArguments)}
    data_keys = {f.name for f in fields(DataArguments)}
    train_keys = {f.name for f in fields(TrainingArguments)}  # large set

    model_dict = dict(merged.get("model", {}))
    data_dict = dict(merged.get("data", {}))
    train_dict = dict(merged.get("train", {}))
    other = dict(merged)

    # Pull top-level fields into the right dataclass maps if present
    for k in list(merged.keys()):
        if k in model_keys and k not in model_dict:
            model_dict[k] = merged[k]
            other.pop(k, None)
        elif k in data_keys and k not in data_dict:
            data_dict[k] = merged[k]
            other.pop(k, None)
        elif k in train_keys and k not in train_dict:
            train_dict[k] = merged[k]
            other.pop(k, None)

    # noinspection PyTypeChecker
    return model_dict, data_dict, train_dict, other


def _is_dataclass_type(t) -> bool:
    try:
        return inspect.isclass(t) and is_dataclass(t)
    except TypeError:
        return False


def _coerce_value(f_type, val):
    if _is_dataclass_type(f_type) and isinstance(val, dict):
        return _coerce_dataclass(f_type, val)

    origin = get_origin(f_type)
    args = get_args(f_type)

    if origin is Union:
        nested = next((a for a in args if _is_dataclass_type(a)), None)
        if nested and isinstance(val, dict):
            return _coerce_dataclass(nested, val)

    if origin in (list, List):
        inner = args[0] if args else None
        if _is_dataclass_type(inner):
            return [(_coerce_dataclass(inner, x) if isinstance(x, dict) else x) for x in val]

    if origin in (dict, Dict) and len(args) == 2:
        key_t, val_t = args
        if _is_dataclass_type(val_t) and isinstance(val, dict):
            return {k: _coerce_dataclass(val_t, v) if isinstance(v, dict) else v for k, v in val.items()}

    return val


def _coerce_dataclass(cls, data: Dict[str, Any]):
    kwargs = {}
    for f in fields(cls):
        if f.name not in data:
            continue
        val = data[f.name]
        kwargs[f.name] = _coerce_value(f.type, val)
    return cls(**kwargs)


def _coerce_nested_dataclasses(obj):
    if not is_dataclass(obj):
        return obj
    updates = {}
    for f in fields(obj):
        val = getattr(obj, f.name)
        coerced = _coerce_value(f.type, val)
        if is_dataclass(coerced):
            coerced = _coerce_nested_dataclasses(coerced)
        elif isinstance(coerced, list) and coerced:
            new_list = []
            changed = False
            for item in coerced:
                new_item = _coerce_nested_dataclasses(item) if is_dataclass(item) else item
                changed = changed or (new_item is not item)
                new_list.append(new_item)
            if changed:
                coerced = new_list
        elif isinstance(coerced, dict) and coerced:
            new_dict = {}
            changed = False
            for k, v in coerced.items():
                new_v = _coerce_nested_dataclasses(v) if is_dataclass(v) else v
                changed = changed or (new_v is not v)
                new_dict[k] = new_v
            if changed:
                coerced = new_dict
        if coerced is not val:
            updates[f.name] = coerced
    if updates:
        return replace(obj, **updates)
    return obj


def _parse_hf_args(merged: Dict[str, Any]) -> Tuple[ModelArguments, DataArguments, TrainingArguments, Dict[str, Any]]:
    model_dict, data_dict, train_dict, extras = _split_for_dataclasses(merged)
    parser = HfArgumentParser((ModelArguments, DataArguments))
    model_args, data_args = parser.parse_dict({**model_dict, **data_dict}, allow_extra_keys=True)
    model_args = _coerce_nested_dataclasses(model_args)
    data_args = _coerce_nested_dataclasses(data_args)
    training_args = TrainingArguments(**train_dict)
    return model_args, data_args, training_args, extras


# ---------------------------
# Logging
# ---------------------------

def _config_logger(args, script: str, path: Path, level: str = "INFO") -> Tuple[Logger, str]:
    cfg_names = [Path(c).stem for c in args.config] if args.config else []
    postfix = '.'.join(cfg_names) if cfg_names else ''
    logger_name = f"{script}.{args.task}"
    logger_file = f"{script}_{args.task}"
    run_name = ''
    if args.context:
        logger_name += f".{args.context}"
        logger_file += f"_{args.context}"
        run_name += f"{args.context}"
    if cfg_names:
        logger_name += f".{postfix}"
        logger_file += f"_{postfix}"
        run_name += f".{postfix}"

    logger_file += '.log'

    log_cfg = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {"format": "%(asctime)s | %(name)s | %(levelname)s | %(message)s"}
        },
        "handlers": {
            "console": {"class": "logging.StreamHandler", "level": level, "formatter": "default"},
            "file": {
                "class": "logging.FileHandler",
                "level": level,
                "formatter": "default",
                "filename": str(path / logger_file),
                "encoding": "utf-8",
            },
        },
        "loggers": {
            "": {"handlers": ["console"], "level": level, "propagate": True},
            logger_name: {"handlers": ["console", "file"], "level": level, "propagate": False},
        },
    }

    logging.config.dictConfig(log_cfg)
    return logging.getLogger(logger_name), run_name


# ---------------------------
# Runner
# ---------------------------

def _build_parser(argv: List[str], script: str, src: Path) -> argparse.ArgumentParser:
    dashes = any("-" in arg[1:-1] for arg in argv if len(arg) >= 3)

    parser = argparse.ArgumentParser(prog=script, description=f"{script} actions")
    subparsers = parser.add_subparsers(dest="task", metavar="task", required=True)

    subactions = _list_subactions(src, script)
    for sa in subactions:
        p = subparsers.add_parser(sa, help=f"{script} {sa}")
        original_names = list_names(src, script, sa)
        names = [n.replace('_', '-') for n in original_names] if dashes else original_names.copy()

        # name optional; validate choices if provided
        p.add_argument(
            "context", nargs="?", choices=names if names else None,
            help="optional context python module under a specific task"
        )
        p.add_argument(
            "func_name", nargs="?", default=None,
            help="optional function name to call inside the resolved module; defaults to main"
        )
        p.add_argument(
            "-c", "--config", action="append", default=[], metavar="CONF.yaml",
            help="config file(s), order matters"
        )
        p.add_argument(
            "-s", "--set", action="append", default=[], metavar="PATH=VALUE",
            help="override merged config values, e.g. -s data.attributes.run_name=multi8"
        )
    return parser


def _ensure_dirs(root: Path, script: str, task: str, context: str) -> Paths:
    p = Paths(
        curr_context=context,
        curr_script=script,
        curr_task=task,
        base=PathSet(
            root=root,
            tmp=root / 'tmp',
            src=root / 'src',
            log=root / 'log',
            result=ResultPathSet(
                root=root / 'result',
                data=root / 'result' / 'data',
                test=root / 'result' / 'test',
                train=root / 'result' / 'train',
                eval=root / 'result' / 'eval'
            )
        ),
        task=root / 'result' / script / task,
        context=root / 'result' / script / task / context
    )

    for path in (
            p.base.tmp,
            p.base.log,
            p.base.result.root,
            p.base.result.data,
            p.base.result.test,
            p.base.result.train,
            p.base.result.eval):
        path.mkdir(parents=True, exist_ok=True)
    p.context.mkdir(parents=True, exist_ok=True)
    return p


def _inject_module_globals(module, g: Dict[str, Any]) -> None:
    for k, v in g.items():
        setattr(module, k, v)


def _call_module(script: str, task: str, context: str | None, func_name: str | None,
                 module_globals: Dict[str, Any]) -> int:
    # Try script.task.context first if context present and module exists
    # noinspection PyTypeChecker
    base_dir = os.path.basename(os.path.dirname(os.path.dirname(__file__)))
    fn = None
    mod = None
    target_func_name = func_name or "main"
    if context:
        mod_name = f"{base_dir}.{script}.{task}.{context}".replace('-', '_')
        if _module_exists(mod_name):
            mod = importlib.import_module(mod_name)
            fn = getattr(mod, target_func_name, None)
        else:
            pkg_path = f"{script}.{task}".replace('-', '_')
            if _module_exists(pkg_path, None):
                mod = importlib.import_module(pkg_path)
                fn = getattr(mod, func_name or context, None)
    else:
        pkg_path = f"{base_dir}.{script}.{task}".replace('-', '_')
        if _module_exists(pkg_path):
            mod = importlib.import_module(pkg_path)
            fn = getattr(mod, target_func_name, None)

    if mod is not None and fn is not None and callable(fn):
        # inject module globals
        for k, v in module_globals.items():
            setattr(mod, k, v)

        sig = inspect.signature(fn)
        kwargs = {k: v for k, v in module_globals.items() if k in sig.parameters}
        # If the target takes no params, this becomes an empty dict
        # noinspection PyCallingNonCallable
        return fn(**kwargs)

    raise ImportError(f"No module to execute for {script} {task} {context or ''} {func_name or ''}".strip())


def main(argv: List[str]) -> int:
    script = os.environ.get("APP_SCRIPT")
    if not script:
        # Fallback: derive from argv[0] like 'data'
        script = Path(sys.argv[0]).name

    paths = _project_paths()
    parser = _build_parser(argv, script, paths["src"])
    args = parser.parse_args(argv)

    # Prepare directories and runtime vars
    src_dir = Path(__file__).resolve().parents[1]
    repo = src_dir.parent
    runtime = Runtime(
        paths=_ensure_dirs(repo, script, args.task, args.context),
        script=script,
        task=args.task,
        context=args.context,
        func_name=args.func_name,
        config=[Path(c).stem for c in args.config] if args.config else [],
    )

    # Load and merge config
    merged_cfg, loaded = _load_and_merge_configs(paths["conf"], args.task, args.context, args.config)
    merged_cfg = _apply_cli_overrides(merged_cfg, args.set)

    # Hugging Face args
    model_args, data_args, training_args, extras = _parse_hf_args(merged_cfg)

    # Logging
    logger, run_name = _config_logger(args, script, runtime.paths.base.log)
    logger.info("Loaded config files: %s", [str(x) for x in loaded])
    if args.set:
        logger.info("Applied CLI overrides: %s", args.set)

    # Prepare globals for the module
    module_globals = {
        "run_args": runtime,
        "model_args": model_args,
        "data_args": data_args,
        "train_args": training_args,
        "extra_args": extras,
        "paths": runtime.paths,
        "logger": logger,
    }

    # Execute
    return _call_module(script, args.task, args.context, args.func_name, module_globals)


# Keep imports needed by main at bottom to avoid circular
import os   # noqa: E402
import sys  # noqa: E402


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
