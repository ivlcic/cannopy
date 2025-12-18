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


def _resolve_config_stack(conf_dir: Path, script: str, sub_action: str, name: str, extra_confs: List[str]) \
        -> List[Path]:
    # Follow the exact order:
    # [name].yaml, each -c, then repeat in conf/{script}/, then conf/{script}/{sub_action}/,
    # then conf/{script}/{sub_action}/{name}
    # Accept either bare names or .yaml filenames in -c
    def norm(x: str) -> str:
        return x if x.endswith(".yaml") else f"{x}.yaml"

    ordered = [norm(name)] + [norm(x) for x in extra_confs]

    paths: List[Path] = []

    # root conf
    for fn in ordered:
        paths.append(conf_dir / fn)

    # conf/{script}
    for fn in ordered:
        paths.append(conf_dir / script / fn)

    # conf/{script}/{sub_action}
    for fn in ordered:
        paths.append(conf_dir / script / sub_action / fn)

    # conf/{script}/{sub_action}/{name}
    for fn in ordered:
        paths.append(conf_dir / script / sub_action / name / fn)

    return paths


def _load_and_merge_configs(conf_dir: Path, script: str, sub_action: str, name: str, extra_confs: List[str]) \
        -> Tuple[Dict[str, Any], List[Path]]:
    name = name or sub_action  # allow missing name; but keep subaction as a default name if needed
    files = _resolve_config_stack(conf_dir, script, sub_action, name, extra_confs)
    merged: Dict[str, Any] = {}
    loaded_files: List[Path] = []
    for p in files:
        d = _load_yaml_if_exists(p)
        if d:
            merged = _deep_merge(merged, d)
            loaded_files.append(p)
    return merged, loaded_files


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
    postfix = '.'.join(cfg_names[0]) if cfg_names else ''
    logger_name = f"{script}.{args.sub_action}"
    logger_file = f"{script}_{args.sub_action}"
    run_name = ''
    if args.name:
        logger_name += f".{args.name}"
        logger_file += f"_{args.name}"
        run_name += f"{args.name}"
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

def _build_parser(script: str, src: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=script, description=f"{script} actions")
    subparsers = parser.add_subparsers(dest="sub_action", metavar="sub_action", required=True)

    subactions = _list_subactions(src, script)
    for sa in subactions:
        p = subparsers.add_parser(sa, help=f"{script} {sa}")
        names = list_names(src, script, sa)
        # name optional; validate choices if provided
        p.add_argument(
            "name", nargs="?", choices=names if names else None,
            help="optional name module under sub-action"
        )
        p.add_argument(
            "-c", "--config", action="append", default=[], metavar="CONF.yaml",
            help="config file(s), order matters"
        )
    return parser


def _ensure_dirs(paths: Dict[str, Path], sub_action: str, name: str) -> Dict[str, Dict[str, Path]]:
    run = {
        'base': {
            "tmp": paths["repo"] / "tmp",
            "log": paths["repo"] / "log",
            "result": paths["repo"] / "result",
            "data": paths["repo"] / "result" / "data",
        }
    }

    if sub_action:
        run[sub_action] = {
            "tmp": paths["repo"] / "tmp" / sub_action,
            "result": paths["repo"] / "result" / sub_action,
            "data": paths["repo"] / "result" / "data" / sub_action,
        }

    if name and sub_action != name:
        run[name] = {
            "tmp": paths["repo"] / "tmp" / name,
            "result": paths["repo"] / "result" / name,
            "data": paths["repo"] / "result" / "data" / name,
        }

    for p in run['base'].values():
        p.mkdir(parents=True, exist_ok=True)
    return run


def _inject_module_globals(module, g: Dict[str, Any]) -> None:
    for k, v in g.items():
        setattr(module, k, v)


def _call_module(script: str, sub_action: str, name: str | None, module_globals: Dict[str, Any]) -> int:
    # Try verb.subaction.name first if name present and module exists
    base_dir = os.path.basename(os.path.dirname(os.path.dirname(__file__)))
    fn = None
    mod = None
    if name:
        mod_name = f"{base_dir}.{script}.{sub_action}.{name}"
        if _module_exists(mod_name):
            mod = importlib.import_module(mod_name)
            fn = getattr(mod, "main", None)
        else:
            pkg_path = f"{script}.{sub_action}"
            if _module_exists(pkg_path, None):
                mod = importlib.import_module(pkg_path)
                fn = getattr(mod, name, None)
    else:
        pkg_path = f"{base_dir}.{script}.{sub_action}"
        if _module_exists(pkg_path):
            mod = importlib.import_module(pkg_path)
            fn = getattr(mod, "main", None)

    if mod is not None and fn is not None and callable(fn):
        # inject module globals
        for k, v in module_globals.items():
            setattr(mod, k, v)

        sig = inspect.signature(fn)
        kwargs = {k: v for k, v in module_globals.items() if k in sig.parameters}
        # If the target takes no params, this becomes an empty dict
        # noinspection PyCallingNonCallable
        return fn(**kwargs)

    raise ImportError(f"No module to execute for {script} {sub_action} {name or ''}")


def main(argv: List[str]) -> int:
    script = os.environ.get("APP_SCRIPT")
    if not script:
        # Fallback: derive from argv[0] like 'data'
        script = Path(sys.argv[0]).name

    paths = _project_paths()
    parser = _build_parser(script, paths["src"])
    args = parser.parse_args(argv)

    cfg_names = [Path(c).stem for c in args.config] if args.config else []
    run = {
        "name": args.name or '',
        "script": script,
        "action": args.sub_action,
        "config": cfg_names,
    }

    # Load and merge config
    merged_cfg, loaded = _load_and_merge_configs(paths["conf"], script, args.sub_action, args.name, args.config)

    # Prepare directories
    run_dirs = _ensure_dirs(paths, args.sub_action, args.name)

    # Hugging Face args
    model_args, data_args, training_args, extras = _parse_hf_args(merged_cfg)

    # Logging
    logger, run_name = _config_logger(args, script, run_dirs['base']['log'])
    logger.info("Loaded config files: %s", [str(x) for x in loaded])

    # Prepare globals for the module
    module_globals = {
        "run_args": run,
        "model_args": model_args,
        "data_args": data_args,
        "train_args": training_args,
        "extra_args": extras,
        "paths": run_dirs,
        "logger": logger,
    }

    # Execute
    return _call_module(script, args.sub_action, args.name, module_globals)


# Keep imports needed by main at bottom to avoid circular
import os   # noqa: E402
import sys  # noqa: E402


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
