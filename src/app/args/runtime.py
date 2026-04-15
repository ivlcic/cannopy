from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass
class ResultPathSet:
    root: Path
    data: Path
    test: Path
    train: Path
    eval: Path


@dataclass
class PathSet:
    root: Path
    tmp: Path
    src: Path
    log: Path
    result: ResultPathSet


@dataclass
class Paths:
    curr_context: str
    curr_script: str
    curr_task: str
    base: PathSet
    task: Path
    context: Path

    def get_std_task_paths(self, task: str) -> ResultPathSet:
        return ResultPathSet(
            root=self.base.root / 'result',
            data=self.base.root / 'result' / 'data' / task,
            test=self.base.root / 'result' / 'test' / task,
            train=self.base.root / 'result' / 'train' / task,
            eval=self.base.root / 'result' / 'eval' / task
        )

    def get_std_ctx_paths(self, task: str) -> ResultPathSet:
        return ResultPathSet(
            root=self.base.root / 'result',
            data=self.base.root / 'result' / 'data' / task / self.curr_context,
            test=self.base.root / 'result' / 'test' / task / self.curr_context,
            train=self.base.root / 'result' / 'train' / task / self.curr_context,
            eval=self.base.root / 'result' / 'eval' / task / self.curr_context
        )

    def get_task_path(self, task: str) -> Path:
        return self.base.root / 'result' / self.curr_script / task

    def get_ctx_path(self, task: str) -> Path:
        return self.base.root / 'result' / self.curr_script / task / self.curr_context

    def get_script_ctx_path(self, script: str, task: str) -> Path:
        return self.base.root / 'result' / script / task / self.curr_context

    def get_script_path(self, script: str) -> Path:
        return self.base.root / 'result' / script / self.curr_task / self.curr_context


@dataclass
class Runtime:
    paths: Paths
    context: str
    script: str
    task: str
    config: List[str]
