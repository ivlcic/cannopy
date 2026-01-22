import os
import sys
import subprocess
import importlib.util
from typing import Optional, List


class Pip:

    @classmethod
    def install_packages(cls, pkg: str, ver: str, args: Optional[List[str]] = None):
        if sys.prefix != sys.base_prefix:
            print(f"Running inside venv: {sys.prefix}")
        else:
            raise RuntimeError("Virtual environment is NOT active")

        if importlib.util.find_spec(pkg) is None:
            exec_cl = [sys.executable, "-m", "pip", "install", f'{pkg}=={ver}']
            if args:
                exec_cl.extend(args)
            env = os.environ.copy()
            env["MAX_JOBS"] = "2"  # limit parallel build jobs
            env["NINJAFLAGS"] = "-j1"    # optional: also cap ninja
            subprocess.check_call(exec_cl, env=env)
