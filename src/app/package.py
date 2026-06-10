import importlib.util
import os
import subprocess
import sys
from typing import List, Optional


class Package:

    @classmethod
    def install_packages(
        cls,
        pkg: str,
        ver: str,
        args: Optional[List[str]] = None,
        module_name: Optional[str] = None
    ) -> None:
        if sys.prefix == sys.base_prefix:
            raise RuntimeError("Virtual environment is NOT active. We need PIP virtual environment to operate.")

        probe_name = module_name or pkg.replace("-", "_")
        if importlib.util.find_spec(probe_name) is not None:
            return

        package_spec = f"{pkg}=={ver}"
        install_args = list(args or [])
        env = os.environ.copy()
        env["MAX_JOBS"] = "2"
        env["NINJAFLAGS"] = "-j1"

        pip_cmd = [sys.executable, "-m", "pip", "install", package_spec, *install_args]
        uv_cmd = ["uv", "pip", "install", package_spec, *install_args]

        pip_error: Optional[Exception] = None
        try:
            subprocess.check_call(pip_cmd, env=env)
            return
        except (subprocess.CalledProcessError, FileNotFoundError) as exc:
            pip_error = exc

        try:
            subprocess.check_call(uv_cmd, env=env)
        except (subprocess.CalledProcessError, FileNotFoundError) as exc:
            if pip_error is not None:
                raise RuntimeError(
                    f"Failed to install {package_spec} with pip and uv pip."
                ) from pip_error
            raise RuntimeError(
                f"Failed to install {package_spec} with pip and uv pip."
            ) from exc
