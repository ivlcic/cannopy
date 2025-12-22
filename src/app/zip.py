import pyzipper
import shutil
import gzip
import tarfile

from pathlib import Path
from typing import Optional, List
from ..app.common import PathLike


class Zip:

    @staticmethod
    def _safe_extract_zipfile(zf: pyzipper.AESZipFile, target_dir: Path) -> None:
        """
        Safely extract all files from a pyzipper ZipFile, preventing path traversal.
        """
        target_dir = target_dir.resolve()

        for member in zf.namelist():
            member_path = Path(member)
            resolved_path = (target_dir / member_path).resolve()

            # prevent path traversal (e.g., ../../etc/passwd)
            if not str(resolved_path).startswith(str(target_dir)):
                raise RuntimeError(f'Unsafe zip entry detected: {member!r}')

            if member.endswith('/'):
                resolved_path.mkdir(parents=True, exist_ok=True)
            else:
                resolved_path.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(member) as source, resolved_path.open('wb') as target:
                    shutil.copyfileobj(source, target)

    @staticmethod
    def extract(zip_path: PathLike, output_dir: PathLike, password: Optional[str] = None) -> Path:
        """
        Extracts a zip (ZipCrypto or AES) using pyzipper only, safely.
        """
        zip_path = Path(zip_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        with pyzipper.AESZipFile(zip_path) as zf:
            if password:
                zf.pwd = password.encode()
            Zip._safe_extract_zipfile(zf, output_dir)
        return output_dir.resolve()

    @staticmethod
    def ungzip(path: Path) -> Optional[Path]:
        if path.suffix != '.gz':
            return None

        target = path.with_suffix('')
        tmp = target.with_suffix(target.suffix + '.part')
        with gzip.open(path, 'rb') as src, tmp.open('wb') as dst:
            # noinspection PyTypeChecker
            shutil.copyfileobj(src, dst)
        tmp.replace(target)
        path.unlink(missing_ok=True)
        return target

    @staticmethod
    def untar(path: Path) -> List[Path]:
        out_dir = path.parent

        with tarfile.open(path, mode="r:*") as tf:
            roots = set()

            for m in tf.getmembers():
                name = (m.name or "").lstrip("./")
                if not name:
                    continue

                root = name.split("/", 1)[0]

                # Only count as a "root dir" if it is a dir entry OR it has children (implied dir)
                is_explicit_root_dir = m.isdir() and name.rstrip("/") == root
                is_implied_dir = "/" in name
                if is_explicit_root_dir or is_implied_dir:
                    roots.add(root)

            tf.extractall(path=out_dir)

        return sorted((out_dir / r) for r in roots)
