import requests

from pathlib import Path
from app.common import PathLike


class Downloader:

    @staticmethod
    def download(url: str, dest: PathLike, chunk_size: int = 8192) -> Path:
        """
        Download a file from `url` and save it to `dest` safely.
        """
        if not dest.suffix:
            for splitter in ['-o', 'file=', 'files=', 'filename=']:
                if not splitter in url:
                    continue
                parts = url.split(splitter)
                if ' ' in url:
                    url = parts[0].strip()

                filename = parts[1].strip()
                dest = dest / filename

        dest_path = Path(dest)
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = dest_path.with_suffix(dest_path.suffix + ".part")

        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with tmp_path.open("wb") as f:
                for chunk in r.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)

        tmp_path.replace(dest_path)
        return dest_path
