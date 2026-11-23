"""Download the MovieLens 1M dataset into ``data/ml-1m``.

The unpacked ``.dat`` files use a ``::`` delimiter, Latin-1 encoding, and no
header row.

Column order:

- ``movies.dat``: MovieID, Title, Genres
- ``ratings.dat``: UserID, MovieID, Rating, Timestamp
- ``users.dat``: UserID, Gender, Age, Occupation, Zip-code
"""

import hashlib
import shutil
import sys
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
SHA256 = "a6898adb50b9ca05aa231689da44c217cb524e7ebd39d264c56e2832f2c54e20"
DATA_DIR = Path(__file__).resolve().parents[1] / "data"  # works from any cwd
FIELD_COUNTS = {"movies.dat": 3, "ratings.dat": 4, "users.dat": 5}
TIMEOUT_SECONDS = 60


def download(url: str, dest: Path) -> None:
    """Stream ``url`` into ``dest``, creating parent directories if needed."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    with (
        urllib.request.urlopen(url, timeout=TIMEOUT_SECONDS) as response,
        dest.open("wb") as out,
    ):
        shutil.copyfileobj(response, out)


def sha256_of(path: Path) -> str:
    """Return the SHA-256 hex digest of ``path``."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract(zip_path: Path, dest_dir: Path) -> None:
    """Unpack ``zip_path`` into ``dest_dir``."""
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(dest_dir)


def count_rows(path: Path, field_count: int) -> int:
    """Return how many Latin-1 rows ``path`` has.

    Split each row on ``::``. Raise ``ValueError`` when a row does not have
    ``field_count`` fields.
    """
    rows = 0
    with path.open(encoding="latin-1") as handle:
        for line_number, line in enumerate(handle, start=1):
            fields = line.rstrip("\n").split("::")
            if len(fields) != field_count:
                raise ValueError(
                    f"{path}: line {line_number} has {len(fields)} fields, "
                    f"expected {field_count}"
                )
            rows += 1
    return rows


def main() -> int:
    """Download MovieLens 1M when ``data/ml-1m`` is missing.

    Return 0 when the files are already present or the download checks out.
    Return 1 when the download, the hash check, or the zip extract fails.
    """
    dataset_dir = DATA_DIR / "ml-1m"
    if all((dataset_dir / name).is_file() for name in FIELD_COUNTS):
        names = ", ".join(FIELD_COUNTS)
        print(f"{dataset_dir} already has {names}")
        return 0

    zip_path = DATA_DIR / "ml-1m.zip"
    try:
        download(URL, zip_path)
    except urllib.error.URLError as error:
        print(f"download failed: {error}", file=sys.stderr)
        return 1

    digest = sha256_of(zip_path)
    if digest != SHA256:
        zip_path.unlink(missing_ok=True)
        print(
            f"SHA-256 mismatch: expected {SHA256}, got {digest}",
            file=sys.stderr,
        )
        return 1

    try:
        extract(zip_path, DATA_DIR)
    except zipfile.BadZipFile as error:
        print(f"extract failed: {error}", file=sys.stderr)
        return 1
    finally:
        zip_path.unlink(missing_ok=True)

    for name, field_count in FIELD_COUNTS.items():
        print(f"{name}: {count_rows(dataset_dir / name, field_count)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
