"""Load MovieLens tables and map raw IDs onto contiguous indices."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pandas as pd

_RATINGS_COLUMNS = ("user_id", "movie_id", "rating", "timestamp")
_USERS_COLUMNS = ("user_id", "gender", "age", "occupation", "zip_code")
_MOVIES_COLUMNS = ("movie_id", "title", "genres")
_FRAME_COLUMNS = (
    "user_id",
    "movie_id",
    "rating",
    "timestamp",
    "gender",
    "age",
    "occupation",
    "zip_code",
    "title",
    "genres",
)


def read_table(
    path: str | Path,
    columns: Sequence[str],
    dtype: Mapping[str, np.dtype | type | str] | None = None,
) -> pd.DataFrame:
    """Read one MovieLens ``.dat`` file.

    The file has no header. Fields are separated by ``::`` and encoded as
    Latin-1. ``columns`` names those fields in file order. Pass ``dtype`` when
    inference would change a value, such as a zip code that must stay text.
    """
    return pd.read_csv(
        path,
        sep="::",
        engine="python",
        header=None,
        encoding="latin-1",
        names=list(columns),
        dtype=dtype,
    )


def _readonly(values: np.ndarray, dtype: np.dtype | type) -> np.ndarray:
    """Return a 1D copy of ``values`` that rejects item assignment.

    A slice is a view. The copy owns its data, and marking that copy
    read-only makes assignment through the slice raise ``ValueError``.
    """
    array = np.array(values, dtype=dtype, copy=True)
    array.flags.writeable = False
    return array


def _id_lookup(
    values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, Mapping[int, int]]:
    """Map each raw ID to an index, using one ``np.unique`` call.

    Unique sorts the IDs, so index 0 is the smallest raw ID. The inverse is
    the per-row index, and the dict is that same order. The two cannot drift.
    """
    raw_ids, inverse = np.unique(values, return_inverse=True)
    ids = _readonly(raw_ids, raw_ids.dtype)
    indices = _readonly(inverse, np.int64)
    lookup = MappingProxyType(
        dict(zip(ids.tolist(), range(ids.shape[0]), strict=True))
    )
    return ids, indices, lookup


class Dataset:
    """Ratings joined to users and movies, with one stable raw-ID index.

    Indices follow sorted raw ID and include only IDs that appear in
    ``ratings``. Rows stay in ``ratings`` order. The index is built before
    any later split, so every split can share it.
    """

    def __init__(
        self,
        ratings: pd.DataFrame,
        users: pd.DataFrame,
        movies: pd.DataFrame,
    ) -> None:
        merged = ratings.merge(
            users, on="user_id", how="inner", validate="many_to_one"
        ).merge(movies, on="movie_id", how="inner", validate="many_to_one")
        if len(merged) != len(ratings):
            raise ValueError(
                "every rating must match one user and one movie; "
                f"got {len(merged)} merged rows from {len(ratings)} ratings"
            )

        frame = (
            merged.loc[:, list(_FRAME_COLUMNS)].copy().reset_index(drop=True)
        )

        user_ids, user_indices, user_to_index = _id_lookup(
            frame["user_id"].to_numpy()
        )
        movie_ids, movie_indices, movie_to_index = _id_lookup(
            frame["movie_id"].to_numpy()
        )

        self._frame = frame
        self._user_ids = user_ids
        self._movie_ids = movie_ids
        self._user_indices = user_indices
        self._movie_indices = movie_indices
        self._user_to_index = user_to_index
        self._movie_to_index = movie_to_index
        self._ratings = _readonly(frame["rating"].to_numpy(), np.float64)
        self._timestamps = _readonly(frame["timestamp"].to_numpy(), np.int64)

    @classmethod
    def from_dir(cls, data_dir: str | Path) -> Dataset:
        """Read ``ratings.dat``, ``users.dat``, and ``movies.dat`` from ``data_dir``.

        Zip codes are read as strings so a leading zero, or a ZIP+4 code,
        survives.
        """
        directory = Path(data_dir)
        ratings = read_table(directory / "ratings.dat", _RATINGS_COLUMNS)
        users = read_table(
            directory / "users.dat",
            _USERS_COLUMNS,
            dtype={"zip_code": str},
        )
        movies = read_table(directory / "movies.dat", _MOVIES_COLUMNS)
        return cls(ratings, users, movies)

    @property
    def frame(self) -> pd.DataFrame:
        """One row per rating, in file order. This is a copy."""
        return self._frame.copy()

    @property
    def user_to_index(self) -> Mapping[int, int]:
        """Raw user ID to index. Lookup is O(1), and the mapping is read-only."""
        return self._user_to_index

    @property
    def movie_to_index(self) -> Mapping[int, int]:
        """Raw movie ID to index. Unrated movies are left out."""
        return self._movie_to_index

    @property
    def user_ids(self) -> np.ndarray:
        """Raw user ID for each index. Shape ``(n_users,)``."""
        return self._user_ids

    @property
    def movie_ids(self) -> np.ndarray:
        """Raw movie ID for each index. Shape ``(n_movies,)``."""
        return self._movie_ids

    @property
    def user_indices(self) -> np.ndarray:
        """User index of each rating. Shape ``(n_ratings,)``, dtype int64."""
        return self._user_indices

    @property
    def movie_indices(self) -> np.ndarray:
        """Movie index of each rating. Shape ``(n_ratings,)``, dtype int64."""
        return self._movie_indices

    @property
    def ratings(self) -> np.ndarray:
        """Rating of each row. Shape ``(n_ratings,)``, dtype float64."""
        return self._ratings

    @property
    def timestamps(self) -> np.ndarray:
        """Unix timestamp of each rating. Shape ``(n_ratings,)``, dtype int64."""
        return self._timestamps

    @property
    def n_users(self) -> int:
        """Number of distinct users who have a rating."""
        return int(self._user_ids.shape[0])

    @property
    def n_movies(self) -> int:
        """Number of distinct movies that have a rating."""
        return int(self._movie_ids.shape[0])

    def __len__(self) -> int:
        """Number of ratings."""
        return int(self._ratings.shape[0])

    def __repr__(self) -> str:
        return (
            f"Dataset(n_users={self.n_users}, n_movies={self.n_movies}, "
            f"n_ratings={len(self)})"
        )
