"""Recommender contract and the index checks every model shares."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Self

import numpy as np

from rec_engine.dataset import Dataset


def check_indices(
    indices: np.ndarray | int,
    size: int,
    *,
    name: str,
    match_shape: tuple[int, ...] | None = None,
    scalar: bool = False,
    nonempty: bool = False,
) -> np.ndarray:
    """Return ``indices`` after rejecting a bad shape or an illegal index.

    ``indices`` holds dataset indices, not raw ids. ``size`` is the length
    of that axis: ``n_users``, ``n_movies``, or the number of ratings.

    By default ``indices`` must be an integer array of shape ``(n,)``.
    Pass ``match_shape`` so a second array must share the first array's
    shape, which is how ``predict`` rejects a mismatched pair. Pass
    ``nonempty=True`` for ``fit`` rows. Pass ``scalar=True`` for the one
    user index in ``score_all``.

    A non-1D array, a shape other than ``match_shape``, a non-integer
    dtype, or an empty array when ``nonempty`` is true raises
    ``ValueError``. A scalar that is not an integer raises ``TypeError``.
    An index outside ``[0, size)`` raises ``IndexError``. A negative
    index is included: NumPy would wrap it and read the wrong row.

    A 1D array is returned as given. A scalar is returned as a new int64
    array of shape ``(1,)``.
    """
    if scalar:
        value = _scalar_index(indices, name)
        if value < 0 or value >= size:
            raise IndexError(
                f"{name} must lie in [0, {size}); got {value!r}"
            )
        return np.array([value], dtype=np.int64)

    array = np.asarray(indices)
    if array.ndim != 1:
        raise ValueError(f"{name} must have shape (n,); got {array.shape}")
    if match_shape is not None and array.shape != match_shape:
        raise ValueError(
            f"{name} must have shape {match_shape}; got {array.shape}"
        )
    if nonempty and array.shape[0] == 0:
        raise ValueError(f"{name} must be non-empty")
    if not np.issubdtype(array.dtype, np.integer):
        raise ValueError(
            f"{name} must have an integer dtype; got {array.dtype}"
        )
    if array.shape[0] and (np.any(array < 0) or np.any(array >= size)):
        # Report one illegal value. NumPy would wrap a negative index.
        invalid = (array < 0) | (array >= size)
        first = int(array[np.flatnonzero(invalid)[0]])
        raise IndexError(f"{name} must lie in [0, {size}); got {first!r}")
    return array


def _scalar_index(value: object, name: str) -> int:
    """Return one Python int from a user index, or raise ``TypeError``."""
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(
            f"{name} must be an integer; got {type(value).__name__}"
        )
    if isinstance(value, (int, np.integer)):
        return int(value)
    if (
        isinstance(value, np.ndarray)
        and value.ndim == 0
        and np.issubdtype(value.dtype, np.integer)
    ):
        return int(value)
    raise TypeError(f"{name} must be an integer; got {type(value).__name__}")


class Recommender(ABC):
    """Predict ratings from dataset indices.

    ``fit`` learns from row positions into one ``Dataset``. ``predict``
    scores user and movie pairs. ``score_all`` scores every movie for one
    user. The two reads agree: ``predict(users, movies)[k]`` equals
    ``score_all(users[k])[movies[k]]``.

    Subclasses call ``check_indices`` before they use an index. That check
    rejects a bad shape and a negative index, which NumPy would wrap.
    ``predict`` and ``score_all`` before ``fit`` raise ``RuntimeError``.
    """

    @abstractmethod
    def fit(self, dataset: Dataset, rows: np.ndarray) -> Self:
        """Fit on ``dataset`` rows and return this model.

        ``rows`` is a 1D integer array of row positions, for example
        ``split.train``. Read ``dataset`` only through those rows, and do
        not keep a reference to ``dataset``.

        Check ``rows`` with ``check_indices`` and ``nonempty=True``. An
        empty array raises ``ValueError``. A row outside
        ``[0, n_ratings)``, including a negative row, raises ``IndexError``.
        """
        ...

    @abstractmethod
    def predict(
        self, user_indices: np.ndarray, movie_indices: np.ndarray
    ) -> np.ndarray:
        """Return predicted ratings for each user and movie pair.

        ``user_indices`` and ``movie_indices`` are dataset indices, not
        raw ids. Both have shape ``(n,)``. The result has shape ``(n,)``
        and dtype float64, and ``predict(users, movies)[k]`` equals
        ``score_all(users[k])[movies[k]]``.

        Calling this before ``fit`` raises ``RuntimeError``. Check both
        arrays with ``check_indices``, and pass the user shape as
        ``match_shape`` for the movies. A mismatched or non-1D shape
        raises ``ValueError``. An index outside ``[0, n_users)`` or
        ``[0, n_movies)``, including a negative index, raises
        ``IndexError``.
        """
        ...

    @abstractmethod
    def score_all(self, user_index: int) -> np.ndarray:
        """Return a predicted rating for every movie.

        ``user_index`` is one dataset user index. The result is a new
        float64 array of shape ``(n_movies,)``. The caller owns it, so a
        later step can mask seen movies in place.

        Calling this before ``fit`` raises ``RuntimeError``. Check
        ``user_index`` with ``check_indices`` and ``scalar=True``. An
        index outside ``[0, n_users)``, including a negative index, raises
        ``IndexError``.
        """
        ...
