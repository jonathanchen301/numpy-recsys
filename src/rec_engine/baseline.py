"""Movie-level Bayesian mean baseline."""

from __future__ import annotations

import math
from typing import Self

import numpy as np

from rec_engine.dataset import Dataset
from rec_engine.recommender import Recommender, check_indices


def _prior_strength(value: object) -> float:
    """Return ``value`` as a float that is finite and greater than 0.

    ``0``, a negative number, NaN, and an infinity raise ``ValueError``.
    A bool does too: ``True`` is an ``int``, and ``float(True)`` is ``1.0``.
    """
    strength: float | None = None
    if not isinstance(value, (bool, np.bool_)) and isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        strength = float(value)
    if strength is None or not math.isfinite(strength) or not strength > 0:
        raise ValueError(
            "prior_strength must be finite and greater than 0; "
            f"got {value!r}"
        )
    return strength


class BayesianMeanBaseline(Recommender):
    """Score each movie by a Bayesian average of its training ratings.

    The score is ``(v * R + m * C) / (v + m)``. ``v`` is the movie's
    train-row count, ``R`` its train mean, ``C`` the mean of the train
    ratings, and ``m`` is ``prior_strength``. The train-row sum is
    ``v * R``, so a movie with no train row scores ``C`` and the
    denominator is never 0.

    ``predict`` and ``score_all`` both return that movie score. The user
    index is checked and then ignored, because this baseline has no user
    term. ``score_all`` returns a new array, so the caller can mask seen
    movies in place. ``fit`` reads ``dataset`` only through ``rows`` and
    keeps no reference to it.
    """

    def __init__(self, prior_strength: float = 10.0) -> None:
        """Set the shrinkage ``m``.

        ``prior_strength`` must be finite and greater than 0. Any other
        value raises ``ValueError``. The default is ``10.0``.
        """
        super().__init__()
        self._prior_strength = _prior_strength(prior_strength)
        self._scores: np.ndarray | None = None
        self._n_users: int | None = None
        self._n_movies: int | None = None
        self._global_mean: float | None = None

    @property
    def prior_strength(self) -> float:
        """Shrinkage ``m``. Finite and greater than 0."""
        return self._prior_strength

    @property
    def global_mean(self) -> float:
        """Mean of the ratings in the fitted rows.

        Calling this before ``fit`` raises ``RuntimeError``.
        """
        _scores, _n_users, _n_movies, global_mean = self._fitted()
        return global_mean

    def fit(self, dataset: Dataset, rows: np.ndarray) -> Self:
        """Fit movie scores on ``rows`` and return this model.

        ``rows`` is a 1D integer array of positions into ``dataset``, for
        example ``split.train``. An empty array raises ``ValueError``. A
        row outside ``[0, n_ratings)``, including a negative row, raises
        ``IndexError``.
        """
        checked = check_indices(
            rows, len(dataset), name="rows", nonempty=True
        )
        movies = dataset.movie_indices[checked]  # (n_rows,)
        ratings = dataset.ratings[checked]  # (n_rows,)
        # (n_movies,). A movie with no selected row stays 0.
        counts = np.bincount(movies, minlength=dataset.n_movies)
        sums = np.bincount(
            movies, weights=ratings, minlength=dataset.n_movies
        )
        global_mean = float(ratings.mean())
        strength = self._prior_strength
        scores = (sums + strength * global_mean) / (counts + strength)
        # ``m`` is not a power of two, so ``(m * C) / m`` can differ from
        # ``C`` by a rounding error. An unseen movie scores ``C`` exactly.
        scores[counts == 0] = global_mean
        self._scores = np.asarray(scores, dtype=np.float64)
        self._n_users = dataset.n_users
        self._n_movies = dataset.n_movies
        self._global_mean = global_mean
        return self

    def predict(
        self, user_indices: np.ndarray, movie_indices: np.ndarray
    ) -> np.ndarray:
        """Return the fitted movie score for each pair.

        Both arguments are dataset indices of shape ``(n,)``. The result
        has shape ``(n,)`` and dtype float64, and
        ``predict(users, movies)[k]`` equals
        ``score_all(users[k])[movies[k]]``. Calling this before ``fit``
        raises ``RuntimeError``. A mismatched or non-1D shape raises
        ``ValueError``. An index outside ``[0, n_users)`` or
        ``[0, n_movies)``, including a negative index, raises
        ``IndexError``.
        """
        scores, n_users, n_movies, _global_mean = self._fitted()
        users = check_indices(
            user_indices, n_users, name="user_indices"
        )
        movies = check_indices(
            movie_indices,
            n_movies,
            name="movie_indices",
            match_shape=users.shape,
        )
        return scores[movies]

    def score_all(self, user_index: int) -> np.ndarray:
        """Return a copy of every movie score.

        ``user_index`` is one dataset user index. The result is a new
        float64 array of shape ``(n_movies,)``. The user is validated and
        does not change the scores. Calling this before ``fit`` raises
        ``RuntimeError``. An index outside ``[0, n_users)``, including a
        negative index, raises ``IndexError``.
        """
        scores, n_users, _n_movies, _global_mean = self._fitted()
        check_indices(user_index, n_users, name="user_index", scalar=True)
        return scores.copy()

    def _fitted(self) -> tuple[np.ndarray, int, int, float]:
        """Return scores, ``n_users``, ``n_movies``, and the train mean.

        Raises ``RuntimeError`` when ``fit`` has not been called.
        """
        scores = self._scores
        n_users = self._n_users
        n_movies = self._n_movies
        global_mean = self._global_mean
        if (
            scores is None
            or n_users is None
            or n_movies is None
            or global_mean is None
        ):
            raise RuntimeError(
                "BayesianMeanBaseline.fit must be called before reading scores"
            )
        return scores, n_users, n_movies, global_mean

    def __repr__(self) -> str:
        return (
            f"BayesianMeanBaseline(prior_strength={self._prior_strength!r})"
        )
