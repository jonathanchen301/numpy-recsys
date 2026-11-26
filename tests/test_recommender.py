"""Tests for the recommender interface."""

from __future__ import annotations

from typing import Self

import numpy as np
import pytest

from rec_engine import BayesianMeanBaseline, Dataset, Recommender


def test_recommender_cannot_be_instantiated() -> None:
    with pytest.raises(TypeError, match="abstract"):
        Recommender()


def test_recommender_subclass_missing_score_all_cannot_be_instantiated() -> None:
    class Partial(Recommender):
        def fit(self, dataset: Dataset, rows: np.ndarray) -> Self:
            return self

        def predict(
            self, user_indices: np.ndarray, movie_indices: np.ndarray
        ) -> np.ndarray:
            return np.empty(user_indices.shape[0], dtype=np.float64)

    with pytest.raises(TypeError, match="score_all"):
        Partial()


def test_recommender_subclass_with_every_method_can_be_instantiated() -> None:
    class Complete(Recommender):
        def fit(self, dataset: Dataset, rows: np.ndarray) -> Self:
            return self

        def predict(
            self, user_indices: np.ndarray, movie_indices: np.ndarray
        ) -> np.ndarray:
            return np.empty(user_indices.shape[0], dtype=np.float64)

        def score_all(self, user_index: int) -> np.ndarray:
            return np.empty(0, dtype=np.float64)

    model = Complete()
    assert isinstance(model, Recommender)


def test_bayesian_mean_baseline_is_a_recommender() -> None:
    assert isinstance(BayesianMeanBaseline(), Recommender)
