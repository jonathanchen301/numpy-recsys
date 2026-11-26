"""Tests for the movie-level Bayesian mean baseline."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rec_engine import BayesianMeanBaseline, Dataset, Split, temporal_split

_ML_1M = Path(__file__).resolve().parents[1] / "data" / "ml-1m"

# Row 1 is the only rating of movie 4. Train is every other row, so a fit
# that takes a prefix, or every row, sees that held-out rating.
_HELD_OUT_ROW = 1
_TRAIN_ROW_POSITIONS = (0, 2, 3, 4, 5, 6, 7)


def _toy_dataset(held_out_rating: float) -> Dataset:
    """Four movies: four 5s, one 5, two 1s, and one held-out-only movie.

    Movie 3's 1s pull the train mean below 5, so movie 2's single 5 shrinks
    below movie 1. ``held_out_rating`` is stored only on row 1.
    """
    ratings = pd.DataFrame(
        {
            "user_id": [10, 20, 30, 40, 10, 20, 30, 40],
            "movie_id": [1, 4, 1, 1, 1, 2, 3, 3],
            "rating": [5, held_out_rating, 5, 5, 5, 5, 1, 1],
            "timestamp": [1, 2, 3, 4, 5, 6, 7, 8],
        }
    )
    users = pd.DataFrame(
        {
            "user_id": [10, 20, 30, 40],
            "gender": ["F", "M", "M", "F"],
            "age": [25, 35, 18, 45],
            "occupation": [4, 7, 1, 2],
            "zip_code": ["02139", "98107", "10001", "94107"],
        }
    )
    movies = pd.DataFrame(
        {
            "movie_id": [1, 2, 3, 4],
            "title": [
                "Heavy (1995)",
                "Single Five (1995)",
                "Low (1995)",
                "Cold (1995)",
            ],
            "genres": ["Drama", "Drama", "Drama", "Drama"],
        }
    )
    return Dataset(ratings, users, movies)


def _rmse(predicted: np.ndarray, actual: np.ndarray) -> float:
    """Root mean squared error of two rating vectors."""
    residual = predicted - actual
    return float(np.sqrt(np.mean(residual * residual)))


@pytest.fixture
def dataset() -> Dataset:
    return _toy_dataset(1.0)


@pytest.fixture
def train_rows() -> np.ndarray:
    """Every row except the held-out movie. Not a prefix of the file."""
    return np.array(_TRAIN_ROW_POSITIONS, dtype=np.int64)


@pytest.fixture
def fitted(
    dataset: Dataset, train_rows: np.ndarray
) -> BayesianMeanBaseline:
    return BayesianMeanBaseline().fit(dataset, train_rows)


@pytest.fixture(scope="module")
def ml1m_fitted() -> tuple[Dataset, Split, BayesianMeanBaseline]:
    ml1m = Dataset.from_dir(_ML_1M)
    split = temporal_split(ml1m)
    model = BayesianMeanBaseline().fit(ml1m, split.train)
    return ml1m, split, model


@pytest.mark.parametrize(
    "strength",
    [10.0, 2.0],
    ids=["default-strength", "strength-2"],
)
def test_bayesian_mean_baseline_score_all_matches_hand_computed_scores(
    dataset: Dataset, train_rows: np.ndarray, strength: float
) -> None:
    assert dataset.movie_to_index == {1: 0, 2: 1, 3: 2, 4: 3}
    np.testing.assert_array_equal(
        dataset.movie_indices[train_rows],
        np.array([0, 0, 0, 0, 1, 2, 2], dtype=np.int64),
    )
    np.testing.assert_array_equal(
        dataset.ratings[train_rows],
        np.array([5, 5, 5, 5, 5, 1, 1], dtype=np.float64),
    )
    # Movie 1 sum 20 count 4, movie 2 sum 5 count 1, movie 3 sum 2 count 2.
    # Movie 4 has no train row, so its score is the train mean.
    train_mean = (5.0 + 5.0 + 5.0 + 5.0 + 5.0 + 1.0 + 1.0) / 7.0
    expected = np.array(
        [
            (20.0 + strength * train_mean) / (4.0 + strength),
            (5.0 + strength * train_mean) / (1.0 + strength),
            (2.0 + strength * train_mean) / (2.0 + strength),
            train_mean,
        ],
        dtype=np.float64,
    )
    model = BayesianMeanBaseline(strength).fit(dataset, train_rows)
    scores = model.score_all(0)
    assert scores.shape == (dataset.n_movies,)
    assert scores.dtype == np.float64
    np.testing.assert_allclose(model.prior_strength, strength, rtol=0, atol=0)
    np.testing.assert_allclose(scores, expected)


def test_bayesian_mean_baseline_single_five_scores_below_heavy_movie(
    dataset: Dataset, train_rows: np.ndarray, fitted: BayesianMeanBaseline
) -> None:
    heavy = dataset.movie_to_index[1]
    single = dataset.movie_to_index[2]
    train_movies = dataset.movie_indices[train_rows]
    train_ratings = dataset.ratings[train_rows]
    heavy_ratings = train_ratings[train_movies == heavy]
    single_ratings = train_ratings[train_movies == single]
    assert heavy_ratings.shape[0] > 1
    assert single_ratings.shape == (1,)
    np.testing.assert_allclose(single_ratings, np.array([5.0]), rtol=0, atol=0)
    np.testing.assert_allclose(
        heavy_ratings, np.full(heavy_ratings.shape, 5.0), rtol=0, atol=0
    )
    scores = fitted.score_all(0)
    assert scores[single] < scores[heavy]


def test_bayesian_mean_baseline_unseen_movie_scores_train_mean(
    dataset: Dataset, train_rows: np.ndarray, fitted: BayesianMeanBaseline
) -> None:
    cold = dataset.movie_to_index[4]
    assert int(np.count_nonzero(dataset.movie_indices[train_rows] == cold)) == 0
    assert int(np.count_nonzero(dataset.movie_indices == cold)) == 1
    train_mean = float(dataset.ratings[train_rows].mean())
    scores = fitted.score_all(0)
    np.testing.assert_allclose(scores[cold], train_mean, rtol=0, atol=0)
    np.testing.assert_allclose(fitted.global_mean, train_mean, rtol=0, atol=0)


def test_bayesian_mean_baseline_held_out_ratings_do_not_change_scores(
    train_rows: np.ndarray,
) -> None:
    assert _HELD_OUT_ROW not in train_rows.tolist()
    low_held_out = _toy_dataset(1.0)
    high_held_out = _toy_dataset(5.0)
    same_rows = np.ones(len(low_held_out), dtype=bool)
    same_rows[_HELD_OUT_ROW] = False
    np.testing.assert_array_equal(
        low_held_out.ratings[same_rows], high_held_out.ratings[same_rows]
    )
    np.testing.assert_allclose(
        low_held_out.ratings[_HELD_OUT_ROW], 1.0, rtol=0, atol=0
    )
    np.testing.assert_allclose(
        high_held_out.ratings[_HELD_OUT_ROW], 5.0, rtol=0, atol=0
    )

    low_model = BayesianMeanBaseline().fit(low_held_out, train_rows)
    high_model = BayesianMeanBaseline().fit(high_held_out, train_rows)
    np.testing.assert_allclose(
        low_model.score_all(0), high_model.score_all(0), rtol=0, atol=0
    )
    train_mean = float(low_held_out.ratings[train_rows].mean())
    all_rows_mean = float(low_held_out.ratings.mean())
    np.testing.assert_allclose(low_model.global_mean, train_mean, rtol=0, atol=0)
    np.testing.assert_allclose(
        high_model.global_mean, train_mean, rtol=0, atol=0
    )
    assert abs(train_mean - all_rows_mean) > 0.2
    assert abs(low_model.global_mean - all_rows_mean) > 0.2


def test_bayesian_mean_baseline_predict_matches_score_all_for_each_pair(
    dataset: Dataset, fitted: BayesianMeanBaseline
) -> None:
    users = np.array([0, dataset.n_users - 1, 1, 2], dtype=np.int64)
    movies = np.array([0, 1, dataset.n_movies - 1, 2], dtype=np.int64)
    predicted = fitted.predict(users, movies)
    assert predicted.shape == (users.shape[0],)
    assert predicted.dtype == np.float64
    expected = np.array(
        [
            fitted.score_all(int(user))[int(movie)]
            for user, movie in zip(users, movies, strict=True)
        ],
        dtype=np.float64,
    )
    np.testing.assert_allclose(predicted, expected, rtol=0, atol=0)


def test_bayesian_mean_baseline_predict_one_pair_returns_one_float64(
    fitted: BayesianMeanBaseline,
) -> None:
    predicted = fitted.predict(
        np.array([0], dtype=np.int64), np.array([1], dtype=np.int64)
    )
    assert predicted.shape == (1,)
    assert predicted.dtype == np.float64
    np.testing.assert_allclose(
        predicted, fitted.score_all(0)[1], rtol=0, atol=0
    )


def test_bayesian_mean_baseline_predict_empty_returns_empty_float64(
    fitted: BayesianMeanBaseline,
) -> None:
    predicted = fitted.predict(
        np.array([], dtype=np.int64), np.array([], dtype=np.int64)
    )
    assert predicted.shape == (0,)
    assert predicted.dtype == np.float64


def test_bayesian_mean_baseline_score_all_matches_for_two_users(
    dataset: Dataset, fitted: BayesianMeanBaseline
) -> None:
    assert dataset.n_users >= 2
    first = fitted.score_all(0)
    last = fitted.score_all(dataset.n_users - 1)
    assert first.shape == (dataset.n_movies,)
    assert last.dtype == np.float64
    np.testing.assert_allclose(first, last, rtol=0, atol=0)


def test_bayesian_mean_baseline_score_all_copy_ignores_caller_writes(
    fitted: BayesianMeanBaseline,
) -> None:
    returned = fitted.score_all(0)
    original = returned.copy()
    returned[0] = 0.0
    np.testing.assert_allclose(fitted.score_all(0), original, rtol=0, atol=0)


def test_bayesian_mean_baseline_fit_returns_the_same_instance(
    dataset: Dataset, train_rows: np.ndarray
) -> None:
    model = BayesianMeanBaseline()
    assert model.fit(dataset, train_rows) is model


def test_bayesian_mean_baseline_fit_last_row_uses_that_rating(
    dataset: Dataset,
) -> None:
    last = len(dataset) - 1
    model = BayesianMeanBaseline().fit(
        dataset, np.array([last], dtype=np.int64)
    )
    np.testing.assert_allclose(
        model.global_mean,
        float(dataset.ratings[last]),
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize(
    "prior_strength",
    [0.0, -1.0, math.nan, math.inf, -math.inf],
    ids=["zero", "negative", "nan", "inf", "negative-inf"],
)
def test_bayesian_mean_baseline_bad_prior_strength_raises(
    prior_strength: float,
) -> None:
    with pytest.raises(ValueError, match="prior_strength"):
        BayesianMeanBaseline(prior_strength)


def test_bayesian_mean_baseline_predict_before_fit_raises() -> None:
    model = BayesianMeanBaseline()
    with pytest.raises(RuntimeError, match="fit"):
        model.predict(np.array([0], dtype=np.int64), np.array([0], dtype=np.int64))


def test_bayesian_mean_baseline_score_all_before_fit_raises() -> None:
    model = BayesianMeanBaseline()
    with pytest.raises(RuntimeError, match="fit"):
        model.score_all(0)


def test_bayesian_mean_baseline_global_mean_before_fit_raises() -> None:
    model = BayesianMeanBaseline()
    with pytest.raises(RuntimeError, match="fit"):
        _ = model.global_mean


def test_bayesian_mean_baseline_score_all_user_at_n_users_raises(
    dataset: Dataset, fitted: BayesianMeanBaseline
) -> None:
    user_index = dataset.n_users
    with pytest.raises(IndexError, match=f"got {user_index}"):
        fitted.score_all(user_index)


def test_bayesian_mean_baseline_score_all_negative_user_raises(
    fitted: BayesianMeanBaseline,
) -> None:
    with pytest.raises(IndexError, match="got -1"):
        fitted.score_all(-1)


def test_bayesian_mean_baseline_predict_user_at_n_users_raises(
    dataset: Dataset, fitted: BayesianMeanBaseline
) -> None:
    users = np.array([dataset.n_users], dtype=np.int64)
    movies = np.array([0], dtype=np.int64)
    with pytest.raises(IndexError, match=f"got {dataset.n_users}"):
        fitted.predict(users, movies)


def test_bayesian_mean_baseline_predict_negative_user_raises(
    fitted: BayesianMeanBaseline,
) -> None:
    users = np.array([-1], dtype=np.int64)
    movies = np.array([0], dtype=np.int64)
    with pytest.raises(IndexError, match="got -1"):
        fitted.predict(users, movies)


def test_bayesian_mean_baseline_predict_movie_at_n_movies_raises(
    dataset: Dataset, fitted: BayesianMeanBaseline
) -> None:
    users = np.array([0], dtype=np.int64)
    movies = np.array([dataset.n_movies], dtype=np.int64)
    with pytest.raises(IndexError, match=f"got {dataset.n_movies}"):
        fitted.predict(users, movies)


def test_bayesian_mean_baseline_predict_negative_movie_raises(
    fitted: BayesianMeanBaseline,
) -> None:
    users = np.array([0], dtype=np.int64)
    movies = np.array([-1], dtype=np.int64)
    with pytest.raises(IndexError, match="got -1"):
        fitted.predict(users, movies)


def test_bayesian_mean_baseline_fit_row_at_n_ratings_raises(
    dataset: Dataset,
) -> None:
    rows = np.array([len(dataset)], dtype=np.int64)
    with pytest.raises(IndexError, match=f"got {len(dataset)}"):
        BayesianMeanBaseline().fit(dataset, rows)


def test_bayesian_mean_baseline_fit_negative_row_raises(
    dataset: Dataset,
) -> None:
    with pytest.raises(IndexError, match="got -1"):
        BayesianMeanBaseline().fit(dataset, np.array([-1], dtype=np.int64))


def test_bayesian_mean_baseline_predict_mismatched_shapes_raises(
    fitted: BayesianMeanBaseline,
) -> None:
    users = np.array([0, 1], dtype=np.int64)
    movies = np.array([0], dtype=np.int64)
    with pytest.raises(ValueError, match="movie_indices"):
        fitted.predict(users, movies)


def test_bayesian_mean_baseline_predict_non_1d_user_indices_raises(
    fitted: BayesianMeanBaseline,
) -> None:
    users = np.array([0, 1], dtype=np.int64).reshape(2, 1)
    movies = np.array([0, 1], dtype=np.int64)
    with pytest.raises(ValueError, match="user_indices"):
        fitted.predict(users, movies)


def test_bayesian_mean_baseline_predict_non_1d_movie_indices_raises(
    fitted: BayesianMeanBaseline,
) -> None:
    users = np.array([0, 1], dtype=np.int64)
    movies = np.array([0, 1], dtype=np.int64).reshape(1, 2)
    with pytest.raises(ValueError, match="movie_indices"):
        fitted.predict(users, movies)


def test_bayesian_mean_baseline_predict_float_indices_raises(
    fitted: BayesianMeanBaseline,
) -> None:
    with pytest.raises(ValueError, match="integer dtype"):
        fitted.predict(np.array([0.0]), np.array([0.0]))


def test_bayesian_mean_baseline_score_all_float_user_index_raises(
    fitted: BayesianMeanBaseline,
) -> None:
    with pytest.raises(TypeError, match="user_index"):
        fitted.score_all(1.5)


def test_bayesian_mean_baseline_score_all_bool_user_index_raises(
    fitted: BayesianMeanBaseline,
) -> None:
    with pytest.raises(TypeError, match="user_index"):
        fitted.score_all(True)


def test_bayesian_mean_baseline_fit_empty_rows_raises(dataset: Dataset) -> None:
    with pytest.raises(ValueError, match="non-empty"):
        BayesianMeanBaseline().fit(dataset, np.array([], dtype=np.int64))


def test_bayesian_mean_baseline_fit_non_1d_rows_raises(dataset: Dataset) -> None:
    rows = np.array([0, 1, 2], dtype=np.int64).reshape(1, 3)
    with pytest.raises(ValueError, match="rows"):
        BayesianMeanBaseline().fit(dataset, rows)


def test_bayesian_mean_baseline_fit_float_rows_raises(dataset: Dataset) -> None:
    with pytest.raises(ValueError, match="integer dtype"):
        BayesianMeanBaseline().fit(dataset, np.array([0.0]))


@pytest.mark.skipif(not _ML_1M.is_dir(), reason="data/ml-1m is not downloaded")
def test_bayesian_mean_baseline_ml1m_unseen_movies_score_train_mean(
    ml1m_fitted: tuple[Dataset, Split, BayesianMeanBaseline],
) -> None:
    dataset, split, model = ml1m_fitted
    counts = np.bincount(
        dataset.movie_indices[split.train], minlength=dataset.n_movies
    )
    cold = np.flatnonzero(counts == 0)
    assert dataset.n_movies == 3_706
    assert cold.shape == (51,)
    scores = model.score_all(0)
    assert scores.shape == (dataset.n_movies,)
    assert scores.dtype == np.float64
    np.testing.assert_allclose(scores[cold], model.global_mean, rtol=0, atol=0)
    train_mean = float(dataset.ratings[split.train].mean())
    np.testing.assert_allclose(model.global_mean, train_mean, rtol=0, atol=0)
    assert model.global_mean == pytest.approx(3.6314, abs=5e-5)
    all_rows_mean = float(dataset.ratings.mean())
    assert all_rows_mean == pytest.approx(3.5816, abs=5e-5)
    assert abs(model.global_mean - all_rows_mean) > 0.04


@pytest.mark.skipif(not _ML_1M.is_dir(), reason="data/ml-1m is not downloaded")
def test_bayesian_mean_baseline_ml1m_validation_rmse_matches_pinned(
    ml1m_fitted: tuple[Dataset, Split, BayesianMeanBaseline],
) -> None:
    dataset, split, model = ml1m_fitted
    assert split.val.shape[0] > 0
    actual = dataset.ratings[split.val]
    predicted = model.predict(
        dataset.user_indices[split.val], dataset.movie_indices[split.val]
    )
    assert predicted.shape == actual.shape
    assert predicted.dtype == np.float64
    rmse = _rmse(predicted, actual)
    global_predicted = np.full(actual.shape, model.global_mean)
    global_rmse = _rmse(global_predicted, actual)
    assert rmse == pytest.approx(0.98797, abs=1e-5)
    assert global_rmse == pytest.approx(1.12556, abs=1e-5)
    assert rmse < global_rmse
