"""Tests for loading MovieLens tables into a Dataset."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pandas.errors import MergeError

from rec_engine import Dataset

_ML_1M = Path(__file__).resolve().parents[1] / "data" / "ml-1m"

_FRAME_COLUMNS = [
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
]


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    """Three Latin-1 tables: 3 users, 4 movies (one unrated), 5 ratings.

    Rating order is not sorted by id. First-seen order and sorted-id order
    disagree, so an index built the wrong way fails the exact array asserts.
    """
    (tmp_path / "users.dat").write_text(
        "10::F::25::4::02139\n"
        "20::M::35::7::98107-2117\n"
        "30::M::18::1::10001\n",
        encoding="latin-1",
    )
    (tmp_path / "movies.dat").write_text(
        "1::Toy Story (1995)::Animation|Children's|Comedy\n"
        "5::Misérables, Les (1995)::Drama\n"
        "9::Star Wars: Episode IV - A New Hope (1977)::Action|Adventure|Sci-Fi\n"
        "42::Unrated Movie (1999)::Documentary\n",
        encoding="latin-1",
    )
    (tmp_path / "ratings.dat").write_text(
        "30::9::5::978300000\n"
        "10::1::4::978300100\n"
        "20::5::3::978300200\n"
        "10::9::5::978300300\n"
        "20::1::2::978300400\n",
        encoding="latin-1",
    )
    return tmp_path


@pytest.fixture
def dataset(data_dir: Path) -> Dataset:
    return Dataset.from_dir(data_dir)


def test_dataset_user_to_index_follows_sorted_raw_id(dataset: Dataset) -> None:
    assert dataset.user_to_index == {10: 0, 20: 1, 30: 2}
    assert dataset.user_ids.shape == (3,)
    assert dataset.user_ids.dtype == np.int64
    np.testing.assert_array_equal(
        dataset.user_ids, np.array([10, 20, 30], dtype=np.int64)
    )


def test_dataset_movie_to_index_omits_unrated_movie(dataset: Dataset) -> None:
    assert dataset.movie_to_index == {1: 0, 5: 1, 9: 2}
    assert 42 not in dataset.movie_to_index
    assert dataset.movie_ids.shape == (3,)
    assert dataset.movie_ids.dtype == np.int64
    np.testing.assert_array_equal(
        dataset.movie_ids, np.array([1, 5, 9], dtype=np.int64)
    )


def test_dataset_user_indices_follow_file_order(dataset: Dataset) -> None:
    expected = np.array([2, 0, 1, 0, 1], dtype=np.int64)
    assert dataset.user_indices.shape == expected.shape
    assert dataset.user_indices.dtype == np.int64
    np.testing.assert_array_equal(dataset.user_indices, expected)


def test_dataset_movie_indices_follow_file_order(dataset: Dataset) -> None:
    expected = np.array([2, 0, 1, 2, 0], dtype=np.int64)
    assert dataset.movie_indices.shape == expected.shape
    assert dataset.movie_indices.dtype == np.int64
    np.testing.assert_array_equal(dataset.movie_indices, expected)


def test_dataset_ratings_are_float64_in_file_order(dataset: Dataset) -> None:
    expected = np.array([5, 4, 3, 5, 2], dtype=np.float64)
    assert dataset.ratings.shape == expected.shape
    assert dataset.ratings.dtype == np.float64
    np.testing.assert_array_equal(dataset.ratings, expected)


def test_dataset_timestamps_are_int64_in_file_order(dataset: Dataset) -> None:
    expected = np.array(
        [978300000, 978300100, 978300200, 978300300, 978300400],
        dtype=np.int64,
    )
    assert dataset.timestamps.shape == expected.shape
    assert dataset.timestamps.dtype == np.int64
    np.testing.assert_array_equal(dataset.timestamps, expected)


def test_dataset_user_ids_round_trip_to_raw_column(dataset: Dataset) -> None:
    raw_user_ids = dataset.frame["user_id"].to_numpy()
    np.testing.assert_array_equal(
        dataset.user_ids[dataset.user_indices], raw_user_ids
    )


def test_dataset_movie_ids_round_trip_to_raw_column(dataset: Dataset) -> None:
    raw_movie_ids = dataset.frame["movie_id"].to_numpy()
    np.testing.assert_array_equal(
        dataset.movie_ids[dataset.movie_indices], raw_movie_ids
    )


def test_dataset_frame_known_row_has_title_gender_and_zip(
    dataset: Dataset,
) -> None:
    frame = dataset.frame
    assert list(frame.columns) == _FRAME_COLUMNS
    assert len(frame) == 5
    row = frame.iloc[2]
    assert row["user_id"] == 20
    assert row["movie_id"] == 5
    assert row["title"] == "Misérables, Les (1995)"
    assert row["gender"] == "M"
    assert row["zip_code"] == "98107-2117"
    assert row["genres"] == "Drama"


def test_dataset_frame_title_keeps_single_colon(dataset: Dataset) -> None:
    row = dataset.frame.iloc[0]
    assert row["movie_id"] == 9
    assert row["title"] == "Star Wars: Episode IV - A New Hope (1977)"


def test_dataset_frame_zip_code_keeps_leading_zero(dataset: Dataset) -> None:
    row = dataset.frame.iloc[1]
    assert row["user_id"] == 10
    assert row["zip_code"] == "02139"


def test_dataset_sizes_count_ratings_and_rated_ids(dataset: Dataset) -> None:
    assert len(dataset) == 5
    assert dataset.n_users == 3
    assert dataset.n_movies == 3


def test_dataset_frame_copy_ignores_later_writes(dataset: Dataset) -> None:
    frame = dataset.frame
    frame.loc[0, "title"] = "changed"
    assert (
        dataset.frame.iloc[0]["title"]
        == "Star Wars: Episode IV - A New Hope (1977)"
    )


@pytest.mark.parametrize(
    "array_name",
    [
        "user_indices",
        "movie_indices",
        "user_ids",
        "movie_ids",
        "ratings",
        "timestamps",
    ],
)
def test_dataset_array_item_assignment_raises(
    dataset: Dataset, array_name: str
) -> None:
    values = getattr(dataset, array_name)
    with pytest.raises(ValueError, match="read-only"):
        values[0] = 0


@pytest.mark.parametrize(
    ("map_name", "raw_id"),
    [("user_to_index", 10), ("movie_to_index", 1)],
    ids=["user", "movie"],
)
def test_dataset_id_map_item_assignment_raises(
    dataset: Dataset, map_name: str, raw_id: int
) -> None:
    lookup = getattr(dataset, map_name)
    with pytest.raises(TypeError, match="item assignment"):
        lookup[raw_id] = 0


def test_dataset_unknown_movie_id_raises_value_error() -> None:
    ratings = pd.DataFrame(
        {
            "user_id": [10],
            "movie_id": [99],
            "rating": [5],
            "timestamp": [978300000],
        }
    )
    users = pd.DataFrame(
        {
            "user_id": [10],
            "gender": ["F"],
            "age": [25],
            "occupation": [4],
            "zip_code": ["02139"],
        }
    )
    movies = pd.DataFrame(
        {
            "movie_id": [1],
            "title": ["Toy Story (1995)"],
            "genres": ["Animation|Children's|Comedy"],
        }
    )
    with pytest.raises(ValueError, match="every rating must match"):
        Dataset(ratings, users, movies)


def test_dataset_duplicate_movie_id_raises_merge_error() -> None:
    ratings = pd.DataFrame(
        {
            "user_id": [10],
            "movie_id": [1],
            "rating": [5],
            "timestamp": [978300000],
        }
    )
    users = pd.DataFrame(
        {
            "user_id": [10],
            "gender": ["F"],
            "age": [25],
            "occupation": [4],
            "zip_code": ["02139"],
        }
    )
    movies = pd.DataFrame(
        {
            "movie_id": [1, 1],
            "title": ["Toy Story (1995)", "Toy Story (1995)"],
            "genres": ["Animation", "Animation"],
        }
    )
    with pytest.raises(MergeError, match="many-to-one"):
        Dataset(ratings, users, movies)


@pytest.mark.skipif(not _ML_1M.is_dir(), reason="data/ml-1m is not downloaded")
def test_dataset_ml1m_counts_users_movies_and_ratings() -> None:
    dataset = Dataset.from_dir(_ML_1M)
    assert dataset.n_users == 6_040
    assert dataset.n_movies == 3_706
    assert len(dataset) == 1_000_209
