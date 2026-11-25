"""Tests for the per-user temporal train, validation, and test split."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rec_engine import Dataset, Split, temporal_split

_ML_1M = Path(__file__).resolve().parents[1] / "data" / "ml-1m"

# File order is scrambled against timestamp order. A cut that takes the last
# rows of the file, or the last rows of each user in file order, selects
# different positions than a time cut. User 10 has two ratings at timestamp
# 150: row 4, then row 11. Those sit on the train/validation cut.
_RATINGS = [
    (20, 240),
    (10, 180),
    (30, 310),
    (10, 100),
    (10, 150),
    (40, 400),
    (20, 200),
    (10, 170),
    (30, 330),
    (10, 120),
    (20, 220),
    (10, 150),
    (30, 300),
    (10, 140),
    (20, 210),
    (10, 160),
    (30, 320),
    (10, 110),
    (10, 130),
    (20, 230),
]


@pytest.fixture
def dataset() -> Dataset:
    """Users of 10, 5, 4, and 1 ratings, one of them tied on the cut.

    User 10 has 10 ratings, user 20 has 5, user 30 has 4, and user 40 has 1.
    """
    user_id, timestamp = zip(*_RATINGS, strict=True)
    ratings = pd.DataFrame(
        {
            "user_id": user_id,
            "movie_id": 1,
            "rating": 4,
            "timestamp": timestamp,
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
            "movie_id": [1],
            "title": ["Toy Story (1995)"],
            "genres": ["Animation|Children's|Comedy"],
        }
    )
    return Dataset(ratings, users, movies)


@pytest.fixture(scope="module")
def ml1m_dataset() -> Dataset:
    return Dataset.from_dir(_ML_1M)


def _rows_for_user(
    dataset: Dataset, rows: np.ndarray, user_index: int
) -> np.ndarray:
    """Row positions in ``rows`` that belong to ``user_index``, order kept."""
    return rows[dataset.user_indices[rows] == user_index]


def _timestamp_bounds_by_user(dataset: Dataset, split: Split) -> pd.DataFrame:
    """Min and max timestamp of each piece, one groupby row per user and piece.

    Piece labels come from the split's positions. The bounds come from that
    labelled frame, not from the sort that built the split.
    """
    labels = np.full(len(dataset), None, dtype=object)
    labels[split.train] = "train"
    labels[split.val] = "val"
    labels[split.test] = "test"
    labelled = pd.DataFrame(
        {
            "user_index": dataset.user_indices.copy(),
            "timestamp": dataset.timestamps.copy(),
            "piece": labels,
        }
    )
    return (
        labelled.groupby(["user_index", "piece"], sort=True)["timestamp"]
        .agg(timestamp_min="min", timestamp_max="max", n_rows="size")
        .reset_index()
    )


def _piece_bound(
    bounds: pd.DataFrame, piece: str, column: str
) -> pd.Series:
    """``column`` indexed by user, for rows whose piece is ``piece``."""
    rows = bounds.loc[bounds["piece"] == piece, ["user_index", column]]
    indexed = rows.set_index("user_index")
    return indexed[column]


def test_temporal_split_scrambled_file_matches_time_cuts(
    dataset: Dataset,
) -> None:
    split = temporal_split(dataset, val_fraction=0.25)
    expected_train = np.array(
        [2, 3, 4, 5, 6, 9, 10, 12, 13, 14, 16, 17, 18],
        dtype=np.int64,
    )
    expected_val = np.array([8, 11, 15, 19], dtype=np.int64)
    expected_test = np.array([0, 1, 7], dtype=np.int64)

    assert split.train.shape == expected_train.shape
    assert split.train.dtype == np.int64
    np.testing.assert_array_equal(split.train, expected_train)
    assert split.val.shape == expected_val.shape
    assert split.val.dtype == np.int64
    np.testing.assert_array_equal(split.val, expected_val)
    assert split.test.shape == expected_test.shape
    assert split.test.dtype == np.int64
    np.testing.assert_array_equal(split.test, expected_test)


def test_temporal_split_pieces_cover_each_row_once(dataset: Dataset) -> None:
    split = temporal_split(dataset, val_fraction=0.25)
    combined = np.concatenate([split.train, split.val, split.test])
    combined.sort()
    expected = np.arange(len(dataset), dtype=np.int64)
    assert combined.shape == expected.shape
    assert combined.dtype == np.int64
    np.testing.assert_array_equal(combined, expected)


def test_temporal_split_groupby_finds_no_future_leakage(
    dataset: Dataset,
) -> None:
    split = temporal_split(dataset, val_fraction=0.25)
    bounds = _timestamp_bounds_by_user(dataset, split)
    assert dataset.n_users == 4
    assert bounds["user_index"].nunique() == dataset.n_users
    assert bounds["n_rows"].sum() == len(dataset)

    train_max = _piece_bound(bounds, "train", "timestamp_max")
    val_min = _piece_bound(bounds, "val", "timestamp_min")
    val_max = _piece_bound(bounds, "val", "timestamp_max")
    test_min = _piece_bound(bounds, "test", "timestamp_min")
    train_and_val = train_max.index.intersection(val_min.index)
    val_and_test = val_max.index.intersection(test_min.index)
    assert train_and_val.shape[0] == 3
    assert val_and_test.shape[0] == 2
    assert (train_max.loc[train_and_val] <= val_min.loc[train_and_val]).all()
    assert (val_max.loc[val_and_test] <= test_min.loc[val_and_test]).all()


def test_temporal_split_tied_timestamp_keeps_earlier_file_row_in_train(
    dataset: Dataset,
) -> None:
    split = temporal_split(dataset, val_fraction=0.25)
    user_index = dataset.user_to_index[10]
    user_rows = np.flatnonzero(dataset.user_indices == user_index)
    timestamps = dataset.timestamps[user_rows]
    values, counts = np.unique(timestamps, return_counts=True)
    repeated = values[counts > 1]
    assert repeated.shape == (1,)
    tied_rows = user_rows[timestamps == repeated[0]]
    assert tied_rows.shape == (2,)
    assert tied_rows.dtype == np.int64
    earlier_file_row = tied_rows[0]
    later_file_row = tied_rows[1]
    assert earlier_file_row < later_file_row

    earlier_in_train = split.train[split.train == earlier_file_row]
    later_in_val = split.val[split.val == later_file_row]
    assert earlier_in_train.shape == (1,)
    assert earlier_in_train.dtype == np.int64
    np.testing.assert_array_equal(
        earlier_in_train, np.array([earlier_file_row], dtype=np.int64)
    )
    assert later_in_val.shape == (1,)
    assert later_in_val.dtype == np.int64
    np.testing.assert_array_equal(
        later_in_val, np.array([later_file_row], dtype=np.int64)
    )
    assert split.val[split.val == earlier_file_row].shape == (0,)
    assert split.train[split.train == later_file_row].shape == (0,)


def test_temporal_split_four_ratings_assigns_no_test_rows(
    dataset: Dataset,
) -> None:
    split = temporal_split(dataset, val_fraction=0.25)
    user_index = dataset.user_to_index[30]
    assert int(np.count_nonzero(dataset.user_indices == user_index)) == 4
    train_rows = _rows_for_user(dataset, split.train, user_index)
    val_rows = _rows_for_user(dataset, split.val, user_index)
    test_rows = _rows_for_user(dataset, split.test, user_index)
    expected_train = np.array([2, 12, 16], dtype=np.int64)
    expected_val = np.array([8], dtype=np.int64)

    assert train_rows.shape == expected_train.shape
    assert train_rows.dtype == np.int64
    np.testing.assert_array_equal(train_rows, expected_train)
    assert val_rows.shape == expected_val.shape
    assert val_rows.dtype == np.int64
    np.testing.assert_array_equal(val_rows, expected_val)
    assert test_rows.shape == (0,)
    assert test_rows.dtype == np.int64


def test_temporal_split_five_ratings_assigns_one_test_row(
    dataset: Dataset,
) -> None:
    split = temporal_split(dataset, val_fraction=0.25)
    user_index = dataset.user_to_index[20]
    assert int(np.count_nonzero(dataset.user_indices == user_index)) == 5
    train_rows = _rows_for_user(dataset, split.train, user_index)
    val_rows = _rows_for_user(dataset, split.val, user_index)
    test_rows = _rows_for_user(dataset, split.test, user_index)
    expected_train = np.array([6, 10, 14], dtype=np.int64)
    expected_val = np.array([19], dtype=np.int64)
    expected_test = np.array([0], dtype=np.int64)

    assert train_rows.shape == expected_train.shape
    assert train_rows.dtype == np.int64
    np.testing.assert_array_equal(train_rows, expected_train)
    assert val_rows.shape == expected_val.shape
    assert val_rows.dtype == np.int64
    np.testing.assert_array_equal(val_rows, expected_val)
    assert test_rows.shape == expected_test.shape
    assert test_rows.dtype == np.int64
    np.testing.assert_array_equal(test_rows, expected_test)


def test_temporal_split_one_rating_stays_in_train(dataset: Dataset) -> None:
    split = temporal_split(dataset, val_fraction=0.25)
    user_index = dataset.user_to_index[40]
    assert int(np.count_nonzero(dataset.user_indices == user_index)) == 1
    train_rows = _rows_for_user(dataset, split.train, user_index)
    val_rows = _rows_for_user(dataset, split.val, user_index)
    test_rows = _rows_for_user(dataset, split.test, user_index)
    expected_train = np.array([5], dtype=np.int64)

    assert train_rows.shape == expected_train.shape
    assert train_rows.dtype == np.int64
    np.testing.assert_array_equal(train_rows, expected_train)
    assert val_rows.shape == (0,)
    assert val_rows.dtype == np.int64
    assert test_rows.shape == (0,)
    assert test_rows.dtype == np.int64


def test_temporal_split_zero_val_fraction_empties_val_and_keeps_test(
    dataset: Dataset,
) -> None:
    split = temporal_split(dataset, val_fraction=0)
    expected_train = np.array(
        [2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
        dtype=np.int64,
    )
    expected_val = np.array([], dtype=np.int64)
    expected_test = np.array([0, 1, 7], dtype=np.int64)

    assert split.train.shape == expected_train.shape
    assert split.train.dtype == np.int64
    np.testing.assert_array_equal(split.train, expected_train)
    assert split.val.shape == expected_val.shape
    assert split.val.dtype == np.int64
    np.testing.assert_array_equal(split.val, expected_val)
    assert split.test.shape == expected_test.shape
    assert split.test.dtype == np.int64
    np.testing.assert_array_equal(split.test, expected_test)


@pytest.mark.parametrize(
    ("test_fraction", "val_fraction", "message"),
    [
        (0, 0.25, "test_fraction"),
        (1, 0.25, "test_fraction"),
        (-0.1, 0.25, "test_fraction"),
        (1.1, 0.25, "test_fraction"),
        (0.2, -0.01, "val_fraction"),
        (0.2, 1, "val_fraction"),
        (0.2, 1.1, "val_fraction"),
    ],
    ids=[
        "test-zero",
        "test-one",
        "test-negative",
        "test-above-one",
        "val-negative",
        "val-one",
        "val-above-one",
    ],
)
def test_temporal_split_bad_fraction_raises(
    dataset: Dataset,
    test_fraction: float,
    val_fraction: float,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        temporal_split(
            dataset,
            test_fraction=test_fraction,
            val_fraction=val_fraction,
        )


@pytest.mark.parametrize(
    "piece",
    ["train", "val", "test"],
    ids=["train", "val", "test"],
)
def test_temporal_split_item_assignment_raises(
    dataset: Dataset, piece: str
) -> None:
    split = temporal_split(dataset, val_fraction=0.25)
    values = getattr(split, piece)
    with pytest.raises(ValueError, match="read-only"):
        values[0] = 0


@pytest.mark.skipif(not _ML_1M.is_dir(), reason="data/ml-1m is not downloaded")
def test_temporal_split_ml1m_default_sizes_match_dry_run(
    ml1m_dataset: Dataset,
) -> None:
    split = temporal_split(ml1m_dataset)
    assert split.train.shape == (724_973,)
    assert split.train.dtype == np.int64
    assert split.val.shape == (77_580,)
    assert split.val.dtype == np.int64
    assert split.test.shape == (197_656,)
    assert split.test.dtype == np.int64


@pytest.mark.skipif(not _ML_1M.is_dir(), reason="data/ml-1m is not downloaded")
def test_temporal_split_ml1m_pieces_cover_each_row_once(
    ml1m_dataset: Dataset,
) -> None:
    split = temporal_split(ml1m_dataset)
    combined = np.concatenate([split.train, split.val, split.test])
    combined.sort()
    expected = np.arange(len(ml1m_dataset), dtype=np.int64)
    assert combined.shape == expected.shape
    assert combined.dtype == np.int64
    np.testing.assert_array_equal(combined, expected)


@pytest.mark.skipif(not _ML_1M.is_dir(), reason="data/ml-1m is not downloaded")
def test_temporal_split_ml1m_groupby_finds_no_future_leakage(
    ml1m_dataset: Dataset,
) -> None:
    split = temporal_split(ml1m_dataset)
    bounds = _timestamp_bounds_by_user(ml1m_dataset, split)
    assert ml1m_dataset.n_users == 6_040
    assert bounds["user_index"].nunique() == ml1m_dataset.n_users
    assert bounds["n_rows"].sum() == len(ml1m_dataset)

    train_max = _piece_bound(bounds, "train", "timestamp_max")
    val_min = _piece_bound(bounds, "val", "timestamp_min")
    val_max = _piece_bound(bounds, "val", "timestamp_max")
    test_min = _piece_bound(bounds, "test", "timestamp_min")
    assert train_max.shape[0] == 6_040
    assert val_min.shape[0] == 6_040
    assert val_max.shape[0] == 6_040
    assert test_min.shape[0] == 6_040
    assert (train_max <= val_min).all()
    assert (val_max <= test_min).all()
