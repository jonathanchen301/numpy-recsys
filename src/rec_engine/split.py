"""Per-user temporal train, validation, and test row positions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rec_engine.dataset import Dataset, _readonly


@dataclass(frozen=True, eq=False)
class Split:
    """Ascending, read-only row positions into one ``Dataset``.

    ``train``, ``val``, and ``test`` partition that dataset. Each array has
    dtype int64 and shape ``(n,)``. Item assignment raises ``ValueError``.

    Comparisons use identity. Array ``==`` is elementwise, so value equality
    would not return a bool.
    """

    train: np.ndarray  # shape (n_train,), int64, ascending, read-only
    val: np.ndarray
    test: np.ndarray

    def __repr__(self) -> str:
        return (
            f"Split(n_train={self.train.shape[0]}, "
            f"n_val={self.val.shape[0]}, n_test={self.test.shape[0]})"
        )


def temporal_split(
    dataset: Dataset,
    test_fraction: float = 0.2,
    val_fraction: float = 0.1,
) -> Split:
    """Cut each user's ratings into train, validation, and test by time.

    Within a user, rows are ordered by timestamp. Rows with the same
    timestamp keep file order, because the ordering sort is stable.
    Position 0 is that user's oldest rating.

    The last ``floor(n * test_fraction)`` rows are test. Of the rows that
    remain, the last ``floor(n_fit * val_fraction)`` are validation and the
    earlier ones are train. ``floor`` can leave a piece empty. With both
    defaults, a user with fewer than 5 ratings contributes no test rows and
    no validation rows, so every one of that user's rows stays in train.

    The returned positions index ``dataset`` and are sorted ascending, so
    every split can share its user, movie, and rating arrays. Time is
    ordered per user only: one user's train rating can be later than
    another user's test rating.

    ``test_fraction`` must satisfy ``0 < test_fraction < 1``.
    ``val_fraction`` must satisfy ``0 <= val_fraction < 1``. Any other
    value raises ``ValueError``.
    """
    if not 0 < test_fraction < 1:
        raise ValueError(
            "test_fraction must satisfy 0 < test_fraction < 1; "
            f"got {test_fraction!r}"
        )
    if not 0 <= val_fraction < 1:
        raise ValueError(
            "val_fraction must satisfy 0 <= val_fraction < 1; "
            f"got {val_fraction!r}"
        )

    # Last key is primary. Stable, so equal timestamps keep file order.
    order = np.lexsort((dataset.timestamps, dataset.user_indices))
    users = dataset.user_indices[order]
    counts = np.bincount(dataset.user_indices, minlength=dataset.n_users)
    # Start of each user's block in ``order``. Position 0 is the oldest row.
    starts = np.cumsum(counts) - counts
    position = np.arange(len(dataset), dtype=np.int64) - starts[users]

    n_test = np.floor(counts * test_fraction).astype(np.int64)
    n_fit = counts - n_test
    n_val = np.floor(n_fit * val_fraction).astype(np.int64)
    n_train = n_fit - n_val

    train_end = n_train[users]
    fit_end = n_fit[users]
    # Train is the oldest slice, then validation, then test.
    train_mask = position < train_end
    val_mask = (position >= train_end) & (position < fit_end)
    test_mask = position >= fit_end
    return Split(
        _row_positions(order, train_mask),
        _row_positions(order, val_mask),
        _row_positions(order, test_mask),
    )


def _row_positions(order: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Sorted ascending row indices for ``mask``, as a read-only int64 array."""
    return _readonly(np.sort(order[mask]), np.int64)
