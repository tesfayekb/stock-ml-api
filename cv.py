


"""
Purged Walk-Forward Cross-Validation.

Reference: de Prado, "Advances in Financial Machine Learning" (2018), Ch. 7
"""

import numpy as np
import pandas as pd


class PurgedWalkForwardCV:
    """
    Walk-forward CV with purge + embargo to prevent information leakage.

    - purge_days: Remove training samples within N days BEFORE the validation fold start.
    - embargo_days: Remove training samples within N days AFTER the validation fold end.
    """

    def __init__(self, n_splits: int = 3, purge_days: int = 2, embargo_days: int = 1):
        self.n_splits = n_splits
        self.purge_days = purge_days
        self.embargo_days = embargo_days
        self.purged_samples = 0

    def split(self, X, dates=None):
        """
        Generate purged train/val indices.

        Args:
            X: Feature matrix (n_samples, n_features)
            dates: Array of date strings (YYYY-MM-DD). If None, falls back to index-based purge.

        Yields:
            (train_indices, val_indices) tuples
        """
        n = len(X)
        self.purged_samples = 0

        if dates is not None:
            date_arr = pd.to_datetime(dates)
            fold_size = n // (self.n_splits + 1)

            for i in range(self.n_splits):
                val_start = fold_size * (i + 1)
                val_end = min(val_start + fold_size, n)
                if val_end - val_start < 3:
                    continue

                val_start_date = date_arr[val_start]
                purge_start = val_start_date - pd.Timedelta(days=self.purge_days)

                train_mask = np.ones(val_start, dtype=bool)
                for j in range(val_start):
                    if date_arr[j] >= purge_start:
                        train_mask[j] = False
                        self.purged_samples += 1

                train_idx = np.where(train_mask)[0]
                val_idx = np.arange(val_start, val_end)

                if len(train_idx) >= 5 and len(val_idx) >= 3:
                    yield train_idx, val_idx
        else:
            fold_size = n // (self.n_splits + 1)
            for i in range(self.n_splits):
                val_start = fold_size * (i + 1)
                val_end = min(val_start + fold_size, n)

                purge_count = min(self.purge_days, val_start)
                train_end = val_start - purge_count
                self.purged_samples += purge_count

                train_idx = np.arange(0, train_end)
                val_idx = np.arange(val_start, val_end)

                if len(train_idx) >= 5 and len(val_idx) >= 3:
                    yield train_idx, val_idx

