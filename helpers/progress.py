"""Tiny progress-bar wrapper so scripts use the same tqdm settings."""

from tqdm import tqdm


def progress_bar(iterable, desc=None, total=None):
    """Return a tqdm progress bar with terminal-friendly sizing."""
    return tqdm(iterable, desc=desc, total=total, dynamic_ncols=True)
