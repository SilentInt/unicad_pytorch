"""Dataset discovery for benchmark and inference scripts."""

from __future__ import annotations

import os


def find_datasets(
    data_dir: str,
    datasets: list[str] | None = None,
    extensions: tuple[str, ...] = (".npz",),
) -> list[tuple[str, str]]:
    """Find data files in a directory, optionally filtered by name.

    Args:
        data_dir: Directory to search recursively.
        datasets: If provided, only include files whose name (without extension)
            matches one of these strings.
        extensions: File extensions to include (e.g. (".npz", ".csv")).

    Returns:
        List of (name, path) pairs sorted by name.
    """
    results: list[tuple[str, str]] = []
    if not os.path.isdir(data_dir):
        print(f"Data directory not found: {data_dir}")
        return results

    for root, _dirs, files in os.walk(data_dir):
        for f in sorted(files):
            if not any(f.endswith(ext) for ext in extensions):
                continue
            name = os.path.splitext(f)[0]
            path = os.path.join(root, f)
            if datasets is None or name in datasets or f in datasets:
                results.append((name, path))

    if datasets is not None:
        requested = set(datasets)
        found = {name for name, _ in results}
        missing = requested - found
        if missing:
            print(f"Warning: datasets not found: {', '.join(sorted(missing))}")

    return results
