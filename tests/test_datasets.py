"""Tests for datasets module."""

from __future__ import annotations

import os
import tempfile

from unicad_torch.datasets import find_datasets


class TestFindDatasets:
    def test_finds_npz_files(self):
        with tempfile.TemporaryDirectory() as d:
            for name in ["a.npz", "b.npz", "c.txt"]:
                open(os.path.join(d, name), "w").close()
            results = find_datasets(d)
            names = [n for n, _ in results]
            assert "a" in names
            assert "b" in names
            assert "c" not in names

    def test_finds_csv_files(self):
        with tempfile.TemporaryDirectory() as d:
            for name in ["a.csv", "b.npz"]:
                open(os.path.join(d, name), "w").close()
            results = find_datasets(d, extensions=(".npz", ".csv"))
            names = [n for n, _ in results]
            assert "a" in names
            assert "b" in names

    def test_filter_by_name(self):
        with tempfile.TemporaryDirectory() as d:
            for name in ["38_thyroid.npz", "2_annthyroid.npz", "other.npz"]:
                open(os.path.join(d, name), "w").close()
            results = find_datasets(d, datasets=["38_thyroid"])
            assert len(results) == 1
            assert results[0][0] == "38_thyroid"

    def test_missing_dataset_warning(self, capsys):
        with tempfile.TemporaryDirectory() as d:
            for name in ["a.npz"]:
                open(os.path.join(d, name), "w").close()
            find_datasets(d, datasets=["nonexistent"])
            captured = capsys.readouterr()
            assert "not found" in captured.out

    def test_nonexistent_dir(self, capsys):
        results = find_datasets("/nonexistent/path")
        assert results == []
