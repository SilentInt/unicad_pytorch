"""Tests for the run directory management module."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from unicad_torch.config import GalaxyConfig
from unicad_torch.run import (
    create_run_dir,
    find_runs,
    generate_run_name,
    load_summary,
    resolve_resume_path,
    save_config,
    save_history,
    save_summary,
)


class TestGenerateRunName:
    def test_no_tag(self):
        name = generate_run_name()
        assert len(name) == 19  # YYYY-MM-DD_HH-MM-SS
        assert "_" in name

    def test_with_tag(self):
        name = generate_run_name(tag="exp1")
        assert name.endswith("_exp1")

    def test_tag_sanitized(self):
        name = generate_run_name(tag="my experiment/v2")
        assert " " not in name
        assert "/" not in name


class TestCreateRunDir:
    def test_creates_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = create_run_dir(tmpdir)
            assert run_dir.is_dir()
            assert (run_dir / "checkpoints").is_dir()

    def test_with_tag(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = create_run_dir(tmpdir, tag="test")
            assert "_test" in run_dir.name

    def test_collision_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Pre-create the exact directory that create_run_dir would make
            run_name = generate_run_name("collision_test")
            (Path(tmpdir) / run_name).mkdir()
            # The timestamp will differ, so mock generate_run_name
            import unittest.mock

            with unittest.mock.patch(
                "unicad_torch.run.generate_run_name", return_value=run_name
            ):
                with pytest.raises(FileExistsError):
                    create_run_dir(tmpdir, tag="collision_test")


class TestSaveConfig:
    def test_saves_yaml(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            config = GalaxyConfig(k=5, em_iters=2)
            save_config(config, run_dir)
            # Should have either config.yaml or config.json
            yaml_path = run_dir / "config.yaml"
            json_path = run_dir / "config.json"
            assert yaml_path.exists() or json_path.exists()
            if yaml_path.exists():
                content = yaml_path.read_text()
                assert "k: 5" in content
            else:
                content = json.loads(json_path.read_text())
                assert content["k"] == 5

    def test_excludes_callbacks(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            from unicad_torch.callbacks import History

            config = GalaxyConfig(callbacks=[History()])
            save_config(config, run_dir)
            # Check whichever file was created
            for fname in ("config.yaml", "config.json"):
                fpath = run_dir / fname
                if fpath.exists():
                    content = fpath.read_text()
                    assert "callbacks" not in content


class TestSaveHistory:
    def test_saves_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            history = {"pretrain_loss": [1.0, 0.5, 0.3], "n_excluded": [5, 3]}
            save_history(history, run_dir)
            path = run_dir / "history.json"
            assert path.exists()
            loaded = json.loads(path.read_text())
            assert loaded["pretrain_loss"] == [1.0, 0.5, 0.3]


class TestSaveSummary:
    def test_saves_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            save_summary(
                run_dir,
                fit_time=12.34,
                threshold=-3.45,
                n_samples=100,
                n_features=6,
            )
            path = run_dir / "summary.json"
            assert path.exists()
            loaded = json.loads(path.read_text())
            assert loaded["run_name"] == run_dir.name
            assert loaded["fit_duration_seconds"] == 12.34
            assert loaded["threshold"] == -3.45
            assert loaded["n_samples"] == 100

    def test_minimal_fields(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            save_summary(run_dir)
            loaded = json.loads((run_dir / "summary.json").read_text())
            assert "run_name" in loaded
            assert "timestamp" in loaded


class TestResolveResumePath:
    def test_file_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pt_file = Path(tmpdir) / "model.pt"
            pt_file.write_text("fake")
            result = resolve_resume_path(pt_file)
            assert result == pt_file

    def test_directory_with_checkpoints(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_dir = Path(tmpdir) / "checkpoints"
            ckpt_dir.mkdir()
            (ckpt_dir / "latest.pt").write_text("fake")
            result = resolve_resume_path(tmpdir)
            assert result == ckpt_dir / "latest.pt"

    def test_directory_with_model_pt(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "model.pt").write_text("fake")
            result = resolve_resume_path(tmpdir)
            assert result == Path(tmpdir) / "model.pt"

    def test_directory_empty_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(FileNotFoundError, match="No checkpoint"):
                resolve_resume_path(tmpdir)


class TestFindRuns:
    def test_finds_runs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create two runs
            r1 = create_run_dir(tmpdir, tag="first")
            save_summary(r1)
            r2 = create_run_dir(tmpdir, tag="second")
            save_summary(r2)

            runs = find_runs(tmpdir)
            assert len(runs) == 2
            # Sorted newest-first
            assert runs[0].name.endswith("_second")

    def test_empty_root(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            runs = find_runs(tmpdir)
            assert runs == []

    def test_nonexistent_root(self):
        runs = find_runs("/nonexistent/path")
        assert runs == []

    def test_ignores_dirs_without_summary(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "some_dir").mkdir()
            runs = find_runs(tmpdir)
            assert runs == []


class TestLoadSummary:
    def test_loads_existing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            save_summary(run_dir, threshold=1.23)
            summary = load_summary(run_dir)
            assert summary["threshold"] == 1.23

    def test_missing_returns_empty(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            summary = load_summary(Path(tmpdir))
            assert summary == {}
