"""Training run directory management.

Provides utilities for organizing training runs into timestamped
directories, each containing model, config, history, and summary
artifacts. Used by the CLI (``galaxy-train``, ``galaxy-runs``) but
also usable programmatically.

Typical directory layout::

    runs/
    └── 2026-04-29_14-30-22/
        ├── config.yaml
        ├── model.pt
        ├── history.json
        ├── summary.json
        └── checkpoints/
            └── latest.pt
"""

from __future__ import annotations

import dataclasses
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from unicad_torch.config import GalaxyConfig


def generate_run_name(tag: str | None = None) -> str:
    """Generate a run directory name: ``YYYY-MM-DD_HH-MM-SS[_tag]``."""
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if tag:
        safe_tag = tag.replace(" ", "_").replace("/", "_").replace("\\", "_")
        return f"{ts}_{safe_tag}"
    return ts


def create_run_dir(root: str | Path, tag: str | None = None) -> Path:
    """Create and return a new run directory under *root*.

    Creates ``{root}/{run_name}/`` and ``{root}/{run_name}/checkpoints/``.
    """
    root = Path(root)
    run_name = generate_run_name(tag)
    run_dir = root / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "checkpoints").mkdir()
    return run_dir


def save_config(config: GalaxyConfig, run_dir: Path) -> None:
    """Save config as YAML (or JSON fallback) to ``{run_dir}/config.yaml``."""
    config_dict = dataclasses.asdict(config)
    config_dict.pop("callbacks", None)

    try:
        import yaml

        (run_dir / "config.yaml").write_text(
            yaml.dump(config_dict, default_flow_style=False)
        )
    except ImportError:
        (run_dir / "config.json").write_text(json.dumps(config_dict, indent=2))


def save_history(history: dict[str, list[float]], run_dir: Path) -> None:
    """Save training history as JSON to ``{run_dir}/history.json``."""
    (run_dir / "history.json").write_text(json.dumps(history, indent=2))


def _get_git_info() -> dict[str, str | bool | None]:
    """Best-effort git commit and dirty status."""
    result: dict[str, str | bool | None] = {"git_commit": None, "git_dirty": None}
    try:
        commit = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
        result["git_commit"] = commit
        dirty = (
            subprocess.check_output(
                ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
        result["git_dirty"] = bool(dirty)
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    return result


def _get_version() -> str | None:
    try:
        from unicad_torch import __version__

        return __version__
    except Exception:
        return None


def save_summary(
    run_dir: Path,
    *,
    fit_time: float | None = None,
    threshold: float | None = None,
    train_score_mean: float | None = None,
    train_score_std: float | None = None,
    effective_outlier_ratio: float | None = None,
    outlier_ratio_source: str | None = None,
    cli_args: list[str] | None = None,
    data_path: str | None = None,
    n_samples: int | None = None,
    n_features: int | None = None,
) -> None:
    """Save run summary to ``{run_dir}/summary.json``."""
    summary: dict[str, Any] = {
        "run_name": run_dir.name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    if fit_time is not None:
        summary["fit_duration_seconds"] = round(fit_time, 2)
    if threshold is not None:
        summary["threshold"] = threshold
    if train_score_mean is not None:
        summary["train_score_mean"] = train_score_mean
    if train_score_std is not None:
        summary["train_score_std"] = train_score_std
    if effective_outlier_ratio is not None:
        summary["effective_outlier_ratio"] = effective_outlier_ratio
    if outlier_ratio_source is not None:
        summary["outlier_ratio_source"] = outlier_ratio_source
    if data_path is not None:
        summary["data_path"] = data_path
    if n_samples is not None:
        summary["n_samples"] = n_samples
    if n_features is not None:
        summary["n_features"] = n_features
    if cli_args is not None:
        summary["cli_args"] = cli_args

    summary.update(_get_git_info())
    summary["unicad_torch_version"] = _get_version()

    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))


def resolve_resume_path(resume: str | Path) -> Path:
    """Resolve a ``--resume`` argument.

    If *resume* is a directory, look for ``{dir}/checkpoints/latest.pt``,
    then ``{dir}/model.pt``.  If it is a file, use it directly.
    """
    resume = Path(resume)
    if resume.is_dir():
        checkpoint = resume / "checkpoints" / "latest.pt"
        if checkpoint.exists():
            return checkpoint
        model_pt = resume / "model.pt"
        if model_pt.exists():
            return model_pt
        raise FileNotFoundError(
            f"No checkpoint found in run directory {resume}. "
            f"Expected checkpoints/latest.pt or model.pt"
        )
    return resume


def find_runs(root: str | Path) -> list[Path]:
    """List all run directories under *root*, sorted newest-first."""
    root = Path(root)
    if not root.is_dir():
        return []
    return [
        p
        for p in sorted(root.iterdir(), reverse=True)
        if p.is_dir() and (p / "summary.json").exists()
    ]


def load_summary(run_dir: Path) -> dict[str, Any]:
    """Load ``summary.json`` from a run directory."""
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        return {}
    return json.loads(summary_path.read_text())
