"""Saved results identify the installed source, independently of the caller's cwd."""

import hashlib
import json
import subprocess
from pathlib import Path

from exobalancelab import simulation


def _repository(directory):
    subprocess.run(["git", "init", "--quiet", str(directory)], check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Test Fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "commit",
            "--quiet",
            "--allow-empty",
            "-m",
            "Unrelated repository",
        ],
        cwd=directory,
        check=True,
    )


def _save(directory):
    simulation.run(simulation.RunConfig(duration_s=0.01)).save(directory)
    return json.loads((directory / "provenance.json").read_text())


def test_source_checkout_is_recorded_from_an_unrelated_repository(tmp_path, monkeypatch):
    source_root = Path(simulation.__file__).resolve().parents[1]
    expected = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source_root, text=True).strip()
    expected_dirty = bool(
        subprocess.check_output(["git", "status", "--porcelain"], cwd=source_root, text=True).strip()
    )
    _repository(tmp_path)
    monkeypatch.chdir(tmp_path)
    saved = _save(tmp_path / "episode")
    assert saved["git_commit"] == expected
    assert saved["working_tree_dirty"] == expected_dirty


def test_wheel_does_not_claim_the_callers_or_parent_repository(tmp_path, monkeypatch):
    _repository(tmp_path)
    source = tmp_path / "site-packages" / "exobalancelab" / "simulation.py"
    source.parent.mkdir(parents=True)
    source.write_bytes(Path(simulation.__file__).read_bytes())
    monkeypatch.setattr(simulation, "__file__", str(source))
    monkeypatch.chdir(tmp_path)
    saved = _save(tmp_path / "episode")
    assert saved["git_commit"] is None
    assert saved["working_tree_dirty"] is None
    assert saved["source_sha256"]["simulation.py"] == hashlib.sha256(source.read_bytes()).hexdigest()


def test_git_is_optional_for_saving_results(tmp_path, monkeypatch):
    def unavailable(*args, **kwargs):
        raise FileNotFoundError("git")

    monkeypatch.setattr(simulation.subprocess, "run", unavailable)
    saved = _save(tmp_path / "episode")
    assert saved["git_commit"] is None
    assert saved["working_tree_dirty"] is None
    assert saved["source_sha256"]
    assert saved["model_sha256"]
