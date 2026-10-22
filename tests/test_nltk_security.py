# Copyright (c) Microsoft. All rights reserved.

from pathlib import Path

import pytest

nltk = pytest.importorskip("nltk")
pathsec = pytest.importorskip("nltk.pathsec")
perceptron = pytest.importorskip("nltk.tag.perceptron")


@pytest.fixture
def model_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "models"
    root.mkdir(mode=0o700)
    monkeypatch.setattr(nltk.data, "path", [str(root)])
    monkeypatch.setattr(pathsec, "ENFORCE", True)
    return root


def test_perceptron_model_roundtrip(model_root: Path) -> None:
    model = perceptron.AveragedPerceptron()
    model.weights = {"bias": {"label": 1.0}}
    filename = str(model_root / "model.json")
    model.save(filename)
    loaded = perceptron.AveragedPerceptron()
    loaded.load(filename)
    assert loaded.weights == model.weights


@pytest.mark.parametrize("operation", ["save", "load"])
@pytest.mark.parametrize("symlink", [False, True])
def test_perceptron_rejects_outside_model_paths(model_root: Path, operation: str, symlink: bool) -> None:
    outside = model_root.parent / "outside.json"
    original = '{"private": {"label": 7.0}}'
    outside.write_text(original)
    target = outside
    if symlink:
        target = model_root / "link.json"
        target.symlink_to(outside)
    model = perceptron.AveragedPerceptron()
    model.weights = {"bias": {"label": 1.0}}
    with pytest.raises(PermissionError):
        getattr(model, operation)(str(target))
    assert outside.read_text() == original
    assert model.weights == {"bias": {"label": 1.0}}


def test_sql_word_tokenization_without_external_models() -> None:
    assert nltk.word_tokenize("SELECT name FROM users WHERE age > 18;", preserve_line=True) == [
        "SELECT",
        "name",
        "FROM",
        "users",
        "WHERE",
        "age",
        ">",
        "18",
        ";",
    ]
