# Copyright (c) Microsoft. All rights reserved.

from pathlib import Path
from uuid import uuid4

import pytest


@pytest.mark.parametrize("sink", ["reference", "template", "generator"])
def test_reference_templates_reject_python_object_access(sink: str) -> None:
    fsspec = pytest.importorskip("fsspec")
    exceptions = pytest.importorskip("jinja2.exceptions")
    payload = "{{ joiner.__init__.__globals__ }}"
    document: dict[str, object] = {"version": 1, "templates": {"base": "memory://safe"}, "refs": {}}
    if sink == "reference":
        document["refs"] = {"data": [payload]}
    elif sink == "template":
        document["templates"] = {"base": payload}
        document["refs"] = {"data": ["{{ base() }}"]}
    else:
        document["gen"] = [{"key": payload, "url": "memory://safe", "dimensions": {"i": [0]}}]
    with pytest.raises(exceptions.SecurityError):
        fsspec.filesystem("reference", fo=document, simple_templates=False)


def test_reference_templates_preserve_byte_ranges() -> None:
    fsspec = pytest.importorskip("fsspec")
    path = f"memory://{uuid4().hex}/data"
    with fsspec.open(path, "wb") as handle:
        handle.write(b"abcdef")
    filesystem = fsspec.filesystem(
        "reference",
        fo={
            "version": 1,
            "templates": {"base": path},
            "refs": {"head": ["{{ base }}", 0, 3]},
            "gen": [
                {
                    "key": "part{{ i }}",
                    "url": "{{ base }}",
                    "offset": "{{ i }}",
                    "length": "2",
                    "dimensions": {"i": [3]},
                }
            ],
        },
        simple_templates=False,
    )
    assert filesystem.cat_file("head") == b"abc"
    assert filesystem.cat_file("part3") == b"de"


def test_datasets_local_parquet_and_memory_roundtrip(tmp_path: Path) -> None:
    datasets = pytest.importorskip("datasets")
    expected = {"prompt": ["one", "two"], "reward": [1.0, 0.0]}
    dataset = datasets.Dataset.from_dict(expected)
    parquet = tmp_path / "training.parquet"
    dataset.to_parquet(str(parquet))
    loaded = datasets.load_dataset("parquet", data_files=str(parquet), split="train")
    assert loaded.to_dict() == expected
    destination = f"memory://{uuid4().hex}/training"
    loaded.save_to_disk(destination)
    assert datasets.load_from_disk(destination).to_dict() == expected
