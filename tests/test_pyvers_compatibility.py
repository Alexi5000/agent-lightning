# Copyright (c) Microsoft. All rights reserved.

import hashlib
import importlib.metadata
import json
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest
from packaging.version import Version

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "vendor" / "pyvers"


def test_vendored_upstream_provenance():
    provenance = json.loads((VENDOR / "provenance.json").read_text())
    for name, expected in provenance["files"].items():
        assert hashlib.sha256((VENDOR / name).read_bytes()).hexdigest() == expected, name
    assert set(provenance["files"]) == {str(path.relative_to(VENDOR)) for path in VENDOR.rglob("*.py")} | {"LICENSE"}


def test_installed_compatibility_graph():
    pytest.importorskip("pyvers")
    pytest.importorskip("tensordict")
    assert importlib.metadata.version("pyvers") == "0.1.0+agl.1"
    assert Version("0.10") <= Version(importlib.metadata.version("tensordict")) < Version("0.11")
    assert Version(importlib.metadata.version("packaging")) >= Version("26.3")
    assert Version(importlib.metadata.version("virtualenv")) >= Version("21.14.9")
    lock = tomllib.loads((ROOT / "uv.lock").read_text())
    td = next(package for package in lock["package"] if package["name"] == "tensordict")
    assert any("macosx_14_0_arm64" in wheel["url"] for wheel in td["wheels"])


@pytest.mark.parametrize(
    ("version", "lower", "upper", "matches"),
    [
        ("2.9.0", "2.9", "2.10", True),
        ("2.10.0", "2.9", "2.10", False),
        ("2.8.9", "2.9", None, False),
        ("2.9.0+cpu", "2.9", "2.10", True),
        ("2.9.0.dev20250901+cu128", "2.9", "2.10", True),
        ("2.9.0rc1", "2.9", "2.10", True),
        ("2.9.0.post1", "2.9", "2.10", True),
        ("0.10", None, "0.11", True),
    ],
)
def test_version_dispatch_boundaries(version, lower, upper, matches):
    pyvers = pytest.importorskip("pyvers")
    assert pyvers.implement_for.check_version(version, lower, upper) is matches


def test_upstream_dispatch_and_backend_contracts():
    pytest.importorskip("pyvers")
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "--override-ini=addopts=", str(VENDOR / "tests")],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "27 passed" in result.stdout


def test_real_tensordict_operations(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("tensordict")
    import torch
    from tensordict import TensorDict, TensorDictBase

    data = TensorDict({"values": torch.arange(6).reshape(2, 3)}, batch_size=[2])
    clone = data.clone()
    clone["values"][0, 0] = 99
    assert data["values"][0, 0].item() == 0
    row = data[0]
    assert isinstance(row, TensorDictBase)
    assert row["values"].tolist() == [0, 1, 2]
    stacked = torch.stack([data, data], 0)
    assert isinstance(stacked, TensorDictBase)
    assert stacked.batch_size == torch.Size([2, 2])
    concatenated = torch.cat([data, data], 0)
    assert isinstance(concatenated, TensorDictBase)
    assert concatenated.batch_size == torch.Size([4])
    destination = tmp_path / "tensor-data"
    data.memmap(destination)
    restored = TensorDict.load_memmap(destination)
    assert torch.equal(restored["values"], data["values"])
