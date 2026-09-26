import json
import subprocess
import tarfile

import pytest

from runboard import Run


def _git(repo, *args):
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path):
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.email", "t@t")
    _git(tmp_path, "config", "user.name", "t")
    (tmp_path / "train.py").write_text("print('hi')\n")
    (tmp_path / ".gitignore").write_text("data/\nruns/*/artifacts/\n")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-qm", "init")
    return tmp_path


def test_run_writes_record(repo):
    with Run("exp", {"lr": 0.1, "grid": {"lr": [0.1], "seeds": [0]}}, tags={"step": "S1"}, repo=repo) as run:
        run.log({"loss": 0.5}, step=1)
        run.finish({"best": 0.5})
    assert run.dir.parent == repo / "runs"
    assert json.loads((run.dir / "config.json").read_text())["lr"] == 0.1
    meta = json.loads((run.dir / "meta.json").read_text())
    assert meta["git_dirty"] is False and meta["format"] == 1
    assert meta["dims"] == ["lr", "seed"]  # from config["grid"], "seeds" -> "seed"
    assert not (run.dir / "code_snapshot.tar.gz").exists()
    assert json.loads((run.dir / "summary.json").read_text())["best"] == 0.5
    assert json.loads((run.dir / "metrics.jsonl").read_text().splitlines()[0])["step"] == 1
    reg = [json.loads(l) for l in (repo / "runs" / "registry.jsonl").read_text().splitlines()]
    assert len(reg) == 1 and reg[0]["status"] == "ok"


def test_dirty_tree_saves_diff_and_snapshot(repo):
    (repo / "train.py").write_text("print('changed')\n")
    (repo / "new_module.py").write_text("x = 1\n")
    (repo / "data").mkdir()
    (repo / "data" / "big.bin").write_bytes(b"0" * 10)
    with Run("exp", {}, repo=repo) as run:
        pass
    meta = json.loads((run.dir / "meta.json").read_text())
    assert meta["git_dirty"] is True
    assert "changed" in (run.dir / "git.diff").read_text()
    with tarfile.open(run.dir / "code_snapshot.tar.gz") as tar:
        names = set(tar.getnames())
    assert {"train.py", "new_module.py"} <= names
    assert not any(n.startswith(("data", "runs")) for n in names)  # ignored data and the runs dir itself


def test_snapshot_can_be_limited_or_disabled(repo):
    (repo / "a.py").write_text("a\n")
    (repo / "b.py").write_text("b\n")
    with Run("exp", {}, repo=repo, snapshot=["a.py"]) as run:
        pass
    with tarfile.open(run.dir / "code_snapshot.tar.gz") as tar:
        assert tar.getnames() == ["a.py"]
    with Run("exp2", {}, repo=repo, snapshot=False) as run2:
        pass
    assert not (run2.dir / "code_snapshot.tar.gz").exists()


def test_failed_run_is_recorded(tmp_path):
    with pytest.raises(RuntimeError):
        with Run("exp", {}, root=tmp_path / "r", repo=tmp_path):
            raise RuntimeError("boom")
    reg = json.loads((tmp_path / "r" / "registry.jsonl").read_text())
    assert reg["status"] == "failed"


def test_numpy_values_are_serialized(tmp_path):
    np = pytest.importorskip("numpy")
    with Run("exp", {"a": np.float32(1.5)}, root=tmp_path / "r", repo=tmp_path) as run:
        run.log({"m": np.float64(0.25), "v": np.arange(3)})
    row = json.loads((run.dir / "metrics.jsonl").read_text())
    assert row["m"] == 0.25 and row["v"] == [0, 1, 2]


def test_runs_started_in_the_same_second_get_unique_ids(tmp_path):
    ids = []
    for _ in range(3):
        with Run("exp", {}, root=tmp_path / "r", repo=tmp_path) as run:
            ids.append(run.run_id)
    assert len(set(ids)) == 3
