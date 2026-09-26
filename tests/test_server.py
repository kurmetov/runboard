import json
import threading
import urllib.error
import urllib.request

import pytest

from runboard import Run
from runboard.config import load
from runboard.server import make_server


@pytest.fixture
def project(tmp_path):
    (tmp_path / "docs").mkdir()
    (tmp_path / "runboard.toml").write_text(
        'title = "demo"\nlang = "ru"\ndoc_order = ["plan"]\nprereg = ["docs/experiments.md"]\n'
        '[sync]\nlabel = "Pull"\ncommand = "echo synced"\n'
    )
    with Run("grid", {"grid": {"lr": [0.1, 0.2], "seeds": [0]}}, tags={"step": "S1.2"},
             root=tmp_path / "runs", repo=tmp_path) as run:
        for lr in (0.1, 0.2):
            run.log({"acc": lr * 3}, lr=lr, seed=0)
        run.finish({"best": 0.6})
    (tmp_path / "docs" / "plan.md").write_text("# The plan\n")
    (tmp_path / "docs" / "experiments.md").write_text(
        f"# Registry\n\n## S1.1 · other\n\ntext\n\n## S1.2 · grid search\n\nprediction\n\n## S1.3 · cites {run.run_id}\n"
    )
    (tmp_path / "secret.txt").write_text("nope")
    return tmp_path, run.run_id


@pytest.fixture
def client(project):
    root, run_id = project
    server = make_server(load(root), port=0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"

    def get(path, method="GET"):
        req = urllib.request.Request(base + path, method=method)
        with urllib.request.urlopen(req) as r:
            body = r.read()
            return json.loads(body) if r.headers["Content-Type"].startswith("application/json") else body

    yield get, run_id
    server.shutdown()


def test_config_and_index(client):
    get, _ = client
    cfg = get("/api/config")
    assert cfg["title"] == "demo" and cfg["lang"] == "ru" and cfg["sync_label"] == "Pull" and cfg["has_docs"]
    assert b"runboard" in get("/")


def test_runs_and_run_detail(client):
    get, run_id = client
    runs = get("/api/runs")
    assert [r["run_id"] for r in runs] == [run_id] and runs[0]["status"] == "ok"
    detail = get(f"/api/run/{run_id}")
    assert detail["meta"]["dims"] == ["lr", "seed"]
    # A section citing the run id wins over the tag match.
    assert detail["preregistration"]["text"].startswith("## S1.3")
    assert detail["preregistration"]["file"] == "docs/experiments.md"
    assert len(get(f"/api/run/{run_id}/metrics")) == 2
    assert get(f"/api/run/{run_id}/file/config.json")["text"].startswith("{")


def test_prereg_falls_back_to_tag(project):
    from runboard.server import Board
    root, run_id = project
    board = Board(load(root))
    text = (root / "docs" / "experiments.md").read_text().replace(run_id, "something-else")
    (root / "docs" / "experiments.md").write_text(text)
    assert board.run(run_id)["preregistration"]["text"].startswith("## S1.2")


def test_docs_order(client):
    get, _ = client
    assert [d["name"] for d in get("/api/docs")] == ["plan", "experiments"]
    assert get("/api/doc/plan")["text"] == "# The plan\n"


@pytest.mark.parametrize("path", [
    "/api/run/..%2F..%2Fsecret.txt",
    "/api/run/{run}/file/..%2F..%2Fsecret.txt",
    "/api/run/{run}/file/../../secret.txt",
    "/api/doc/..%2Fsecret",
    "/api/run/nope",
])
def test_no_path_traversal(client, path):
    get, run_id = client
    with pytest.raises(urllib.error.HTTPError) as e:
        get(path.format(run=run_id))
    assert e.value.code == 404


def test_sync_runs_configured_command(client):
    get, _ = client
    res = get("/api/sync", method="POST")
    assert res["ok"] and "synced" in res["output"]


def test_defaults_without_config(tmp_path):
    cfg = load(tmp_path)
    assert cfg.title == tmp_path.name and cfg.runs == tmp_path / "runs"
    assert cfg.docs is None and cfg.sync_command is None and cfg.lang == "en"
