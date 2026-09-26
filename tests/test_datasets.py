import gzip
import json

import pytest

from runboard import datasets as ds
from runboard.config import load
from runboard.server import Board

# Triangle a-b-c plus a pendant d (c->d); node attribute "group".
EDGES = [("a", "b"), ("b", "c"), ("c", "a"), ("c", "d")]
GROUPS = {"a": "x", "b": "x", "c": "y", "d": "y"}


def write_formats(d):
    d.mkdir(parents=True, exist_ok=True)
    (d / "edges.csv").write_text("src,dst,amount\n" + "".join(f"{u},{v},{i + 1.5}\n" for i, (u, v) in enumerate(EDGES)))
    (d / "nodes.csv").write_text("user_id,group\n" + "".join(f"{k},{g}\n" for k, g in GROUPS.items()))
    (d / "edges.tsv.gz").write_bytes(gzip.compress(("from\tto\n" + "".join(f"{u}\t{v}\n" for u, v in EDGES)).encode()))
    (d / "g.edgelist").write_text("# comment\n" + "".join(f"{u} {v}\n" for u, v in EDGES))
    nodes = "".join(f'<node id="{k}"><data key="g">{v}</data></node>' for k, v in GROUPS.items())
    edges = "".join(f'<edge source="{u}" target="{v}"/>' for u, v in EDGES)
    (d / "g.graphml").write_text(
        '<?xml version="1.0"?><graphml xmlns="http://graphml.graphdrawing.org/xmlns">'
        '<key id="g" for="node" attr.name="group" attr.type="string"/>'
        f'<graph edgedefault="undirected">{nodes}{edges}</graph></graphml>')
    gnodes = "".join(f'<node id="{k}" label="{k}"><attvalues><attvalue for="0" value="{v}"/></attvalues></node>'
                     for k, v in GROUPS.items())
    gedges = "".join(f'<edge id="{i}" source="{u}" target="{v}"/>' for i, (u, v) in enumerate(EDGES))
    (d / "g.gexf").write_text(
        '<?xml version="1.0"?><gexf xmlns="http://gexf.net/1.3"><graph defaultedgetype="directed">'
        '<attributes class="node"><attribute id="0" title="group" type="string"/></attributes>'
        f'<nodes>{gnodes}</nodes><edges>{gedges}</edges></graph></gexf>')
    gml_nodes = "".join(f'  node [ id {i} label "{k}" group "{v}" ]\n' for i, (k, v) in enumerate(GROUPS.items()))
    idx = {k: i for i, k in enumerate(GROUPS)}
    gml_edges = "".join(f"  edge [ source {idx[u]} target {idx[v]} ]\n" for u, v in EDGES)
    (d / "g.gml").write_text(f"graph [\n  directed 1\n{gml_nodes}{gml_edges}]\n")
    (d / "g.json").write_text(json.dumps({
        "directed": True, "nodes": [{"id": k, "group": v} for k, v in GROUPS.items()],
        "links": [{"source": u, "target": v} for u, v in EDGES]}))
    (d / "m.mtx").write_text("%%MatrixMarket matrix coordinate pattern symmetric\n% c\n4 4 4\n1 2\n2 3\n3 1\n3 4\n")
    (d / "rows.jsonl").write_text("\n".join(json.dumps({"x": i, "tag": "ab"[i % 2], "nested": {"k": i}}) for i in range(30)))
    (d / "arr.json").write_text(json.dumps([{"a": 1, "b": None}, {"a": 2, "b": "z"}]))
    (d / "model.pkl").write_bytes(b"\x80\x04K\x01.")
    (d / "notes.txt").write_text("ignored")


@pytest.fixture
def board(tmp_path):
    write_formats(tmp_path / "data")
    (tmp_path / "secret.csv").write_text("a\n1\n")
    return Board(load(tmp_path))


def test_listing_detects_formats(board):
    listed = {d["path"]: d for d in board.datasets()}
    assert "data/notes.txt" not in listed and "secret.csv" not in listed
    assert listed["data/edges.tsv.gz"]["format"] == "tsv"
    assert listed["data/model.pkl"]["kind"] == "unsafe"
    assert listed["data/g.graphml"]["kind"] == "graph"


@pytest.mark.parametrize("name,directed", [
    ("edges.csv", True), ("edges.tsv.gz", True), ("g.edgelist", True), ("g.graphml", False),
    ("g.gexf", True), ("g.gml", True), ("g.json", True), ("m.mtx", False),
])
def test_graph_formats(board, name, directed):
    res = board.graph(f"data/{name}", None, None, 2, 100)
    assert "error" not in res, res
    s = res["summary"]
    assert (s["nodes"], s["edges"], s["directed"]) == (4, 4, directed)
    assert s["components"] == 1 and s["largest_component"] == 4
    assert s["degree"]["max"] == 3  # c
    assert len(res["sample"]["nodes"]) == 4 and len(res["sample"]["edges"]) == 4


def test_graph_node_attributes(board):
    for name in ("g.graphml", "g.gexf", "g.gml", "g.json"):
        nodes = board.graph(f"data/{name}", None, None, 2, 100)["sample"]["nodes"]
        groups = {n["attrs"].get("label", n["id"]): n["attrs"]["group"] for n in nodes}
        assert groups == GROUPS, name


def test_node_table_join(board):
    res = board.graph("data/edges.csv", None, "d", 1, 100, "data/nodes.csv", "user_id")
    assert res["sample"]["center"] == "d"
    assert {n["id"] for n in res["sample"]["nodes"]} == {"c", "d"}  # 1-hop ego network
    assert all(n["attrs"]["group"] == GROUPS[n["id"]] for n in res["sample"]["nodes"])
    assert "group" in res["summary"]["attributes"]


def test_table_summary_and_paging(board):
    info = board.dataset("data/edges.csv", None)
    assert info["graph"] is True and info["columns"] == ["src", "dst", "amount"]
    assert info["total_rows"] == 4 and info["stats"]["amount"]["type"] == "float"
    assert info["stats"]["src"]["type"] == "str" and info["stats"]["src"]["unique"] == 3
    rows = board.dataset_rows("data/rows.jsonl", None, 10, 5)["rows"]
    assert [r[0] for r in rows] == [10, 11, 12, 13, 14]
    j = board.dataset("data/rows.jsonl", None)
    assert j["stats"]["x"]["type"] == "int" and j["stats"]["nested"]["type"] == "str"
    a = board.dataset("data/arr.json", None)
    assert a["stats"]["b"]["missing"] == 1


def test_node_link_members(board):
    info = board.dataset("data/g.json", "nodes")
    assert info["members"] == ["edges", "nodes"] and "group" in info["columns"]


def test_unsafe_formats_are_not_loaded(board):
    info = board.dataset("data/model.pkl", None)
    assert "unpickling" in info["error"] and "columns" not in info


@pytest.mark.parametrize("rel", ["secret.csv", "../secret.csv", "data/../secret.csv", "/etc/passwd", "data/nope.csv"])
def test_paths_outside_roots_are_refused(board, rel):
    assert board.dataset(rel, None) is None
    assert board.graph(rel, None, None, 1, 10) is None


def test_non_graph_table_reports_error(board):
    assert "error" in board.graph("data/rows.jsonl", None, None, 1, 10)


def test_optional_numpy_formats(tmp_path):
    np = pytest.importorskip("numpy")
    d = tmp_path / "data"
    d.mkdir()
    np.savez(d / "g.npz", edge_index=np.array([[0, 1, 2], [1, 2, 0]]), y=np.array([0, 1, 1]))
    np.save(d / "x.npy", np.arange(12).reshape(4, 3))
    b = Board(load(tmp_path))
    g = b.graph("data/g.npz", None, None, 2, 10)
    assert g["summary"]["nodes"] == 3 and "y" in g["summary"]["attributes"]
    assert b.dataset("data/x.npy", None)["columns"] == ["value[0]", "value[1]", "value[2]"]


def test_optional_parquet(tmp_path):
    pa = pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq
    d = tmp_path / "data"
    d.mkdir()
    pq.write_table(pa.table({"source": ["a", "b"], "target": ["b", "c"], "w": [1.0, 2.0]}), d / "e.parquet")
    b = Board(load(tmp_path))
    info = b.dataset("data/e.parquet", None)
    assert info["total_rows"] == 2 and info["graph"] is True
    assert b.graph("data/e.parquet", None, None, 2, 10)["summary"]["edges"] == 2
