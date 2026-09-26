"""Dataset discovery, readers and statistics for the browser: tables, arrays, graphs.

Formats that need only the standard library: CSV/TSV, JSON, JSONL (also .gz),
GraphML, GEXF, GML, node-link JSON, whitespace edge lists, MatrixMarket (.mtx).
Optional: Parquet / Feather / Arrow IPC need ``pyarrow``; NPY / NPZ need ``numpy``.
Pickle-based formats (.pkl, .pickle, .pt, .pth, .joblib) are never loaded:
unpickling executes arbitrary code.
"""

from __future__ import annotations

import csv
import gzip
import json
import math
import re
import statistics
import xml.etree.ElementTree as ET
from array import array
from collections import Counter, OrderedDict, deque
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

MAX_SCAN_ROWS = 200_000  # rows used for column statistics
MAX_COUNT_ROWS = 5_000_000  # beyond this the row count is reported as a lower bound
MAX_GRAPH_EDGES = 5_000_000
MAX_LIST_FILES = 5_000
UNIQUE_CAP = 10_000
HIST_BINS = 20

FORMATS: dict[str, tuple[str, str]] = {
    ".csv": ("csv", "table"), ".tsv": ("tsv", "table"), ".tab": ("tsv", "table"),
    ".jsonl": ("jsonl", "table"), ".ndjson": ("jsonl", "table"), ".json": ("json", "auto"),
    ".parquet": ("parquet", "table"), ".pq": ("parquet", "table"),
    ".feather": ("arrow", "table"), ".arrow": ("arrow", "table"), ".ipc": ("arrow", "table"),
    ".npy": ("npy", "array"), ".npz": ("npz", "auto"),
    ".graphml": ("graphml", "graph"), ".gexf": ("gexf", "graph"), ".gml": ("gml", "graph"),
    ".edgelist": ("edgelist", "graph"), ".edges": ("edgelist", "graph"), ".el": ("edgelist", "graph"),
    ".mtx": ("mtx", "graph"),
    ".pkl": ("pickle", "unsafe"), ".pickle": ("pickle", "unsafe"), ".pt": ("pickle", "unsafe"),
    ".pth": ("pickle", "unsafe"), ".joblib": ("pickle", "unsafe"),
}
EDGE_COLUMNS = [("src", "dst"), ("source", "target"), ("from", "to"), ("u", "v"), ("head", "tail"),
                ("src_id", "dst_id"), ("source_id", "target_id"), ("node1", "node2"), ("sender", "receiver")]
SKIP_DIRS = {".git", "__pycache__", ".venv", "node_modules", ".pytest_cache"}


class DatasetError(Exception):
    """A dataset cannot be read (missing optional dependency, bad format, unsafe)."""


# ---------------------------------------------------------------- discovery ---

def detect(path: Path) -> tuple[str, str, bool]:
    """(format, kind, gzipped). kind: table | array | graph | auto | unsafe | unknown."""
    suffixes = [s.lower() for s in path.suffixes]
    gz = bool(suffixes) and suffixes[-1] == ".gz"
    ext = suffixes[-2] if gz and len(suffixes) > 1 else (suffixes[-1] if suffixes else "")
    fmt, kind = FORMATS.get(ext, ("unknown", "unknown"))
    return fmt, kind, gz


def list_datasets(roots: list[Path], project: Path) -> list[dict]:
    out, seen = [], set()
    for root in roots:
        if not root.is_dir():
            continue
        for p in sorted(root.rglob("*")):
            if len(out) >= MAX_LIST_FILES:
                return out
            if not p.is_file() or any(part in SKIP_DIRS or part.startswith(".") for part in p.relative_to(root).parts):
                continue
            fmt, kind, _ = detect(p)
            if fmt == "unknown" or p in seen:
                continue
            seen.add(p)
            out.append({"path": str(p.relative_to(project)), "size": p.stat().st_size, "format": fmt, "kind": kind})
    return out


# ------------------------------------------------------------------ readers ---

def _open_text(path: Path, gz: bool):
    return gzip.open(path, "rt", newline="", errors="replace") if gz else open(path, newline="", errors="replace")


def _scalar(v: Any) -> Any:
    if isinstance(v, (dict, list)):
        return json.dumps(v, ensure_ascii=False)
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    if v is not None and not isinstance(v, (str, int, float, bool)):
        return str(v)
    return v


def _need(module: str, fmt: str):
    try:
        return __import__(module)
    except ImportError:
        raise DatasetError(f"{fmt} needs the optional package '{module}' (pip install {module})") from None


@dataclass
class TableSource:
    columns: list[str]
    rows: Iterator[list[Any]]
    total: int | None = None  # exact row count when the format stores it


def _csv_source(path: Path, gz: bool, delimiter: str | None) -> TableSource:
    f = _open_text(path, gz)
    sample = f.read(65536)
    f.seek(0)
    if delimiter is None:
        try:
            delimiter = csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
        except csv.Error:
            delimiter = ","
    reader = csv.reader(f, delimiter=delimiter)
    header = next(reader, [])
    try:
        has_header = csv.Sniffer().has_header(sample)
    except csv.Error:
        has_header = True
    if not has_header and header:
        columns = [f"col{i}" for i in range(len(header))]
        first = [header]
    else:
        columns, first = header, []

    def rows():
        with f:
            for r in first:
                yield [_parse(v) for v in r]
            for r in reader:
                yield [_parse(v) for v in r]
    return TableSource(columns, rows())


_INT_RE = re.compile(r"^[+-]?\d+$")


def _parse(v: str) -> Any:
    """CSV cell -> int / float / str / None (empty)."""
    if v == "":
        return None
    if _INT_RE.match(v):
        try:
            return int(v)
        except ValueError:
            return v
    try:
        x = float(v)
        return None if math.isnan(x) else x
    except ValueError:
        return v


def _records_source(records: Iterator[dict], peek: int = 1000) -> TableSource:
    buf, cols = [], {}
    for rec in records:
        if not isinstance(rec, dict):
            rec = {"value": rec}
        buf.append(rec)
        for k in rec:
            cols.setdefault(k, None)
        if len(buf) >= peek:
            break
    columns = list(cols)

    def rows():
        for rec in buf:
            yield [_scalar(rec.get(c)) for c in columns]
        for rec in records:
            if not isinstance(rec, dict):
                rec = {"value": rec}
            yield [_scalar(rec.get(c)) for c in columns]
    return TableSource(columns, rows())


def _jsonl_records(path: Path, gz: bool) -> Iterator[dict]:
    with _open_text(path, gz) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def _load_json(path: Path, gz: bool) -> Any:
    with _open_text(path, gz) as f:
        return json.load(f)


def _is_node_link(obj: Any) -> bool:
    return isinstance(obj, dict) and isinstance(obj.get("nodes"), list) and (
        isinstance(obj.get("links"), list) or isinstance(obj.get("edges"), list))


def _json_table(obj: Any) -> TableSource:
    if isinstance(obj, list):
        return _records_source(iter(obj))
    if isinstance(obj, dict) and obj and all(isinstance(v, list) for v in obj.values()):
        n = max(len(v) for v in obj.values())
        cols = list(obj)
        return TableSource(cols, ([_scalar(obj[c][i] if i < len(obj[c]) else None) for c in cols] for i in range(n)), n)
    if isinstance(obj, dict):  # a single object: key/value table
        return TableSource(["key", "value"], ([k, _scalar(v)] for k, v in obj.items()), len(obj))
    return TableSource(["value"], iter([[_scalar(obj)]]), 1)


def _arrow_source(path: Path, fmt: str) -> TableSource:
    pa = _need("pyarrow", fmt)
    if fmt == "parquet":
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(path)
        columns = pf.schema_arrow.names

        def rows():
            for batch in pf.iter_batches(batch_size=16384):
                for rec in batch.to_pylist():
                    yield [_scalar(rec[c]) for c in columns]
        return TableSource(columns, rows(), pf.metadata.num_rows)
    import pyarrow.feather as feather
    table = feather.read_table(path, memory_map=True)
    columns = table.column_names

    def rows():
        for batch in table.to_batches(max_chunksize=16384):
            for rec in batch.to_pylist():
                yield [_scalar(rec[c]) for c in columns]
    return TableSource(columns, rows(), table.num_rows)


def _array_source(arr: Any, name: str = "value") -> TableSource:
    if arr.ndim == 0:
        return TableSource([name], iter([[_scalar(arr.item())]]), 1)
    if arr.ndim == 1:
        return TableSource([name], ([_scalar(v)] for v in arr.tolist()), int(arr.shape[0]))
    flat = arr.reshape(arr.shape[0], -1)
    cols = [f"{name}[{i}]" for i in range(flat.shape[1])]
    return TableSource(cols, ([_scalar(v) for v in row] for row in flat.tolist()), int(arr.shape[0]))


def _numpy_load(path: Path, fmt: str):
    np = _need("numpy", fmt)
    try:
        return np.load(path, mmap_mode="r" if fmt == "npy" else None, allow_pickle=False)
    except ValueError as e:  # object arrays need pickle
        raise DatasetError(f"{path.name}: {e} (object arrays are not loaded: they need pickle)") from None


def open_table(path: Path, member: str | None = None) -> TableSource:
    fmt, kind, gz = detect(path)
    if kind == "unsafe":
        raise DatasetError(f"{fmt} files are not loaded: unpickling can execute arbitrary code")
    if fmt in ("csv", "tsv"):
        return _csv_source(path, gz, "\t" if fmt == "tsv" else None)
    if fmt == "jsonl":
        return _records_source(_jsonl_records(path, gz))
    if fmt == "json":
        obj = _load_json(path, gz)
        if _is_node_link(obj):
            key = "links" if "links" in obj else "edges"
            part = obj["nodes"] if member == "nodes" else obj[key]
            return _records_source(iter(part))
        return _json_table(obj)
    if fmt in ("parquet", "arrow"):
        return _arrow_source(path, fmt)
    if fmt == "npy":
        return _array_source(_numpy_load(path, fmt))
    if fmt == "npz":
        z = _numpy_load(path, fmt)
        keys = list(z.keys())
        if not keys:
            raise DatasetError("empty npz")
        key = member if member in keys else keys[0]
        return _array_source(z[key], key)
    if fmt in ("graphml", "gexf", "gml", "edgelist", "mtx"):
        g = load_graph(path)
        cols = ["source", "target"]
        return TableSource(cols, ([g.ids[s], g.ids[d]] for s, d in zip(g.src, g.dst)), len(g.src))
    raise DatasetError(f"unsupported format: {path.suffix}")


def members(path: Path) -> list[str]:
    """Sub-datasets of a container file (npz arrays, node-link nodes/edges)."""
    fmt, _, gz = detect(path)
    if fmt == "npz":
        z = _numpy_load(path, fmt)
        return list(z.keys())
    if fmt == "json":
        try:
            if _is_node_link(_load_json(path, gz)):
                return ["edges", "nodes"]
        except (OSError, json.JSONDecodeError):
            pass
    return []


# ---------------------------------------------------------------- statistics ---

def _col_stats(values: list[Any], total_scanned: int) -> dict:
    non_null = [v for v in values if v is not None]
    missing = total_scanned - len(non_null)
    nums = [v for v in non_null if isinstance(v, (int, float)) and not isinstance(v, bool)]
    uniq: set = set()
    capped = False
    for v in non_null:
        if len(uniq) >= UNIQUE_CAP:
            capped = True
            break
        uniq.add(v if not isinstance(v, float) else round(v, 12))
    st: dict[str, Any] = {"count": len(non_null), "missing": missing,
                          "unique": len(uniq), "unique_capped": capped}
    if non_null and len(nums) >= 0.95 * len(non_null):
        st["type"] = "int" if all(isinstance(v, int) for v in nums) else "float"
        lo, hi = min(nums), max(nums)
        mean = statistics.fmean(nums)
        st.update(min=lo, max=hi, mean=mean,
                  std=statistics.pstdev(nums) if len(nums) > 1 else 0.0,
                  median=statistics.median(nums))
        if st["type"] == "int" and len(uniq) <= 12 and not capped:
            st["top"] = Counter(nums).most_common(12)
        else:
            width = (hi - lo) / HIST_BINS if hi > lo else 1.0
            counts = [0] * HIST_BINS
            for v in nums:
                counts[min(int((v - lo) / width), HIST_BINS - 1) if hi > lo else 0] += 1
            st["hist"] = {"lo": lo, "hi": hi, "counts": counts}
    else:
        st["type"] = "bool" if non_null and all(isinstance(v, bool) for v in non_null) else "str"
        st["top"] = [[str(k), c] for k, c in Counter(str(v) for v in non_null).most_common(8)]
    return st


@dataclass
class TableSummary:
    columns: list[str]
    stats: dict[str, dict]
    rows_scanned: int
    total_rows: int | None
    total_exact: bool
    preview: list[list[Any]] = field(default_factory=list)


def summarize_table(src: TableSource, preview: int = 50) -> TableSummary:
    cols = src.columns
    buckets: list[list[Any]] = [[] for _ in cols]
    head, n = [], 0
    for row in src.rows:
        if n < MAX_SCAN_ROWS:
            for i in range(len(cols)):
                buckets[i].append(row[i] if i < len(row) else None)
        if n < preview:
            head.append(row[: len(cols)])
        n += 1
        if n >= MAX_SCAN_ROWS and src.total is not None:
            break
        if n >= MAX_COUNT_ROWS:
            break
    scanned = min(n, MAX_SCAN_ROWS)
    total, exact = (src.total, True) if src.total is not None else (n, n < MAX_COUNT_ROWS)
    stats = {c: _col_stats(buckets[i], scanned) for i, c in enumerate(cols)}
    return TableSummary(cols, stats, scanned, total, exact, head)


def page(src: TableSource, offset: int, limit: int) -> list[list[Any]]:
    out = []
    for i, row in enumerate(src.rows):
        if i >= offset + limit:
            break
        if i >= offset:
            out.append(row[: len(src.columns)])
    return out


def edge_columns(columns: list[str]) -> tuple[str, str] | None:
    low = {c.lower(): c for c in columns}
    for a, b in EDGE_COLUMNS:
        if a in low and b in low:
            return low[a], low[b]
    return None


# -------------------------------------------------------------------- graphs ---

@dataclass
class Graph:
    ids: list[str]
    src: array
    dst: array
    directed: bool
    node_attrs: dict[str, list[Any]] = field(default_factory=dict)
    truncated: bool = False
    note: str = ""
    _csr: dict = field(default_factory=dict, repr=False)

    @property
    def n(self) -> int:
        return len(self.ids)

    def _build(self, key: str) -> tuple[array, array]:
        if key not in self._csr:
            a, b = (self.src, self.dst) if key == "out" else (self.dst, self.src)
            deg = array("l", [0]) * (self.n + 1)
            for x in a:
                deg[x + 1] += 1
            for i in range(self.n):
                deg[i + 1] += deg[i]
            nbr = array("l", [0]) * len(a)
            pos = array("l", deg)
            for x, y in zip(a, b):
                nbr[pos[x]] = y
                pos[x] += 1
            self._csr[key] = (deg, nbr)
        return self._csr[key]

    def neighbors(self, i: int, key: str) -> array:
        ptr, nbr = self._build(key)
        return nbr[ptr[i]: ptr[i + 1]]

    def degrees(self) -> tuple[list[int], list[int]]:
        out_d, in_d = [0] * self.n, [0] * self.n
        for s in self.src:
            out_d[s] += 1
        for d in self.dst:
            in_d[d] += 1
        return out_d, in_d


class _Builder:
    def __init__(self, directed: bool):
        self.index: dict[str, int] = {}
        self.ids: list[str] = []
        self.src, self.dst = array("l"), array("l")
        self.directed = directed
        self.attrs: dict[str, dict[int, Any]] = {}
        self.truncated = False

    def node(self, nid: Any, attrs: dict | None = None) -> int:
        k = str(nid)
        i = self.index.get(k)
        if i is None:
            i = self.index[k] = len(self.ids)
            self.ids.append(k)
        for a, v in (attrs or {}).items():
            self.attrs.setdefault(a, {})[i] = _scalar(v)
        return i

    def edge(self, u: Any, v: Any) -> bool:
        if len(self.src) >= MAX_GRAPH_EDGES:
            self.truncated = True
            return False
        self.src.append(self.node(u))
        self.dst.append(self.node(v))
        return True

    def graph(self, note: str) -> Graph:
        attrs = {a: [vals.get(i) for i in range(len(self.ids))] for a, vals in self.attrs.items()}
        return Graph(self.ids, self.src, self.dst, self.directed, attrs, self.truncated, note)


def _graph_from_table(src: TableSource, cols: tuple[str, str], directed: bool = True) -> Graph:
    b = _Builder(directed)
    i, j = src.columns.index(cols[0]), src.columns.index(cols[1])
    for row in src.rows:
        if row[i] is None or row[j] is None:
            continue
        if not b.edge(row[i], row[j]):
            break
    return b.graph(f"edge list: {cols[0]} → {cols[1]}")


def _graph_edgelist(path: Path, gz: bool) -> Graph:
    b = _Builder(True)
    with _open_text(path, gz) as f:
        for line in f:
            t = line.split()
            if len(t) >= 2 and not t[0].startswith(("#", "%")):
                if not b.edge(t[0], t[1]):
                    break
    return b.graph("whitespace edge list (first two columns)")


def _graph_mtx(path: Path, gz: bool) -> Graph:
    with _open_text(path, gz) as f:
        header = f.readline().lower()
        symmetric = "symmetric" in header or "hermitian" in header
        b = _Builder(not symmetric)
        dims = None
        for line in f:
            if line.startswith("%") or not line.strip():
                continue
            t = line.split()
            if dims is None:
                dims = [int(x) for x in t[:3]]
                for k in range(1, max(dims[0], dims[1]) + 1):
                    b.node(k)
                continue
            if not b.edge(int(t[0]), int(t[1])):
                break
    return b.graph("MatrixMarket coordinate" + (" (symmetric)" if symmetric else ""))


def _strip_ns(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _graph_graphml(path: Path, gz: bool) -> Graph:
    with (gzip.open(path) if gz else open(path, "rb")) as f:
        root = ET.parse(f).getroot()
    keys = {k.get("id"): k.get("attr.name") or k.get("id") for k in root.iter() if _strip_ns(k.tag) == "key"}
    g = next(e for e in root.iter() if _strip_ns(e.tag) == "graph")
    b = _Builder(g.get("edgedefault", "directed") == "directed")
    for e in g:
        tag = _strip_ns(e.tag)
        if tag == "node":
            b.node(e.get("id"), {keys.get(d.get("key"), d.get("key")): d.text for d in e if _strip_ns(d.tag) == "data"})
        elif tag == "edge":
            if not b.edge(e.get("source"), e.get("target")):
                break
    return b.graph("GraphML")


def _graph_gexf(path: Path, gz: bool) -> Graph:
    with (gzip.open(path) if gz else open(path, "rb")) as f:
        root = ET.parse(f).getroot()
    g = next(e for e in root.iter() if _strip_ns(e.tag) == "graph")
    b = _Builder(g.get("defaultedgetype", "undirected") == "directed")
    titles = {}
    for a in g.iter():
        if _strip_ns(a.tag) == "attribute":
            titles[a.get("id")] = a.get("title") or a.get("id")
    for e in g.iter():
        tag = _strip_ns(e.tag)
        if tag == "node":
            attrs = {"label": e.get("label")} if e.get("label") is not None else {}
            for av in e.iter():
                if _strip_ns(av.tag) == "attvalue":
                    attrs[titles.get(av.get("for"), av.get("for"))] = av.get("value")
            b.node(e.get("id"), attrs)
        elif tag == "edge":
            if not b.edge(e.get("source"), e.get("target")):
                break
    return b.graph("GEXF")


_GML_TOKEN = re.compile(r'"[^"]*"|\[|\]|[^\s\[\]"]+')


def _gml_parse(tokens: list[str], i: int = 0) -> tuple[list[tuple[str, Any]], int]:
    out = []
    while i < len(tokens):
        t = tokens[i]
        if t == "]":
            return out, i + 1
        key, val = t, tokens[i + 1]
        if val == "[":
            sub, i = _gml_parse(tokens, i + 2)
            out.append((key, sub))
        else:
            out.append((key, val.strip('"')))
            i += 2
    return out, i


def _graph_gml(path: Path, gz: bool) -> Graph:
    with _open_text(path, gz) as f:
        text = "\n".join(l for l in f if not l.lstrip().startswith("#"))
    items, _ = _gml_parse(_GML_TOKEN.findall(text))
    graph = next((v for k, v in items if k == "graph"), items)
    directed = any(k == "directed" and v == "1" for k, v in graph)
    b = _Builder(directed)
    for k, v in graph:
        if k == "node":
            d = dict(x for x in v if not isinstance(x[1], list))
            b.node(d.pop("id", None), d)
    for k, v in graph:
        if k == "edge":
            d = dict(x for x in v if not isinstance(x[1], list))
            if not b.edge(d.get("source"), d.get("target")):
                break
    return b.graph("GML")


def _graph_node_link(obj: dict) -> Graph:
    b = _Builder(bool(obj.get("directed", False)))
    for nd in obj["nodes"]:
        nd = dict(nd)
        b.node(nd.pop("id", None), nd)
    for e in obj.get("links", obj.get("edges", [])):
        if not b.edge(e.get("source"), e.get("target")):
            break
    return b.graph("node-link JSON")


def _graph_npz(path: Path) -> Graph:
    z = _numpy_load(path, "npz")
    keys = set(z.keys())
    if "edge_index" in keys:
        ei = z["edge_index"]
        s, d = ei[0], ei[1]
    elif {"row", "col"} <= keys:
        s, d = z["row"], z["col"]
    elif {"src", "dst"} <= keys:
        s, d = z["src"], z["dst"]
    else:
        raise DatasetError("npz has no edge_index / row+col / src+dst arrays")
    b = _Builder(True)
    n = int(max(s.max(initial=-1), d.max(initial=-1))) + 1
    if "num_nodes" in keys:
        n = max(n, int(z["num_nodes"]))
    for k in range(n):
        b.node(k)
    for u, v in zip(s.tolist(), d.tolist()):
        if not b.edge(u, v):
            break
    for key in keys:  # per-node label arrays, e.g. y
        arr = z[key]
        if key not in ("edge_index", "row", "col", "src", "dst") and arr.ndim == 1 and arr.shape[0] == n:
            for k, v in enumerate(arr.tolist()):
                b.attrs.setdefault(key, {})[k] = v
    return b.graph("NPZ edge arrays")


def graph_capable(path: Path) -> bool:
    fmt, kind, gz = detect(path)
    if kind == "graph":
        return True
    try:
        if fmt == "json":
            return _is_node_link(_load_json(path, gz))
        if fmt == "npz":
            keys = set(_numpy_load(path, fmt).keys())
            return "edge_index" in keys or {"row", "col"} <= keys or {"src", "dst"} <= keys
        if kind == "table":
            return edge_columns(open_table(path).columns) is not None
    except (DatasetError, OSError, ValueError, json.JSONDecodeError):
        return False
    return False


def load_graph(path: Path) -> Graph:
    fmt, kind, gz = detect(path)
    if fmt == "graphml":
        return _graph_graphml(path, gz)
    if fmt == "gexf":
        return _graph_gexf(path, gz)
    if fmt == "gml":
        return _graph_gml(path, gz)
    if fmt == "edgelist":
        return _graph_edgelist(path, gz)
    if fmt == "mtx":
        return _graph_mtx(path, gz)
    if fmt == "npz":
        return _graph_npz(path)
    if fmt == "json":
        obj = _load_json(path, gz)
        if _is_node_link(obj):
            return _graph_node_link(obj)
    if kind in ("table", "auto"):
        src = open_table(path)
        cols = edge_columns(src.columns)
        if cols:
            return _graph_from_table(src, cols)
    raise DatasetError("not a graph: no recognised edge columns (e.g. src/dst, source/target, from/to)")


def attach_node_attrs(g: Graph, nodes: TableSource, id_col: str, max_attrs: int = 16) -> list[str]:
    """Join a node table (e.g. users.parquet with user_id) onto the graph's nodes."""
    if id_col not in nodes.columns:
        raise DatasetError(f"node table has no column '{id_col}'")
    idx = {k: i for i, k in enumerate(g.ids)}
    k = nodes.columns.index(id_col)
    names = [c for c in nodes.columns if c != id_col][:max_attrs]
    pos = [nodes.columns.index(c) for c in names]
    cols = {c: [None] * g.n for c in names}
    for row in nodes.rows:
        i = idx.get(str(row[k]))
        if i is not None:
            for c, p in zip(names, pos):
                cols[c][i] = row[p] if p < len(row) else None
    g.node_attrs.update(cols)
    return names


def graph_summary(g: Graph) -> dict:
    n, m = g.n, len(g.src)
    out_d, in_d = g.degrees()
    deg = [a + b for a, b in zip(out_d, in_d)]
    # Connected components on the undirected view (union-find).
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    self_loops = 0
    for s, d in zip(g.src, g.dst):
        if s == d:
            self_loops += 1
            continue
        rs, rd = find(s), find(d)
        if rs != rd:
            parent[rs] = rd
    sizes = Counter(find(i) for i in range(n))
    res: dict[str, Any] = {
        "nodes": n, "edges": m, "directed": g.directed, "self_loops": self_loops,
        "isolated": sum(1 for x in deg if x == 0), "components": len(sizes),
        "largest_component": max(sizes.values(), default=0),
        "density": (m / (n * (n - 1)) if g.directed else 2 * m / (n * (n - 1))) if n > 1 else 0.0,
        "degree": {"mean": statistics.fmean(deg) if deg else 0, "median": statistics.median(deg) if deg else 0,
                   "max": max(deg, default=0)},
        "truncated": g.truncated, "note": g.note, "attributes": list(g.node_attrs),
    }
    if m <= 2_000_000:
        pairs = set(zip(g.src, g.dst))
        res["duplicate_edges"] = m - len(pairs)
        if g.directed:
            nl = [(s, d) for s, d in pairs if s != d]
            res["reciprocity"] = (sum(1 for s, d in nl if (d, s) in pairs) / len(nl)) if nl else 0.0
    # log2-binned degree histogram: 0, 1, 2-3, 4-7, ...
    bins = Counter(0 if x == 0 else x.bit_length() for x in deg)
    res["degree_hist"] = [{"lo": 0 if b == 0 else 1 << (b - 1), "hi": 0 if b == 0 else (1 << b) - 1, "count": bins[b]}
                          for b in range(max(bins, default=0) + 1)]
    if g.directed:
        res["in_degree_max"] = max(in_d, default=0)
        res["out_degree_max"] = max(out_d, default=0)
    return res


def graph_sample(g: Graph, node: str | None = None, hops: int = 2, max_nodes: int = 200,
                 max_edges: int = 3000) -> dict:
    """Ego network around ``node`` (default: highest-degree node) for drawing."""
    out_d, in_d = g.degrees()
    deg = [a + b for a, b in zip(out_d, in_d)]
    if g.n == 0:
        return {"center": None, "nodes": [], "edges": [], "truncated": False}
    lookup = {k: i for i, k in enumerate(g.ids)} if node is not None else {}
    if node in lookup:
        center = lookup[node]
    else:
        center = max(range(g.n), key=deg.__getitem__)
    order, seen = [center], {center: 0}
    q = deque([center])
    truncated = False
    while q:
        u = q.popleft()
        if seen[u] >= hops:
            continue
        nb = sorted(set(g.neighbors(u, "out")) | set(g.neighbors(u, "in")), key=lambda x: -deg[x])
        for v in nb:
            if v in seen:
                continue
            if len(order) >= max_nodes:
                truncated = True
                break
            seen[v] = seen[u] + 1
            order.append(v)
            q.append(v)
        if len(order) >= max_nodes and q:
            truncated = True
            break
    pos = {v: i for i, v in enumerate(order)}
    edges = []
    for u in order:
        for v in g.neighbors(u, "out"):
            if v in pos:
                edges.append([pos[u], pos[v]])
                if len(edges) >= max_edges:
                    truncated = True
                    break
        if len(edges) >= max_edges:
            break
    nodes = [{"id": g.ids[v], "deg": deg[v], "in": in_d[v], "out": out_d[v], "hop": seen[v],
              "attrs": {a: vals[v] for a, vals in g.node_attrs.items()}} for v in order]
    return {"center": g.ids[center], "nodes": nodes, "edges": edges, "truncated": truncated,
            "directed": g.directed}


# --------------------------------------------------------------------- cache ---

class LRU:
    def __init__(self, size: int):
        self.size, self.data = size, OrderedDict()

    def get(self, key, make):
        if key in self.data:
            self.data.move_to_end(key)
            return self.data[key]
        val = make()
        self.data[key] = val
        if len(self.data) > self.size:
            self.data.popitem(last=False)
        return val


def file_key(path: Path, *extra) -> tuple:
    st = path.stat()
    return (str(path), st.st_mtime_ns, st.st_size, *extra)


def to_json_rows(rows: list[list[Any]]) -> list[list[Any]]:
    return [[_scalar(v) for v in r] for r in rows]

