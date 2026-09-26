"""HTTP API + single-page browser over a runs directory and markdown docs.

Reads run directories directly (no database, no import step): a run shows up as
soon as its folder exists.
"""

from __future__ import annotations

import json
import re
import subprocess
import time
from functools import partial
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from . import datasets as ds
from .config import Config

STATIC = Path(__file__).parent / "static"
SAFE_NAME = re.compile(r"^[\w.-]+$")
TEXT_SUFFIXES = {".md", ".csv", ".json", ".jsonl", ".diff", ".txt", ".log", ".yaml", ".yml", ".toml"}
RUNNING_WINDOW_S = 600  # no summary + files touched recently => still running


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


class Board:
    """All data access, independent of HTTP (also used by the CLI)."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._tables = ds.LRU(16)
        self._graphs = ds.LRU(4)

    # --- runs ----------------------------------------------------------------
    def _run_dir(self, run_id: str) -> Path | None:
        if not SAFE_NAME.match(run_id):
            return None
        d = self.cfg.runs / run_id
        return d if d.is_dir() else None

    @staticmethod
    def _status(d: Path, summary: dict) -> str:
        if summary:
            return summary.get("status", "ok")
        newest = max((p.stat().st_mtime for p in d.iterdir() if p.is_file()), default=0)
        return "running" if time.time() - newest < RUNNING_WINDOW_S else "incomplete"

    def runs(self) -> list[dict]:
        out = []
        if not self.cfg.runs.is_dir():
            return out
        for d in sorted(self.cfg.runs.iterdir(), reverse=True):
            if not d.is_dir():
                continue
            meta, summary = _read_json(d / "meta.json"), _read_json(d / "summary.json")
            if not meta:
                continue
            out.append({
                "run_id": d.name,
                "experiment": meta.get("experiment"),
                "tags": meta.get("tags", {}),
                "started_at": meta.get("started_at"),
                "git_commit": (meta.get("git_commit") or "")[:10],
                "git_dirty": meta.get("git_dirty"),
                "status": self._status(d, summary),
                "elapsed_s": summary.get("elapsed_s"),
                "host": meta.get("host"),
            })
        return out

    def _prereg_files(self) -> list[Path]:
        if self.cfg.prereg:
            return [p for p in self.cfg.prereg if p.is_file()]
        return sorted(self.cfg.docs.glob("*.md")) if self.cfg.docs else []

    def preregistration(self, run_id: str, tags: dict) -> dict | None:
        """The "## ..." section of a doc that cites this run id (or matches its tag)."""
        sections = []
        for f in self._prereg_files():
            for s in re.split(r"(?m)^(?=## )", f.read_text()):
                if s.startswith("## "):
                    sections.append((f, s))
        for f, s in sections:
            if run_id in s:
                return {"file": str(f.relative_to(self.cfg.project)), "text": s}
        tag = tags.get(self.cfg.prereg_tag)
        if tag:
            for f, s in sections:
                if re.match(rf"## {re.escape(str(tag))}(\s|$)", s):
                    return {"file": str(f.relative_to(self.cfg.project)), "text": s}
        return None

    def run(self, run_id: str) -> dict | None:
        d = self._run_dir(run_id)
        if d is None:
            return None
        meta, summary = _read_json(d / "meta.json"), _read_json(d / "summary.json")
        files = sorted(str(p.relative_to(d)) for p in d.rglob("*") if p.is_file())
        reports = {f: (d / f).read_text() for f in files if f.endswith(".md") and "/" not in f}
        return {
            "run_id": run_id,
            "config": _read_json(d / "config.json"),
            "meta": meta,
            "summary": summary,
            "status": self._status(d, summary),
            "files": files,
            "reports": reports,
            "preregistration": self.preregistration(run_id, meta.get("tags", {})),
        }

    def metrics(self, run_id: str) -> list[dict] | None:
        d = self._run_dir(run_id)
        if d is None:
            return None
        path = d / "metrics.jsonl"
        if not path.exists():
            return []
        rows = []
        for line in path.read_text().splitlines():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass  # partially written last line of a running job
        return rows

    def file(self, run_id: str, rel: str) -> str | None:
        d = self._run_dir(run_id)
        if d is None:
            return None
        p = (d / rel).resolve()
        if d.resolve() not in p.parents or not p.is_file() or p.suffix not in TEXT_SUFFIXES:
            return None
        return p.read_text(errors="replace")

    # --- docs ----------------------------------------------------------------
    def docs(self) -> list[dict]:
        if not self.cfg.docs:
            return []
        names = [p.stem for p in self.cfg.docs.glob("*.md")]
        order = [n for n in self.cfg.doc_order if n in names]
        out = []
        for n in order + sorted(n for n in names if n not in order):
            text = (self.cfg.docs / f"{n}.md").read_text()
            title = next((l.lstrip("# ").strip() for l in text.splitlines() if l.startswith("# ")), n)
            out.append({"name": n, "title": title})
        return out

    def doc(self, name: str) -> dict | None:
        if not self.cfg.docs or not SAFE_NAME.match(name):
            return None
        p = self.cfg.docs / f"{name}.md"
        return {"name": name, "path": str(p.relative_to(self.cfg.project)), "text": p.read_text()} if p.is_file() else None

    # --- datasets ------------------------------------------------------------
    def _data_roots(self) -> list[Path]:
        roots = [r for r in self.cfg.datasets if r.is_dir()]
        if self.cfg.runs.is_dir():
            roots += sorted(p for p in self.cfg.runs.glob("*/artifacts") if p.is_dir())
        return roots

    def _resolve_data(self, rel: str | None) -> Path | None:
        if not rel:
            return None
        p = (self.cfg.project / rel).resolve()
        allowed = [r.resolve() for r in [*self.cfg.datasets, self.cfg.runs]]
        if not p.is_file() or not any(r == p or r in p.parents for r in allowed):
            return None
        return p

    def datasets(self) -> list[dict]:
        return ds.list_datasets(self._data_roots(), self.cfg.project)

    def dataset(self, rel: str | None, member: str | None) -> dict | None:
        p = self._resolve_data(rel)
        if p is None:
            return None
        fmt, kind, _ = ds.detect(p)
        info = {"path": rel, "format": fmt, "kind": kind, "size": p.stat().st_size, "member": member}
        try:
            info["members"] = ds.members(p)
            info["graph"] = ds.graph_capable(p)
            if kind == "unsafe":
                raise ds.DatasetError(f"{fmt} files are not loaded: unpickling can execute arbitrary code")
            if kind == "graph":
                return info  # native graph formats: the Graph view is the table
            t = self._tables.get(ds.file_key(p, member), lambda: ds.summarize_table(ds.open_table(p, member)))
            info.update(columns=t.columns, stats=t.stats, rows_scanned=t.rows_scanned,
                        total_rows=t.total_rows, total_exact=t.total_exact, preview=ds.to_json_rows(t.preview))
        except ds.DatasetError as e:
            info["error"] = str(e)
        except (OSError, ValueError, UnicodeDecodeError) as e:
            info["error"] = f"cannot read {p.name}: {e}"
        return info

    def dataset_rows(self, rel: str | None, member: str | None, offset: int, limit: int) -> dict | None:
        p = self._resolve_data(rel)
        if p is None:
            return None
        try:
            src = ds.open_table(p, member)
            return {"columns": src.columns, "offset": offset,
                    "rows": ds.to_json_rows(ds.page(src, offset, max(1, min(limit, 500))))}
        except (ds.DatasetError, OSError, ValueError) as e:
            return {"error": str(e)}

    def dataset_columns(self, rel: str | None) -> dict | None:
        p = self._resolve_data(rel)
        if p is None:
            return None
        try:
            return {"columns": ds.open_table(p).columns}
        except (ds.DatasetError, OSError, ValueError) as e:
            return {"error": str(e)}

    def graph(self, rel: str | None, member: str | None, node: str | None, hops: int, max_nodes: int,
              nodes_rel: str | None = None, node_id: str | None = None) -> dict | None:
        p = self._resolve_data(rel)
        if p is None:
            return None
        try:
            base = self._graphs.get(ds.file_key(p, member), lambda: ds.load_graph(p))
            g, attrs_note = base, None
            np_ = self._resolve_data(nodes_rel)
            if np_ is not None and node_id:
                def attach():
                    g2 = ds.Graph(base.ids, base.src, base.dst, base.directed, dict(base.node_attrs),
                                  base.truncated, base.note, base._csr)
                    ds.attach_node_attrs(g2, ds.open_table(np_), node_id)
                    return g2
                g = self._graphs.get(ds.file_key(p, member, str(np_), node_id), attach)
                attrs_note = f"{nodes_rel} ({node_id})"
            summary = self._tables.get(ds.file_key(p, member, "graph-summary", str(nodes_rel), node_id),
                                       lambda: ds.graph_summary(g))
            sample = ds.graph_sample(g, node, max(1, min(hops, 4)), max(10, min(max_nodes, 1000)))
            return {"summary": summary, "sample": sample, "node_attrs_from": attrs_note}
        except ds.DatasetError as e:
            return {"error": str(e)}
        except (OSError, ValueError, StopIteration) as e:
            return {"error": f"cannot read graph: {e}"}

    # --- misc ----------------------------------------------------------------
    def client_config(self) -> dict:
        return {"title": self.cfg.title, "lang": self.cfg.lang, "sync_label": self.cfg.sync_label,
                "has_docs": self.cfg.docs is not None, "runs": str(self.cfg.runs),
                "runs_rel": self._rel(self.cfg.runs)}

    def _rel(self, p: Path) -> str | None:
        try:
            return str(p.resolve().relative_to(self.cfg.project.resolve()))
        except ValueError:
            return None

    def sync(self) -> dict:
        if not self.cfg.sync_command:
            return {"ok": False, "output": "sync is not configured"}
        try:
            r = subprocess.run(self.cfg.sync_command, shell=True, cwd=self.cfg.project,
                               capture_output=True, text=True, timeout=600)
            return {"ok": r.returncode == 0, "output": (r.stdout + r.stderr)[-4000:]}
        except (OSError, subprocess.TimeoutExpired) as e:
            return {"ok": False, "output": str(e)}


class Handler(BaseHTTPRequestHandler):
    board: Board

    def __init__(self, *args, board: Board, **kwargs):
        self.board = board
        super().__init__(*args, **kwargs)

    def log_message(self, fmt, *args):  # keep the console quiet
        pass

    def _send(self, body: bytes, ctype: str, status: HTTPStatus = HTTPStatus.OK) -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, status: HTTPStatus = HTTPStatus.OK) -> None:
        self._send(json.dumps(obj, ensure_ascii=False).encode(), "application/json; charset=utf-8", status)

    def _not_found(self) -> None:
        self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)

    def do_GET(self) -> None:
        url = urlparse(self.path)
        parts = [unquote(p) for p in url.path.strip("/").split("/") if p]
        q = {k: v[-1] for k, v in parse_qs(url.query).items()}

        def qint(name: str, default: int) -> int:
            try:
                return int(q.get(name, default))
            except ValueError:
                return default
        if not parts or parts == ["index.html"]:
            return self._send((STATIC / "index.html").read_bytes(), "text/html; charset=utf-8")
        if parts[0] != "api":
            return self._not_found()
        b, route = self.board, parts[1:]
        res = None
        if route == ["config"]:
            res = b.client_config()
        elif route == ["runs"]:
            res = b.runs()
        elif route == ["docs"]:
            res = b.docs()
        elif route == ["datasets"]:
            res = b.datasets()
        elif route == ["dataset"]:
            res = b.dataset(q.get("path"), q.get("member"))
        elif route == ["dataset", "rows"]:
            res = b.dataset_rows(q.get("path"), q.get("member"), qint("offset", 0), qint("limit", 50))
        elif route == ["dataset", "columns"]:
            res = b.dataset_columns(q.get("path"))
        elif route == ["graph"]:
            res = b.graph(q.get("path"), q.get("member"), q.get("node"), qint("hops", 2), qint("max_nodes", 200),
                          q.get("nodes"), q.get("node_id"))
        elif len(route) == 2 and route[0] == "doc":
            res = b.doc(route[1])
        elif len(route) == 2 and route[0] == "run":
            res = b.run(route[1])
        elif len(route) == 3 and route[0] == "run" and route[2] == "metrics":
            res = b.metrics(route[1])
        elif len(route) > 3 and route[0] == "run" and route[2] == "file":
            text = b.file(route[1], "/".join(route[3:]))
            res = None if text is None else {"text": text}
        return self._not_found() if res is None else self._json(res)

    def do_POST(self) -> None:
        if urlparse(self.path).path == "/api/sync":
            return self._json(self.board.sync())
        return self._not_found()


def make_server(cfg: Config, host: str = "127.0.0.1", port: int = 5050) -> ThreadingHTTPServer:
    return ThreadingHTTPServer((host, port), partial(Handler, board=Board(cfg)))
