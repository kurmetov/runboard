"""File-based run tracking.

Every run gets its own directory, which is the whole storage format
(see README, "Run directory format"):

    <runs>/<run_id>/
        config.json            full resolved config
        meta.json              experiment, tags, git commit / dirty flag, argv, versions
        git.diff               uncommitted changes to tracked files (only if dirty)
        code_snapshot.tar.gz   project code at launch time (only if dirty)
        metrics.jsonl          streamed metrics, one JSON object per line
        summary.json           final metrics + status; its absence means running / interrupted
        artifacts/             large outputs (usually git-ignored)
    <runs>/registry.jsonl      one line per finished run
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
import tarfile
import time
from collections.abc import Iterable
from datetime import datetime
from importlib import metadata
from pathlib import Path
from typing import Any

FORMAT_VERSION = 1
DEFAULT_PACKAGES = ("numpy", "pandas", "scipy", "scikit-learn", "torch", "jax", "tensorflow")
SNAPSHOT_MAX_FILE = 1 << 20  # 1 MiB: bigger files are data, not code
SNAPSHOT_MAX_TOTAL = 50 << 20


def _git(repo: Path, *args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args], cwd=repo, capture_output=True, text=True, check=True
        ).stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""


def find_repo(start: Path | None = None) -> Path:
    """Git top-level of ``start`` (default: cwd), or ``start`` itself outside git."""
    start = Path(start or Path.cwd()).resolve()
    top = _git(start, "rev-parse", "--show-toplevel").strip()
    return Path(top) if top else start


def _package_versions(names: Iterable[str]) -> dict[str, str]:
    versions = {}
    for name in names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            pass
    return versions


def _to_jsonable(obj: Any) -> Any:
    if hasattr(obj, "item"):  # numpy scalars
        return obj.item()
    if hasattr(obj, "tolist"):  # numpy arrays
        return obj.tolist()
    if isinstance(obj, (Path, datetime)):
        return str(obj)
    if isinstance(obj, (set, tuple)):
        return list(obj)
    raise TypeError(f"not JSON serializable: {type(obj)}")


def _dump(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=_to_jsonable))


class Run:
    """A single tracked run. Use as a context manager.

    Args:
        experiment: experiment name; runs are grouped by it in the browser.
        config: the full resolved config (any JSON-serializable dict).
        tags: free-form labels, e.g. {"stage": "S1", "step": "S1.1"}.
        root: runs directory (default: ``<repo>/runs``).
        repo: project root used for git info and code snapshots (default: git top-level of cwd).
        dims: names of the config/metric keys that form the experiment grid; the
            browser uses them as pivot dimensions. Defaults to the keys of
            ``config["grid"]`` when present.
        snapshot: what to archive when the tree is dirty. ``None`` = all files git
            would track (tracked + untracked, not ignored), excluding ``root``;
            an iterable of paths relative to ``repo``; ``False`` = never.
        packages: package names whose versions are recorded.
    """

    def __init__(
        self,
        experiment: str,
        config: dict[str, Any],
        tags: dict[str, Any] | None = None,
        root: str | Path | None = None,
        repo: str | Path | None = None,
        dims: Iterable[str] | None = None,
        snapshot: Iterable[str] | bool | None = None,
        packages: Iterable[str] = DEFAULT_PACKAGES,
    ):
        self.experiment = experiment
        self.config = config
        self.tags = tags or {}
        self.repo = Path(repo).resolve() if repo else find_repo()
        root = Path(root) if root else self.repo / "runs"
        root.mkdir(parents=True, exist_ok=True)
        base_id = f"{datetime.now():%Y%m%d-%H%M%S}_{experiment}"
        # Several runs of one experiment can start within the same second (grids, retries).
        for n in range(1, 1000):
            self.run_id = base_id if n == 1 else f"{base_id}-{n}"
            self.dir = root / self.run_id
            try:
                self.dir.mkdir()
                break
            except FileExistsError:
                continue
        self.artifacts = self.dir / "artifacts"
        self.artifacts.mkdir()
        self._root = root
        self._registry = root / "registry.jsonl"
        self._t0 = time.time()
        self._finished = False

        commit = _git(self.repo, "rev-parse", "HEAD").strip()
        diff = _git(self.repo, "diff", "HEAD")
        untracked = [p for p in _git(self.repo, "ls-files", "--others", "--exclude-standard").split()
                     if not self._in_root(p)]
        if dims is None and isinstance(config.get("grid"), dict):
            dims = [("seed" if k == "seeds" else k) for k in config["grid"]]
        self.meta = {
            "format": FORMAT_VERSION,
            "run_id": self.run_id,
            "experiment": experiment,
            "tags": self.tags,
            "dims": list(dims) if dims is not None else None,
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "git_commit": commit,
            "git_dirty": bool(diff.strip() or untracked),
            "git_untracked": untracked,
            "argv": sys.argv,
            "python": sys.version,
            "platform": platform.platform(),
            "host": platform.node(),
            "packages": _package_versions(packages),
        }
        _dump(self.dir / "config.json", config)
        _dump(self.dir / "meta.json", self.meta)
        if diff.strip():
            (self.dir / "git.diff").write_text(diff)
        if self.meta["git_dirty"] and snapshot is not False:
            self._snapshot(snapshot)

    def _in_root(self, rel: str) -> bool:
        try:
            (self.repo / rel).resolve().relative_to(self._root.resolve())
            return True
        except ValueError:
            return False

    def _snapshot(self, spec: Iterable[str] | None) -> None:
        # git.diff misses untracked files, so keep the exact code that ran.
        if spec is None:
            files = _git(self.repo, "ls-files", "--cached", "--others", "--exclude-standard").split()
            if not files:  # not a git repo
                return
        else:
            files = list(spec)
        total = 0
        with tarfile.open(self.dir / "code_snapshot.tar.gz", "w:gz") as tar:
            for rel in files:
                p = self.repo / rel
                if not p.exists() or self._in_root(rel) or "__pycache__" in p.parts:
                    continue
                if p.is_file():
                    size = p.stat().st_size
                    if size > SNAPSHOT_MAX_FILE or total + size > SNAPSHOT_MAX_TOTAL:
                        continue
                    total += size
                tar.add(p, arcname=rel, filter=lambda ti: None if "__pycache__" in ti.name else ti)

    def log(self, metrics: dict[str, Any], **context: Any) -> None:
        """Append one metrics row; ``context`` holds grid coordinates, step, etc."""
        record = {**context, **metrics, "_t": round(time.time() - self._t0, 3)}
        with open(self.dir / "metrics.jsonl", "a") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=_to_jsonable) + "\n")

    def finish(self, summary: dict[str, Any], status: str = "ok") -> None:
        self._finished = True
        elapsed = round(time.time() - self._t0, 1)
        _dump(self.dir / "summary.json", {"status": status, "elapsed_s": elapsed, **summary})
        record = {
            "run_id": self.run_id,
            "experiment": self.experiment,
            "tags": self.tags,
            "status": status,
            "git_commit": self.meta["git_commit"][:10],
            "git_dirty": self.meta["git_dirty"],
            "elapsed_s": elapsed,
        }
        with open(self._registry, "a") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=_to_jsonable) + "\n")

    def __enter__(self) -> "Run":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is not None:
            self.finish({"error": f"{exc_type.__name__}: {exc}"}, status="failed")
        elif not self._finished:
            self.finish({})
