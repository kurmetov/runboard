"""Project configuration: ``runboard.toml`` in the project root (all keys optional).

    title = "my project"            # browser title
    lang = "en"                     # "en" | "ru"
    runs = "runs"                   # runs directory, relative to the project root
    docs = "docs"                   # markdown docs shown in the Docs tab ("" to disable)
    doc_order = ["plan", "log"]     # doc names (without .md) listed first
    prereg = ["docs/experiments.md"]  # where to look for a run's pre-registration
    prereg_tag = "step"             # tag matched against "## <value> ..." headings
    datasets = ["data"]             # folders browsable in the Data tab (run artifacts are always included)

    [sync]                          # optional button that refreshes runs from elsewhere
    label = "Pull from server"
    command = "scripts/remote.sh pull"   # run in the project root
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

CONFIG_NAME = "runboard.toml"


@dataclass
class Config:
    project: Path
    title: str
    lang: str = "en"
    runs: Path = Path("runs")
    docs: Path | None = None
    doc_order: list[str] = field(default_factory=list)
    prereg: list[Path] = field(default_factory=list)
    prereg_tag: str = "step"
    datasets: list[Path] = field(default_factory=list)
    sync_label: str | None = None
    sync_command: str | None = None
    source: Path | None = None  # the runboard.toml that was loaded, if any


def find_config(start: Path) -> Path | None:
    for d in [start, *start.parents]:
        if (d / CONFIG_NAME).is_file():
            return d / CONFIG_NAME
    return None


def load(project: str | Path | None = None) -> Config:
    """Load config for ``project`` (default: nearest runboard.toml above cwd, else cwd)."""
    start = Path(project or Path.cwd()).resolve()
    path = find_config(start)
    root = path.parent if path else start
    raw = tomllib.loads(path.read_text()) if path else {}

    docs_raw = raw.get("docs", "docs")
    docs = (root / docs_raw) if docs_raw else None
    if docs is not None and not docs.is_dir():
        docs = None
    prereg = raw.get("prereg", [])
    if isinstance(prereg, str):
        prereg = [prereg]
    ds = raw.get("datasets", ["data"])
    if isinstance(ds, str):
        ds = [ds]
    sync = raw.get("sync", {})
    lang = raw.get("lang", "en")
    return Config(
        project=root,
        title=raw.get("title", root.name),
        lang=lang if lang in ("en", "ru") else "en",
        runs=root / raw.get("runs", "runs"),
        docs=docs,
        doc_order=list(raw.get("doc_order", [])),
        prereg=[root / p for p in prereg],
        prereg_tag=raw.get("prereg_tag", "step"),
        datasets=[root / d for d in ds],
        sync_label=sync.get("label") if sync.get("command") else None,
        sync_command=sync.get("command"),
        source=path,
    )


TEMPLATE = """\
# runboard project config. All keys are optional.
title = "{title}"
lang = "en"            # "en" | "ru"
runs = "runs"          # where Run() writes run directories
docs = "docs"          # markdown shown in the Docs tab ("" to disable)
# doc_order = ["plan", "experiments", "journal"]
# prereg = ["docs/experiments.md"]   # sections "## ..." citing a run id are shown with the run
# prereg_tag = "step"
datasets = ["data"]    # folders shown in the Data tab; run artifacts are always included

# [sync]
# label = "Pull from server"
# command = "rsync -az server:project/runs/ runs/"
"""
