"""Regenerate docs/screenshots/*.png from the synthetic demo project.

    uv run python examples/screenshots.py

Builds the demo (examples/make_demo.py) in a temp dir, serves it on a free port and
captures pages with headless Chrome/Chromium in a throwaway profile.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import threading
import urllib.request
from pathlib import Path
from urllib.parse import quote

sys.path.insert(0, str(Path(__file__).parent))
from make_demo import make  # noqa: E402

from runboard.config import load  # noqa: E402
from runboard.server import make_server  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "docs" / "screenshots"
BROWSERS = ("google-chrome", "google-chrome-stable", "chromium", "chromium-browser")


def shot(chrome: str, url: str, out: Path, height: int) -> None:
    with tempfile.TemporaryDirectory() as profile:
        subprocess.run([
            chrome, "--headless=new", "--disable-gpu", "--hide-scrollbars", "--no-first-run",
            "--no-default-browser-check", "--disable-extensions", f"--user-data-dir={profile}",
            f"--window-size=1440,{height}", "--force-device-scale-factor=1",
            "--virtual-time-budget=10000", f"--screenshot={out}", url,
        ], check=True, capture_output=True, timeout=120)
    print(f"  {out.name}")


def main() -> None:
    chrome = next((b for b in map(shutil.which, BROWSERS) if b), None)
    if not chrome:
        raise SystemExit("needs Google Chrome or Chromium on PATH")
    OUT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        ids = make(Path(tmp) / "demo")
        server = make_server(load(Path(tmp) / "demo"), port=0)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        base = f"http://127.0.0.1:{server.server_address[1]}/"
        runs = json.loads(urllib.request.urlopen(base + "api/runs").read())
        assert {r["run_id"] for r in runs} >= set(ids.values())
        data = lambda p: quote(p, safe="")  # noqa: E731
        graph = ("?" + "&".join(["node=acc395", "color=is_fraud", "nodes=data/transactions/accounts.csv",
                                 "node_id=account_id"]))
        pages = [
            ("runs", "", "#/runs", 820),
            ("run-overview", "", f"#/run/{ids['e2']}/overview", 1100),
            ("metrics-pivot", "", f"#/run/{ids['e2']}/metrics", 1150),
            ("metrics-curves", "", f"#/run/{ids['e1']}/metrics", 1700),
            ("compare", "", f"#/compare/{ids['e1']},{ids['e3']}", 900),
            ("data-columns", "", f"#/data/{data('data/tables/train_labels.jsonl')}/columns", 1100),
            ("data-graph", graph, f"#/data/{data('data/transactions/transfers.csv')}/graph", 1650),
            ("docs", "", "#/docs/experiments", 900),
            ("runs-dark", "?theme=dark", "#/runs", 820),
            ("data-graph-dark", graph + "&theme=dark", f"#/data/{data('data/transactions/transfers.csv')}/graph", 1650),
        ]
        print(f"writing {OUT}")
        for name, query, frag, h in pages:
            shot(chrome, base + query + frag, OUT / f"{name}.png", h)
        server.shutdown()


if __name__ == "__main__":
    main()
