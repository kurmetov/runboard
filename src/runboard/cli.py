"""runboard command line.

    runboard [serve] [-C DIR] [--port 5050] [--host 127.0.0.1] [--open]
    runboard ls [-C DIR] [-n 20]
    runboard init [-C DIR]
"""

from __future__ import annotations

import argparse
import sys
import webbrowser
from pathlib import Path

from . import __version__
from .config import CONFIG_NAME, TEMPLATE, load
from .server import Board, make_server


def cmd_serve(args) -> None:
    cfg = load(args.project)
    try:
        server = make_server(cfg, args.host, args.port)
    except OSError as e:
        raise SystemExit(f"runboard: port {args.port} is busy ({e.strerror}). It may already be running: "
                         f"open http://{args.host}:{args.port} or pass --port <other>.") from None
    url = f"http://{args.host}:{server.server_address[1]}"
    src = cfg.source or "defaults (no runboard.toml)"
    print(f"runboard {__version__}: {url}\n  project: {cfg.project}\n  runs:    {cfg.runs}\n  config:  {src}",
          flush=True)
    if args.open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


def cmd_ls(args) -> None:
    runs = Board(load(args.project)).runs()[: args.n]
    if not runs:
        print("no runs")
        return
    w = max(len(r["run_id"]) for r in runs)
    for r in runs:
        tags = " ".join(f"{k}={v}" for k, v in (r["tags"] or {}).items())
        el = r["elapsed_s"]
        dur = "" if el is None else f"{el:.0f}s" if el < 60 else f"{el / 60:.1f}m"
        dirty = "*" if r["git_dirty"] else " "
        print(f"{r['run_id']:<{w}}  {r['status']:<10} {r['git_commit'] or '-':<10}{dirty} {dur:>7}  {tags}")


def cmd_init(args) -> None:
    root = Path(args.project or ".").resolve()
    path = root / CONFIG_NAME
    if path.exists():
        raise SystemExit(f"runboard: {path} already exists")
    path.write_text(TEMPLATE.format(title=root.name))
    print(f"created {path}")


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    ap = argparse.ArgumentParser(prog="runboard", description="File-based experiment tracking browser")
    ap.add_argument("--version", action="version", version=f"runboard {__version__}")
    sub = ap.add_subparsers(dest="cmd")

    def common(p):
        p.add_argument("-C", "--project", help="project directory (default: nearest runboard.toml, else cwd)")

    p = sub.add_parser("serve", help="start the browser (default)")
    common(p)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=5050)
    p.add_argument("--open", action="store_true", help="open in the default web browser")
    p.set_defaults(fn=cmd_serve)
    p = sub.add_parser("ls", help="list recent runs in the terminal")
    common(p)
    p.add_argument("-n", type=int, default=20)
    p.set_defaults(fn=cmd_ls)
    p = sub.add_parser("init", help=f"create a {CONFIG_NAME} template")
    common(p)
    p.set_defaults(fn=cmd_init)

    if not argv or argv[0].startswith("-") and argv[0] not in ("-h", "--help", "--version"):
        argv = ["serve", *argv]
    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
