"""Generate a self-contained demo project for runboard (all data is synthetic).

    uv run python examples/make_demo.py /tmp/runboard-demo
    uv run runboard -C /tmp/runboard-demo

Creates a git repo with runboard.toml, docs with pre-registered experiments, a few
tracked runs (a learning-rate sweep with curves, a grid ablation, a failed run, one
still running) and datasets in several formats, including graphs.
"""

from __future__ import annotations

import json
import math
import random
import subprocess
import sys
from pathlib import Path

from runboard import Run

TOML = """\
title = "vision-baselines (demo)"
lang = "en"
runs = "runs"
docs = "docs"
doc_order = ["plan", "experiments"]
prereg = ["docs/experiments.md"]
datasets = ["data"]

[sync]
label = "Pull from cluster"
command = "echo 'demo project: nothing to pull'"
"""

PLAN = """\
# Plan: small-image-model baselines

**Question.** Which optimizer and augmentation give the best accuracy for a fixed
compute budget on a CIFAR-sized benchmark?

| ID | Question | Status |
|---|---|---|
| E1 | Learning rate × optimizer sweep | done |
| E2 | Model size × augmentation ablation | done |
| E3 | Large-batch training | failed (OOM), to be retried |
"""

EXPERIMENTS = """\
# Experiment registry

Each section is written **before** the run (question, prediction) and completed after.

## E1 · Learning rate × optimizer sweep

**Prediction.** Adam peaks at lr=3e-4 (val acc ≈ 0.90); SGD needs a larger lr and ends ≈ 0.02 lower.

**Failure criterion.** No configuration exceeds 0.85.

**Run:** {e1}

**Result.** Adam 3e-4 is best (see Metrics → pivot). Prediction confirmed.

## E2 · Model size × augmentation ablation

**Prediction.** Mixup helps the large model most (+1–2 pp); the small model gains little.

**Run:** {e2}

**Result.** See the pivot: model × augment, mean ± std over 3 seeds.

## E3 · Large-batch training

**Prediction.** Batch 4096 with linear lr scaling matches batch 256 within 0.5 pp.

**Run:** {e3} (failed: out of memory). Retry with gradient accumulation.
"""


def git(repo: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)


def write_datasets(root: Path, rng: random.Random) -> None:
    d = root / "data"
    # 1. Transactions graph with a coordinated fraud ring (edge list + node table).
    tx = d / "transactions"
    tx.mkdir(parents=True)
    n_acc, ring = 400, list(range(390, 400))
    segments = ["retail", "retail", "retail", "sme", "premium"]
    with open(tx / "accounts.csv", "w") as f:
        f.write("account_id,segment,is_fraud,opened_days_ago\n")
        for i in range(n_acc):
            f.write(f"acc{i},{rng.choice(segments)},{int(i in ring)},{rng.randint(5, 2000)}\n")
    with open(tx / "transfers.csv", "w") as f:
        f.write("src,dst,amount,ts\n")
        for _ in range(1400):
            u = rng.randrange(390)
            v = (u + rng.choice([1, 2, 3, 5, 8, 13, rng.randrange(390)])) % 390
            f.write(f"acc{u},acc{v},{round(rng.lognormvariate(3.5, 0.9), 2)},{rng.randint(0, 180)}\n")
        for i, u in enumerate(ring):  # the ring: dense, fast pass-through
            for v in ring:
                if u != v and rng.random() < 0.45:
                    f.write(f"acc{u},acc{v},{round(900 * rng.uniform(0.9, 0.99), 2)},{170 + i % 5}\n")
            for _ in range(2):  # camouflage edges to normal accounts
                f.write(f"acc{u},acc{rng.randrange(390)},{round(rng.lognormvariate(3.5, 0.9), 2)},{rng.randint(0, 180)}\n")
    # 2. A community graph in GraphML (stochastic block model).
    g = d / "graphs"
    g.mkdir()
    sizes, comm = [70, 60, 50, 40], []
    for c, s in enumerate(sizes):
        comm += [c] * s
    nodes = "".join(f'<node id="n{i}"><data key="community">{c}</data></node>' for i, c in enumerate(comm))
    edges = []
    for i in range(len(comm)):
        for j in range(i + 1, len(comm)):
            if rng.random() < (0.08 if comm[i] == comm[j] else 0.004):
                edges.append(f'<edge source="n{i}" target="n{j}"/>')
    (g / "communities.graphml").write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n<graphml xmlns="http://graphml.graphdrawing.org/xmlns">\n'
        '<key id="community" for="node" attr.name="community" attr.type="int"/>\n'
        f'<graph edgedefault="undirected">{nodes}{"".join(edges)}</graph></graphml>\n')
    # 3. The same kind of graph as networkx node-link JSON (a small citation-like DAG).
    papers = [{"id": f"p{i}", "year": 2015 + i // 30, "field": rng.choice(["cv", "nlp", "rl"])} for i in range(150)]
    links = [{"source": f"p{i}", "target": f"p{rng.randrange(i)}"} for i in range(1, 150) for _ in range(rng.randint(1, 3))]
    (g / "citations.json").write_text(json.dumps({"directed": True, "nodes": papers, "links": links}))
    # 4. Plain tables.
    tab = d / "tables"
    tab.mkdir()
    with open(tab / "train_labels.jsonl", "w") as f:
        for i in range(2500):
            cls = rng.choice(["airplane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"])
            f.write(json.dumps({"image_id": i, "label": cls, "width": 32, "height": 32,
                                "brightness": round(rng.gauss(0.47, 0.12), 3), "split": "train" if i % 10 else "val"}) + "\n")
    (tab / "class_metrics.json").write_text(json.dumps([
        {"class": c, "precision": round(rng.uniform(0.82, 0.96), 3), "recall": round(rng.uniform(0.8, 0.97), 3)}
        for c in ["airplane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]]))


def make(root: Path) -> dict[str, str]:
    root.mkdir(parents=True, exist_ok=True)
    rng = random.Random(7)
    (root / "train.py").write_text("# toy training script for the runboard demo\n")
    (root / ".gitignore").write_text("runs/*/artifacts/\n")
    (root / "runboard.toml").write_text(TOML)
    (root / "docs").mkdir(exist_ok=True)
    (root / "docs" / "plan.md").write_text(PLAN)
    (root / "docs" / "experiments.md").write_text(EXPERIMENTS.format(e1="(pending)", e2="(pending)", e3="(pending)"))
    write_datasets(root, rng)
    git(root, "init", "-q")
    git(root, "config", "user.email", "demo@example.com")
    git(root, "config", "user.name", "demo")
    git(root, "add", ".")
    git(root, "commit", "-qm", "demo project")
    ids = {}

    # E1: lr × optimizer sweep with training curves.
    grid = {"optimizer": ["adam", "sgd"], "lr": [1e-3, 3e-4, 1e-4], "seeds": [0, 1, 2]}
    best = {("adam", 1e-3): 0.885, ("adam", 3e-4): 0.905, ("adam", 1e-4): 0.872,
            ("sgd", 1e-3): 0.884, ("sgd", 3e-4): 0.861, ("sgd", 1e-4): 0.823}
    with Run("lr_sweep", {"model": "resnet18", "dataset": "cifar10", "steps": 60, "batch_size": 256, "grid": grid},
             tags={"step": "E1"}, repo=root, snapshot=False) as run:
        for opt in grid["optimizer"]:
            for lr in grid["lr"]:
                for seed in grid["seeds"]:
                    top, tau = best[(opt, lr)] + rng.gauss(0, 0.004), (9 if opt == "adam" else 16) * (3e-4 / lr) ** 0.35
                    for step in range(0, 61, 3):
                        acc = top * (1 - math.exp(-step / tau)) + 0.1 * math.exp(-step / tau) + rng.gauss(0, 0.004)
                        loss = 2.3 * math.exp(-step / tau) + (1 - top) * 1.6 + abs(rng.gauss(0, 0.02))
                        run.log({"loss": round(loss, 4), "val_acc": round(acc, 4)}, optimizer=opt, lr=lr, seed=seed, step=step)
                    run.log({"final_val_acc": round(top, 4)}, optimizer=opt, lr=lr, seed=seed)
        run.finish({"best": {"optimizer": "adam", "lr": 3e-4, "val_acc": 0.905}})
    ids["e1"] = run.run_id

    # E2: model × augmentation ablation (final metrics only) + a predictions artifact.
    grid = {"model": ["small", "base", "large"], "augment": ["none", "mixup", "cutmix"], "seeds": [0, 1, 2]}
    base = {"small": 0.842, "base": 0.884, "large": 0.903}
    gain = {"none": {"small": 0, "base": 0, "large": 0}, "mixup": {"small": 0.004, "base": 0.011, "large": 0.018},
            "cutmix": {"small": 0.002, "base": 0.009, "large": 0.013}}
    with Run("augment_ablation", {"dataset": "cifar10", "epochs": 90, "grid": grid}, tags={"step": "E2"},
             repo=root, snapshot=False) as run:
        for m in grid["model"]:
            for a in grid["augment"]:
                for seed in grid["seeds"]:
                    acc = base[m] + gain[a][m] + rng.gauss(0, 0.003)
                    run.log({"test_acc": round(acc, 4), "macro_f1": round(acc - 0.006 + rng.gauss(0, 0.002), 4),
                             "params_m": {"small": 4.1, "base": 11.2, "large": 25.6}[m]}, model=m, augment=a, seed=seed)
        with open(run.artifacts / "predictions.csv", "w") as f:
            f.write("image_id,label,pred,confidence\n")
            for i in range(300):
                lab = rng.randrange(10)
                f.write(f"{i},{lab},{lab if rng.random() < 0.9 else rng.randrange(10)},{round(rng.uniform(0.3, 1), 3)}\n")
        run.finish({"best": {"model": "large", "augment": "mixup", "test_acc": 0.921}})
    ids["e2"] = run.run_id

    # E3: a failed large-batch run.
    try:
        with Run("lr_sweep", {"model": "resnet18", "dataset": "cifar10", "batch_size": 4096, "lr": 4.8e-3, "grid": {"seeds": [0]}},
                 tags={"step": "E3"}, repo=root, snapshot=False) as run:
            for step in range(0, 9, 3):
                run.log({"loss": round(2.3 - 0.05 * step, 4)}, seed=0, step=step)
            raise RuntimeError("CUDA out of memory. Tried to allocate 3.06 GiB")
    except RuntimeError:
        pass
    ids["e3"] = run.run_id

    # A run that is still going (no summary.json yet).
    live = Run("augment_ablation", {"dataset": "cifar100", "epochs": 90, "grid": {"model": ["base"], "seeds": [0]}},
               tags={"step": "E2"}, repo=root, snapshot=False)
    for step in range(0, 30, 3):
        live.log({"loss": round(4.6 * math.exp(-step / 20) + 0.9, 4)}, model="base", seed=0, step=step)

    (root / "docs" / "experiments.md").write_text(
        EXPERIMENTS.format(**{k: f"`{v}`" for k, v in ids.items()}))
    return ids


if __name__ == "__main__":
    target = Path(sys.argv[1] if len(sys.argv) > 1 else "runboard-demo")
    if target.exists() and any(target.iterdir()):
        raise SystemExit(f"{target} is not empty")
    print(make(target))
    print(f"now run: runboard -C {target}")
