"""Build the reader-facing views of the run record.

Writes experiments/<stage>.csv (one row per run, read from apu_v2_runs/) and copies
apu_v2_runs/paper_assets/ (tables, figures, CLAIMS.md, CHECKLIST.md) to results/.
The record in apu_v2_runs/ is only read, never changed.

Usage (from the repository root):  python scripts/build_indexes.py
"""
import csv, json, shutil, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STORE = ROOT / (sys.argv[1] if len(sys.argv) > 1 else "apu_v2_runs")
STAGES = {"s1_search": "search", "s1_selected": "primary", "s2_confirm": "confirmation", "s1_fixed": "common_recipe"}
FIELDS = ["run_id", "stage", "dataset", "method", "seed", "lr", "weight_decay", "epochs", "status", "failure_kind",
          "val_top1", "test_top1", "test_top5", "trainable_params", "activation_params", "median_epoch_s", "config_digest"]


def load(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def main():
    if not (STORE / "runs").exists():
        sys.exit(f"{STORE} has no runs/ folder")
    rows = {s: [] for s in STAGES.values()}
    for rp in sorted((STORE / "runs").glob("*.json")):
        if rp.name.endswith(".test.json"):
            continue
        r = load(rp)
        cfg = load(STORE / "configs" / rp.name)["config"]
        tp = STORE / "runs" / rp.name.replace(".json", ".test.json")
        t = load(tp) if tp.exists() else {}
        fm, pc = r.get("final_metrics") or {}, r.get("param_counts") or {}
        base = dict(run_id=r["run_id"], dataset=cfg["dataset"]["name"], method=cfg["activation"]["name"], seed=cfg["seed"],
                    lr=cfg["optimizer"]["lr"], weight_decay=cfg["optimizer"]["weight_decay"], epochs=cfg["epochs"],
                    status=r.get("status"), failure_kind=r.get("failure_kind") or "", val_top1=fm.get("val_top1", ""),
                    test_top1=t.get("test_top1", ""), test_top5=t.get("test_top5", ""), trainable_params=pc.get("trainable", ""),
                    activation_params=pc.get("activation_trainable", ""),
                    median_epoch_s=(r.get("times") or {}).get("median_epoch_train_s", ""), config_digest=r["digest"])
        for tag in r.get("tags", []):
            if tag in STAGES:
                rows[STAGES[tag]].append(dict(base, stage=STAGES[tag]))
    out = ROOT / "experiments"; out.mkdir(exist_ok=True)
    for stage, rs in rows.items():
        rs.sort(key=lambda x: (x["method"], x["lr"], x["weight_decay"], x["seed"]))
        with open(out / f"{stage}.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS); w.writeheader(); w.writerows(rs)
        print(f"experiments/{stage}.csv: {len(rs)} runs")
    src, dst = STORE / "paper_assets", ROOT / "results"
    for sub in ("tables", "figures"):
        if (src / sub).exists():
            shutil.copytree(src / sub, dst / sub, dirs_exist_ok=True)
    for name in ("CLAIMS.md", "CHECKLIST.md"):
        if (src / name).exists():
            dst.mkdir(exist_ok=True); shutil.copy2(src / name, dst / name)
    print("results/ refreshed from", src.relative_to(ROOT))


if __name__ == "__main__":
    main()
