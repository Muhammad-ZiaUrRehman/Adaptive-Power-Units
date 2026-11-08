"""Preservation and audit of the historical (v1) study (plan section 1).

Nothing in the historical folders is modified except, optionally, setting files read-only. All outputs go to the
v2 store's audit/ folder and to the archive folder.
"""
import json, os, re, shutil, stat, csv, time, glob, math
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
import pandas as pd

from .config import sha256_file, atomic_write_json, atomic_write_text, now, read_json

# counts quoted in the improvement plan (previous audit); compared, never assumed
REPORTED = dict(distinct_cls_jobs=483, completed_cls_jobs=464, failed_cls_jobs=19, completed_non_holdout=404)
V1_PILOT_ACTS = ["relu", "gelu", "silu", "mish", "apu_fixed", "apu_const", "apu_linear", "apu_sigmoid", "apu@0.8", "apu@0.9",
                 "apu@1.0", "apu@1.5", "apu@0.9,wz=0.1", "apu@1.0,wz=0.1"]
V1_LRSWEEP_ACTS = ["relu", "gelu", "silu", "apu"]
V1_MAIN_ACTS = ["relu", "leakyrelu", "prelu", "gelu", "silu", "mish", "esrelu", "apu"]
V1_ABLATION = ["relu", "apu_fixed", "apu_const", "apu_linear", "apu_sigmoid", "apu"]
V1_MAIN_GRID = [("cifar10", "resnet18"), ("cifar100", "resnet18"), ("cifar10", "vit_tiny"), ("cifar100", "vit_tiny"), ("cifar10", "simplecnn")]


# ---------------------------------------------------------------- archive with checksums
def _files(items):
    out = []
    for it in items:
        it = Path(it)
        if it.is_file():
            out.append((it, it.name))
        elif it.is_dir():
            for p in sorted(it.rglob("*")):
                if p.is_file():
                    out.append((p, str(Path(it.name) / p.relative_to(it))))
    return out


def archive_historical(items, archive_dir, copy_mode="auto", protect=False, margin_gb=5.0):
    """Checksum every historical file; copy into archive_dir when space allows (or copy_mode=True); verify copies.
    Returns a summary dict. An existing manifest is verified instead of copying again."""
    archive_dir = Path(archive_dir); archive_dir.mkdir(parents=True, exist_ok=True)
    present = [Path(i) for i in items if Path(i).exists()]
    missing = [str(i) for i in items if not Path(i).exists()]
    man_path = archive_dir / "MANIFEST.json"
    files = _files(present)
    total = sum(p.stat().st_size for p, _ in files)
    if man_path.exists():
        man = read_json(man_path); bad = []
        for e in man["files"]:
            src = Path(e["source"])
            if src.exists() and src.stat().st_size == e["size"] and sha256_file(src) != e["sha256"]:
                bad.append(e["relpath"])
        man["reverified"] = now(); man["changed_since_archive"] = bad
        atomic_write_json(man_path, man)
        return dict(manifest=str(man_path), n_files=len(man["files"]), bytes=man["bytes"], copied=man["copied"],
                    changed_since_archive=bad, missing_items=missing, reused_existing_manifest=True)
    free = shutil.disk_usage(archive_dir).free
    do_copy = copy_mode is True or (copy_mode == "auto" and free > total * 1.1 + margin_gb * 2**30)
    entries = []
    t0 = time.time()
    for i, (p, rel) in enumerate(files):
        st_ = p.stat(); h = sha256_file(p)
        e = dict(relpath=rel.replace("\\", "/"), source=str(p.resolve()), size=st_.st_size,
                 mtime=time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(st_.st_mtime)), sha256=h)
        if do_copy:
            dst = archive_dir / "files" / rel; dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, dst)
            e["copy_verified"] = sha256_file(dst) == h
        entries.append(e)
        if (i + 1) % 200 == 0:
            print(f"  checksummed {i + 1}/{len(files)} files ({time.time() - t0:.0f}s)", flush=True)
    if protect:
        for p, _ in files:
            try: os.chmod(p, stat.S_IREAD | stat.S_IRGRP | stat.S_IROTH)
            except Exception: pass
    man = dict(created=now(), items=[str(p.resolve()) for p in present], missing_items=missing, n_files=len(entries), bytes=total,
               copied=do_copy, copy_mode=copy_mode, free_bytes_before=free, originals_set_read_only=protect, files=entries,
               note=("copies verified by SHA-256" if do_copy else "not enough free space for a full copy: originals are "
                     "checksummed in place" + (" and set read-only" if protect else "")))
    atomic_write_json(man_path, man)
    with open(archive_dir / "MANIFEST.sha256", "w", encoding="utf-8") as f:
        for e in entries: f.write(f"{e['sha256']}  {e['relpath']}\n")
    if do_copy and not all(e.get("copy_verified") for e in entries):
        raise RuntimeError("some archive copies failed checksum verification")
    return dict(manifest=str(man_path), n_files=len(entries), bytes=total, copied=do_copy, missing_items=missing,
                reused_existing_manifest=False)


# ---------------------------------------------------------------- historical records
def load_v1_results(old_root):
    rows, gan, bad = [], [], []
    for f in sorted((Path(old_root) / "results").glob("*.json")):
        try:
            r = read_json(f)
        except Exception as e:
            bad.append(dict(file=f.name, error=repr(e)[:200])); continue
        if r.get("kind") == "gan":
            j = r["job"]; gan.append(dict(run_id=r["run_id"], dataset=j["dataset"], arch=j["arch"], act=j["act"], seed=j["seed"],
                                          epochs=j["epochs"], fd=r.get("fd"), is_like=r.get("is_like"), fd_error=r.get("fd_error"))); continue
        j = r["job"]; h = r.get("history", [])
        rows.append(dict(run_id=r["run_id"], file=f.name, dataset=j["dataset"], arch=j["arch"], act=j["act"], seed=j["seed"],
                         epochs=j["epochs"], holdout=bool(j.get("holdout", False)), lr_scale=float(j.get("lr_scale", 1.0)),
                         diverged=bool(r.get("diverged", False)), epochs_logged=len(h), top1=r["final"]["val_top1"],
                         top5=r["final"].get("val_top5"), params=r.get("params"), act_params=r.get("act_params"),
                         median_epoch_time=r.get("median_epoch_time"), compiled=r.get("compiled")))
    return pd.DataFrame(rows), pd.DataFrame(gan), bad


def v1_stages(row):
    """All historical stage scopes a classification job belongs to (shared jobs belong to several)."""
    s = []
    if row["holdout"]:
        if row["lr_scale"] != 1.0:
            s.append("lrsweep")
        else:
            if row["act"] in V1_PILOT_ACTS: s.append("pilot")
            if row["act"] in V1_LRSWEEP_ACTS: s.append("lrsweep")
    else:
        d, a, act = row["dataset"], row["arch"], row["act"]
        if d == "tinyimagenet": s.append("tiny")
        elif d in ("mnist", "emnist"): s.append("supp")
        else:
            if (d, a) in V1_MAIN_GRID and act in V1_MAIN_ACTS: s.append("main")
            if d == "cifar100" and a == "resnet18" and act in V1_ABLATION: s.append("ablation")
    return s or ["unclassified"]


def reconcile(df):
    """Counts per explicit scope. Shared jobs are counted once in the totals and listed per stage separately."""
    df = df.copy(); df["stages"] = df.apply(v1_stages, axis=1)
    rows = []
    def add(scope, d):
        rows.append(dict(scope=scope, distinct_jobs=int(d.run_id.nunique()), completed=int((~d.diverged).sum()), failed=int(d.diverged.sum())))
    add("all classification jobs", df)
    add("non-holdout (test-evaluated) jobs", df[~df.holdout])
    add("holdout jobs (pilot + LR sweep)", df[df.holdout])
    for st in ["pilot", "lrsweep", "main", "ablation", "tiny", "supp", "unclassified"]:
        d = df[df.stages.apply(lambda s: st in s)]
        if len(d): add(f"stage: {st}", d)
    shared = df[df.stages.apply(len) > 1]
    add("jobs shared by two stages", shared)
    t = pd.DataFrame(rows)
    allr = t.iloc[0]; nh = t.iloc[1]
    checks = [("distinct classification jobs", REPORTED["distinct_cls_jobs"], int(allr.distinct_jobs)),
              ("completed classification jobs", REPORTED["completed_cls_jobs"], int(allr.completed)),
              ("failed (diverged) classification jobs", REPORTED["failed_cls_jobs"], int(allr.failed)),
              ("completed non-holdout jobs", REPORTED["completed_non_holdout"], int(nh.completed))]
    c = pd.DataFrame([dict(quantity=q, reported_in_plan=a, recovered=b, matches=(a == b)) for q, a, b in checks])
    return t, c, df


def _pm(s):
    m = re.match(r"\s*([\d.]+)\s*\u00b1\s*([\d.]+)", str(s))
    return (float(m.group(1)), float(m.group(2))) if m else (None, None)


def compare_saved_outputs(df, old_root, old_notebook):
    """Recompute historical summaries from raw JSON and compare with the saved table and notebook outputs."""
    corr = []
    nh = df[~df.holdout]
    g = nh[~nh.diverged].groupby(["dataset", "arch", "act"]).top1.agg(["mean", "std", "count"])
    tpath = Path(old_root) / "tables" / "table2_main_results.csv"
    if tpath.exists():
        t = pd.read_csv(tpath, encoding="utf-8")
        for _, r in t.iterrows():
            m, s = _pm(r["top1"]); key = (r["dataset"], r["arch"], r["act"])
            if m is None:
                rec_div = int(nh[(nh.dataset == key[0]) & (nh.arch == key[1]) & (nh.act == key[2])].diverged.sum())
                if str(r["top1"]).strip() == "diverged" and rec_div == int(r["diverged"]):
                    continue
                corr.append(dict(source="table2_main_results.csv", item="/".join(key), saved=r["top1"], recovered=f"diverged={rec_div}",
                                 abs_diff=None, note="divergence count differs")); continue
            if key not in g.index:
                corr.append(dict(source="table2_main_results.csv", item="/".join(key), saved=r["top1"], recovered="no raw runs",
                                 abs_diff=None, note="value without raw record")); continue
            gm, gs = g.loc[key, "mean"], g.loc[key, "std"]
            if abs(round(gm, 2) - m) > 0.0051 or abs(round(gs, 2) - s) > 0.0051:
                corr.append(dict(source="table2_main_results.csv", item="/".join(key), saved=r["top1"], recovered=f"{gm:.3f} +/- {gs:.3f}",
                                 abs_diff=abs(gm - m), note="rounded value differs"))
        abl = t[(t.dataset == "cifar100") & (t.arch == "resnet18") & (~t.act.isin(V1_MAIN_ACTS))]
        if len(abl):
            corr.append(dict(source="table2_main_results.csv", item="cifar100/resnet18 rows", saved=", ".join(abl.act),
                             recovered="ablation-only variants", abs_diff=None,
                             note="Table 2 mixes ablation variants into the main comparison; v2 reports them separately"))
    else:
        corr.append(dict(source="table2_main_results.csv", item="file", saved="unavailable", recovered="", abs_diff=None, note="saved table not found"))
    # notebook findings text
    nb = Path(old_notebook)
    if nb.exists():
        cells = json.load(open(nb, encoding="utf-8"))["cells"]
        text = "\n".join("".join(o.get("text", "")) for c in cells for o in c.get("outputs", []) if o.get("output_type") == "stream")
        pat = re.compile(r"- \*\*(\w+) / ([^*]+)\*\*: APU ([\d.]+) \u00b1 ([\d.]+) \(n=(\d+)\)\. Best baseline (.+?) ([\d.]+)\.")
        arch_map = {"ResNet-18": "resnet18", "SimpleCNN": "simplecnn", "ViT-Tiny": "vit_tiny"}
        found = 0
        for m in pat.finditer(text):
            found += 1
            ds, ar = m.group(1), arch_map.get(m.group(2).strip(), m.group(2).strip())
            key = (ds, ar, "apu")
            if key in g.index:
                gm, n = g.loc[key, "mean"], int(g.loc[key, "count"])
                if abs(round(gm, 2) - float(m.group(3))) > 0.0051 or n != int(m.group(5)):
                    corr.append(dict(source="notebook findings", item=f"{ds}/{ar}/apu", saved=m.group(3), recovered=f"{gm:.3f} (n={n})",
                                     abs_diff=abs(gm - float(m.group(3))), note="findings text differs from raw records"))
        if not found:
            corr.append(dict(source="notebook findings", item="text", saved="not found", recovered="", abs_diff=None, note="no findings block parsed"))
    # numbers quoted in the improvement plan
    key_a, key_g = ("cifar10", "simplecnn", "apu"), ("cifar10", "simplecnn", "gelu")
    plan_checks = []
    if key_a in g.index and key_g in g.index:
        a, b = g.loc[key_a, "mean"], g.loc[key_g, "mean"]
        plan_checks += [("SimpleCNN CIFAR-10 APU mean", 83.326, a), ("SimpleCNN CIFAR-10 GELU mean", 82.530, b), ("APU - GELU", 0.796, a - b)]
    sc = nh[(nh.dataset == "cifar10") & (nh.arch == "simplecnn") & (nh.act == "apu")]
    if len(sc) and sc.act_params.notna().any():
        plan_checks.append(("SimpleCNN APU learned activation parameters", 9, float(sc.act_params.iloc[0])))
    rr = nh[(nh.dataset == "cifar10") & (nh.arch == "resnet18") & (~nh.diverged)]
    if len(rr):
        er = rr.groupby("act").median_epoch_time.mean()
        if "relu" in er and "apu" in er:
            plan_checks.append(("ResNet-18 CIFAR-10 epoch time APU/ReLU - 1 (%)", 39.0, 100 * (er["apu"] / er["relu"] - 1)))
    tol = {"SimpleCNN CIFAR-10 APU mean": 0.0005, "SimpleCNN CIFAR-10 GELU mean": 0.0005, "APU - GELU": 0.0005,
           "SimpleCNN APU learned activation parameters": 0.0, "ResNet-18 CIFAR-10 epoch time APU/ReLU - 1 (%)": 1.0}
    for item, saved, rec in plan_checks:
        if abs(saved - rec) > tol[item]:
            corr.append(dict(source="improvement plan", item=item, saved=saved, recovered=round(rec, 4), abs_diff=abs(saved - rec), note="differs"))
    return pd.DataFrame(corr, columns=["source", "item", "saved", "recovered", "abs_diff", "note"]), pd.DataFrame(
        [dict(item=i, quoted=s, recovered=round(r, 4)) for i, s, r in plan_checks])


def wins_count(df):
    nh = df[(~df.holdout) & (~df.diverged)]
    rows = []
    for (ds, ar), g in nh.groupby(["dataset", "arch"]):
        m = g.groupby("act").top1.mean()
        if "apu" not in m: continue
        base = m[[a for a in m.index if not a.startswith("apu")]]
        if not len(base): continue
        rows.append(dict(dataset=ds, arch=ar, apu=m["apu"], best_baseline=base.idxmax(), best_value=base.max(), apu_highest=m["apu"] > base.max()))
    return pd.DataFrame(rows)


def historical_secondary(old_root, env_files, gan_df):
    """Labels for secondary historical analyses (kept with their limitations)."""
    out = {}
    n_eval = None
    for f in env_files:
        try:
            n_eval = read_json(f)["profile"]["gan"]["n_eval"]; break
        except Exception:
            continue
    gen = (max(n_eval // 256, 1) * 256) if n_eval else None
    out["gan"] = dict(metric="Frechet distance in the feature space of a small classifier trained on the real data (NOT Inception-based FID)",
                      generated_samples_per_score=gen if gen else "unavailable", real_feature_samples=n_eval or "unavailable",
                      n_runs=int(len(gan_df)))
    robust = list((Path(old_root) / "robust").glob("*.json"))
    corrs, srcs = set(), set()
    for f in robust:
        try:
            r = read_json(f)
            if r.get("kind") == "pgd": continue
            corrs |= set(r.get("acc", {}).keys()); srcs.add(r.get("source", "unknown"))
        except Exception:
            pass
    out["corruption"] = dict(corruption_types=sorted(corrs), n_types=len(corrs), sources=sorted(srcs),
                             label=f"{len(corrs)} re-implemented corruption types; not the official CIFAR-10-C / CIFAR-100-C benchmark")
    out["pgd_files"] = sorted(p.name for p in (Path(old_root) / "robust").glob("pgd_*.csv"))
    return out


def historical_environment(old_root):
    envs = sorted(Path(old_root).glob("env_*.json")); sessions = []
    for f in envs:
        try:
            e = read_json(f)["env"]; sessions.append(dict(file=f.name, **{k: e.get(k) for k in ("date", "gpu", "torch", "cuda", "cudnn", "python", "platform")}))
        except Exception as ex:
            sessions.append(dict(file=f.name, error=repr(ex)[:100]))
    return pd.DataFrame(sessions), envs


def shallow_cifar100_inspected(old_root, df, old_notebook):
    """The confirmation setting must not have been run or inspected in the historical study."""
    hits = df[(df.dataset == "cifar100") & (df.arch == "simplecnn")] if len(df) else df
    files = [p.name for p in Path(old_root).rglob("*") if p.is_file() and "cifar100" in p.name.lower() and "simplecnn" in p.name.lower()]
    txt_hits = 0
    if Path(old_notebook).exists():
        cells = json.load(open(old_notebook, encoding="utf-8"))["cells"]
        txt = "\n".join("".join(o.get("text", "")) if isinstance(o.get("text"), (str, list)) else "" for c in cells for o in c.get("outputs", []))
        txt_hits = len(re.findall(r"cifar100[-_ /]+simplecnn", txt, flags=re.I))
    return dict(result_runs=int(len(hits)), files=files, output_text_mentions=txt_hits,
                inspected=bool(len(hits) or files or txt_hits))


def historical_preact_range(old_root):
    """Pre-activation ranges measured in the historical exponent analysis (analysis/exp_*.pt), if present."""
    import torch
    vals = []
    for f in sorted((Path(old_root) / "analysis").glob("exp_*.pt")):
        try:
            d = torch.load(f, map_location="cpu", weights_only=False)
            for layer in d.values():
                z = layer.get("z")
                if z is not None and len(z):
                    vals.append((float(np.min(z)), float(np.max(z)), float(np.percentile(z, 0.1)), float(np.percentile(z, 99.9))))
        except Exception:
            continue
    if not vals:
        return None
    v = np.array(vals)
    return dict(source="historical analysis/exp_*.pt (sampled pre-activations of trained APU models)", n_layers=len(vals),
                min=float(v[:, 0].min()), max=float(v[:, 1].max()), q001=float(v[:, 2].min()), q999=float(v[:, 3].max()))
