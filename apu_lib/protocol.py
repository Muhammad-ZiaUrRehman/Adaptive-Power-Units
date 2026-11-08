"""Predeclared protocol, validation-only selection, locks and job lists (plan sections 2 and 4).

Order of operations enforced in code:
  1. protocol.json is written once before any search run. Changing it later raises ProtocolError.
  2. The equal-budget validation search runs; selection uses validation metrics only.
  3. lock.json (protocol digest + selected configurations + every final-fit digest) is written once.
  4. Final fits train without touching the test partition.
  5. evaluate.evaluate_test refuses to run unless the lock lists the run; each run is evaluated on test once.
Extensions have their own lock (lock_extensions.json), written before their fits.
"""
from pathlib import Path
import numpy as np

from .config import digest_of, atomic_write_json, read_json, now, ProtocolError
from .activations import CORE_METHODS, STANDARD, ADAPTIVE, CONTROLS
from .train import make_config, HISTORICAL_SHALLOW_RECIPE

DECISION_RULES = {
    "A": "Full adaptation exceeds the frozen profile and the strongest validated standard and adaptive baselines in both shallow datasets -> evidence for learned power adaptation in these tested shallow settings, with stated uncertainty and measured costs.",
    "B": "Full and frozen profiles perform similarly -> a useful fixed profile may explain any gain; learning has not demonstrated additional value.",
    "C": "Benefit appears only on CIFAR-10 -> a restricted exploratory observation requiring further confirmation.",
    "D": "Accuracy improves but runtime increases -> an accuracy and computation trade-off, not general efficiency.",
    "E": "No reproducible advantage over strong controls -> revise the central claim or pursue a carefully supported behavioural analysis.",
}


def build_protocol(profile, data_ctx, exec_ctx, source_rev):
    P = profile
    return dict(
        protocol_version=1, created=now(),
        question=("Does learning a bounded input-dependent power improve shallow image classification over a matched frozen "
                  "profile and competitive learned activations, under equal validation and training budgets, and at what "
                  "computational cost?"),
        methods=list(CORE_METHODS), groups=dict(standard=list(STANDARD), adaptive=list(ADAPTIVE), controls=list(CONTROLS), proposed=["apu"]),
        fixed_recipe=dict(HISTORICAL_SHALLOW_RECIPE, epochs=P["epochs"]),
        search=dict(grid=P["grid"], selection_seeds=P["selection_seeds"], epochs=P["epochs"],
                    criterion="mean final-epoch validation top-1 over the selection seeds",
                    tie_break="earlier candidate in grid order (grid[0] is the historical recipe)",
                    failed_candidate="a candidate with any failed selection seed is not selectable",
                    budget="identical grid, selection seeds and epochs for every method, including the frozen control"),
        final_seeds=P["final_seeds"],
        stage1=dict(role="primary (equal-search arm) and fixed-recipe arm", dataset=data_ctx["stage1_dataset"], arch="simplecnn", depth=2,
                    epochs=P["epochs"]),
        stage2=dict(role="predeclared confirmation of the selected protocol (no new selection)", dataset=data_ctx["stage2_dataset"],
                    arch="simplecnn", depth=2, epochs=P["epochs"]),
        analysis=dict(
            primary_outcome="final-epoch test top-1 of the final-epoch checkpoint (checkpoint rule fixed in advance)",
            primary_comparison="apu vs apu_frozen, equal-search arm, stage 1; repeated as the confirmation test in stage 2",
            secondary_family=["apu vs strongest validated standard (relu, gelu)", "apu vs strongest validated adaptive (prelu, acon_c)",
                              "apu vs apu_const (learned constant power)"],
            strongest_rule="highest mean final validation top-1 among the final fits of that group (validation only)",
            alpha=0.05, ci_level=0.95, multiplicity="primary comparison alone at alpha; Holm over the secondary family within each stage",
            paired_rule="paired t-interval when every compared seed has identical backbone-initialisation checksums and the keyed data "
                        "plan was verified identical; otherwise Welch",
            seeds="fixed in advance; no seeds are added based on significance", failures="retained and reported with denominators",
            exploratory="everything else (fixed-recipe arm, diagnostics, cost correlations, historical results) is labelled exploratory"),
        first_passage=dict(metric="val_top1", threshold_fraction=0.95,
                           threshold_rule="0.95 x mean final validation top-1 of relu in the same arm and setting",
                           budget="configured epochs (identical across methods)", censoring="runs that never reach it stay visible as censored"),
        decision_rules=DECISION_RULES,
        extensions=dict(trigger="decision A, or FORCE_EXTENSIONS (then labelled exploratory)",
                        depth_study=dict(depths=P["ext_depths"], methods=["relu", "gelu", "apu_frozen", "apu"], dataset=data_ctx["stage1_dataset"],
                                         hyperparameters="stage-1 selected configuration of each method", seeds=P["final_seeds"]),
                        pade=dict(method="pade", search="same grid, selection seeds and epochs as stage 1", seeds=P["final_seeds"])),
        execution=exec_ctx, data=data_ctx, source_revision=source_rev,
    )


def freeze_protocol(path, protocol):
    """Write once. A later call with different content raises instead of overwriting."""
    path = Path(path)
    body = {k: v for k, v in protocol.items() if k != "created"}
    dg = digest_of(body)
    if path.exists():
        old = read_json(path)
        if old["protocol_digest"] != dg:
            raise ProtocolError(f"{path} is frozen with digest {old['protocol_digest'][:16]}; the current settings give {dg[:16]}. "
                                "Restore the original settings, or start a new protocol file deliberately and report it.")
        return old
    p = dict(protocol, protocol_digest=dg)
    atomic_write_json(path, p)
    return p


def _cfg(ctx, dataset, method, seed, lr, wd, epochs, depth=2):
    return make_config(dataset_version=ctx["versions"][dataset], split=ctx["splits"][dataset], arch="simplecnn", depth=depth,
                       act_name=method, seed=seed, epochs=epochs, lr=lr, weight_decay=wd, precision=ctx["precision"],
                       compile=ctx["compile"], deterministic=ctx["deterministic"], diag_epochs=ctx["diag_epochs"],
                       probe_n=ctx["probe_n"], source_rev=ctx["source_rev"], ckpt_every=ctx["ckpt_every"],
                       batch_size=ctx["batch_size"], train_limit=ctx.get("train_limit"), val_limit=ctx.get("val_limit"))


def search_jobs(protocol, ctx, methods=None, dataset=None, depth=2, tag="s1_search"):
    S = protocol["search"]; ds = dataset or protocol["stage1"]["dataset"]; methods = methods or protocol["methods"]
    jobs = []
    for s in S["selection_seeds"]:
        for ci, c in enumerate(S["grid"]):
            for m in methods:
                jobs.append((_cfg(ctx, ds, m, s, c["lr"], c["weight_decay"], S["epochs"], depth), [tag, f"cand{ci}"]))
    return jobs


def fixed_jobs(protocol, ctx):
    R = protocol["fixed_recipe"]; ds = protocol["stage1"]["dataset"]
    return [(_cfg(ctx, ds, m, s, R["lr"], R["weight_decay"], protocol["stage1"]["epochs"]), ["s1_fixed"])
            for s in protocol["final_seeds"] for m in protocol["methods"]]


def selected_jobs(protocol, ctx, selection, stage="stage1", tag="s1_selected", methods=None, depth=2):
    ds = protocol[stage]["dataset"]; E = protocol[stage]["epochs"]; methods = methods or protocol["methods"]
    jobs = []
    for s in protocol["final_seeds"]:
        for m in methods:
            sel = selection[m]
            if sel is None:
                continue
            jobs.append((_cfg(ctx, ds, m, s, sel["lr"], sel["weight_decay"], E, depth), [tag]))
    return jobs


def select(store, protocol, search, methods=None):
    """Validation-only selection. search: list of (cfg, tags) as produced by search_jobs."""
    from .config import digest_of as _d
    S = protocol["search"]; methods = methods or protocol["methods"]
    table, selection = [], {}
    for m in methods:
        best = None
        for ci, c in enumerate(S["grid"]):
            vals, fails, digests = [], 0, []
            for cfg, _ in search:
                if cfg["activation"]["name"] != m or cfg["optimizer"]["lr"] != c["lr"] or cfg["optimizer"]["weight_decay"] != c["weight_decay"]:
                    continue
                d = _d(cfg); r = store.record(d); digests.append(d[:16])
                if r and r["status"] == "completed":
                    vals.append(r["final_metrics"]["val_top1"])
                else:
                    fails += 1
            ok = fails == 0 and len(vals) == len(S["selection_seeds"])
            mean = float(np.mean(vals)) if vals else float("nan")
            table.append(dict(method=m, candidate=ci, lr=c["lr"], weight_decay=c["weight_decay"], val_top1_mean=mean,
                              n_completed=len(vals), n_failed_or_missing=fails, selectable=ok, run_ids=",".join(digests)))
            if ok and (best is None or mean > best[1]):
                best = (ci, mean)
        if best is None:
            selection[m] = None
        else:
            c = S["grid"][best[0]]
            lrs = sorted({g["lr"] for g in S["grid"]}); wds = sorted({g["weight_decay"] for g in S["grid"]})
            selection[m] = dict(candidate=best[0], lr=c["lr"], weight_decay=c["weight_decay"], val_top1_mean=best[1],
                                at_grid_edge=dict(lr=c["lr"] in (lrs[0], lrs[-1]) and len(lrs) > 1,
                                                  weight_decay=c["weight_decay"] in (wds[0], wds[-1]) and len(wds) > 1))
    return selection, table


def write_lock(path, protocol, selection, final_jobs, extra=None):
    path = Path(path)
    digests = sorted({digest_of(cfg) for cfg, _ in final_jobs})
    body = dict(protocol_digest=protocol["protocol_digest"], selection=selection, final_fit_digests=digests, extra=extra or {})
    ld = digest_of(body)
    if path.exists():
        old = read_json(path)
        if old["lock_digest"] != ld:
            raise ProtocolError(f"{path.name} already exists with different content; locks are written once")
        return old
    lock = dict(body, lock_digest=ld, created=now(),
                note="Written before any final fit was trained or any new test metric was computed.")
    atomic_write_json(path, lock)
    return lock


def load_lock(path):
    path = Path(path)
    if not path.exists():
        return None
    lock = read_json(path)
    body = {k: lock[k] for k in ("protocol_digest", "selection", "final_fit_digests", "extra")}
    if digest_of(body) != lock["lock_digest"]:
        raise ProtocolError(f"{path.name} was modified after it was written")
    return lock
