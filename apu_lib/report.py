"""Tables, statistics, figures, decision and checklist built only from structured records (plan sections 4-7).

Rules followed here: sample standard deviations across independent training seeds; percentage-point differences
for accuracy and percent changes for cost are kept in separate columns; failed runs stay in every table with their
denominators; seed counts, joins and missing values are validated before a figure is drawn; measured curves are
never interpolated; runs that never reach a first-passage threshold stay visible as censored.
"""
import math, json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

from .config import read_json, atomic_write_json, atomic_write_text, sha256_file, now

METHOD_LABEL = dict(relu="ReLU", gelu="GELU", prelu="PReLU", apu_frozen="APU frozen profile", apu_const="Learned constant power",
                    apu="APU (full, learned)", acon_c="ACON-C", pade="Pade (safe PAU)", silu="SiLU", mish="Mish")
METHOD_ORDER = ["relu", "gelu", "prelu", "acon_c", "apu_const", "apu_frozen", "apu", "pade"]
COLORS = dict(relu="#4D4D4D", gelu="#009E73", prelu="#0072B2", acon_c="#CC79A7", apu_const="#E69F00", apu_frozen="#56B4E9",
              apu="#D55E00", pade="#7F3F98")
MARKERS = dict(relu="o", gelu="D", prelu="^", acon_c="P", apu_const="s", apu_frozen="v", apu="*", pade="X")


def script_id():
    return f"apu_lib/report.py sha256:{sha256_file(Path(__file__))[:12]}"


# ---------------------------------------------------------------- records -> frame
def runs_frame(store):
    rows = []
    for r in store.all_records():
        d = r["digest"]
        try:
            cfg = store.config(d)
        except Exception:
            continue
        t = read_json(store.test_path(d)) if store.test_path(d).exists() else {}
        fm = r.get("final_metrics") or {}
        hist = r.get("history") or []
        rows.append(dict(digest=d, run_id=d[:16], status=r.get("status"), failure_kind=r.get("failure_kind"), failure_reason=r.get("failure_reason"),
                         tags=tuple(r.get("tags", [])), dataset=cfg["dataset"]["name"], arch=cfg["model"]["arch"], depth=cfg["model"].get("depth", 2),
                         method=cfg["activation"]["name"], seed=cfg["seed"], lr=cfg["optimizer"]["lr"], wd=cfg["optimizer"]["weight_decay"],
                         epochs=cfg["epochs"], epochs_done=len(hist), val_top1=fm.get("val_top1"), val_loss=fm.get("val_loss"),
                         clean_train_top1=fm.get("clean_train_top1"), test_top1=t.get("test_top1"), test_top5=t.get("test_top5"),
                         test_loss=t.get("test_loss"), trainable_params=(r.get("param_counts") or {}).get("trainable"),
                         act_trainable=(r.get("param_counts") or {}).get("activation_trainable"),
                         median_epoch_s=(r.get("times") or {}).get("median_epoch_train_s"), backbone_init=r.get("backbone_init_sha256"),
                         determinism=(r.get("determinism") or {}).get("actual"), compile=cfg["compile"], history=hist,
                         diagnostics=r.get("diagnostics") or {}))
    return pd.DataFrame(rows)


def tagged(df, tag):
    return df[df.tags.apply(lambda t: tag in t)] if len(df) else df


def validate_frame(df, tag, methods, seeds, issues):
    """Every (method, seed) planned for the arm appears exactly once; report missing, duplicate and failed runs."""
    d = tagged(df, tag)
    for m in methods:
        for s in seeds:
            n = len(d[(d.method == m) & (d.seed == s)])
            if n == 0: issues.append(f"{tag}: {m} seed {s} missing")
            elif n > 1: issues.append(f"{tag}: {m} seed {s} appears {n} times")
    for _, r in d[d.status != "completed"].iterrows():
        issues.append(f"{tag}: {r.method} seed {r.seed} status {r.status} ({r.failure_kind})")
    return d


# ---------------------------------------------------------------- statistics
def holm(p):
    p = np.asarray(p, float); m = len(p); order = np.argsort(p); adj = np.empty(m); run = 0.0
    for k, i in enumerate(order):
        run = max(run, (m - k) * p[i]); adj[i] = min(run, 1.0)
    return adj


def compare(a, b, paired, level=0.95):
    """Effect a - b in percentage points with a t confidence interval. a, b: dict seed -> value."""
    if paired:
        seeds = sorted(set(a) & set(b)); x = np.array([a[s] - b[s] for s in seeds])
        n = len(x)
        if n < 2: return dict(test="paired t", n=n, diff=float(x.mean()) if n else float("nan"), lo=float("nan"), hi=float("nan"), p=float("nan"))
        se = x.std(ddof=1) / math.sqrt(n); h = stats.t.ppf(0.5 + level / 2, n - 1) * se
        p = float(stats.ttest_rel([a[s] for s in seeds], [b[s] for s in seeds]).pvalue) if se > 0 else float("nan")
        return dict(test="paired t", n=n, diff=float(x.mean()), lo=float(x.mean() - h), hi=float(x.mean() + h), p=p)
    xa, xb = np.array(list(a.values()), float), np.array(list(b.values()), float)
    if len(xa) < 2 or len(xb) < 2:
        return dict(test="Welch", n=(len(xa), len(xb)), diff=float(xa.mean() - xb.mean()) if len(xa) and len(xb) else float("nan"),
                    lo=float("nan"), hi=float("nan"), p=float("nan"))
    va, vb = xa.var(ddof=1) / len(xa), xb.var(ddof=1) / len(xb); se = math.sqrt(va + vb)
    dfw = (va + vb) ** 2 / (va ** 2 / (len(xa) - 1) + vb ** 2 / (len(xb) - 1)) if se > 0 else 1
    h = stats.t.ppf(0.5 + level / 2, dfw) * se; dd = float(xa.mean() - xb.mean())
    p = float(stats.ttest_ind(xa, xb, equal_var=False).pvalue) if se > 0 else float("nan")
    return dict(test="Welch", n=(len(xa), len(xb)), diff=dd, lo=dd - h, hi=dd + h, p=p)


def summary(d, metric, methods):
    rows = []
    for m in methods:
        g = d[d.method == m]; ok = g[(g.status == "completed") & g[metric].notna()]
        x = ok[metric].astype(float).values; n = len(x)
        sd = x.std(ddof=1) if n > 1 else float("nan")
        h = stats.t.ppf(0.975, n - 1) * sd / math.sqrt(n) if n > 1 else float("nan")
        rows.append(dict(method=m, label=METHOD_LABEL.get(m, m), n_completed=n, n_planned=len(g), n_failed=int((g.status == "failed").sum()),
                         denominator=f"{n}/{len(g)}", mean=x.mean() if n else float("nan"), sd=sd, ci95_lo=(x.mean() - h) if n > 1 else float("nan"),
                         ci95_hi=(x.mean() + h) if n > 1 else float("nan"), lr=sorted(set(g.lr)), weight_decay=sorted(set(g.wd)),
                         run_ids=",".join(g.run_id)))
    return pd.DataFrame(rows)


def by_seed(d, method, metric):
    g = d[(d.method == method) & (d.status == "completed") & d[metric].notna()]
    return {int(s): float(v) for s, v in zip(g.seed, g[metric])}


def paired_ok(d, m1, m2, data_plan_verified):
    if not data_plan_verified: return False
    a = d[(d.method == m1) & (d.status == "completed")].set_index("seed").backbone_init
    b = d[(d.method == m2) & (d.status == "completed")].set_index("seed").backbone_init
    common = sorted(set(a.index) & set(b.index))
    return len(common) >= 2 and all(a[s] == b[s] for s in common) and len(common) == len(a) == len(b)


def strongest(d, group):
    """Strongest member of a group by mean final validation top-1 (validation only)."""
    s = d[(d.method.isin(group)) & (d.status == "completed")].groupby("method").val_top1.mean()
    return (s.idxmax(), float(s.max())) if len(s) else (None, float("nan"))


def analyse_arm(df, tag, protocol, data_plan_verified, metric="test_top1"):
    methods = protocol["methods"]; seeds = protocol["final_seeds"]; issues = []
    d = validate_frame(df, tag, methods, seeds, issues)
    if not len(d):
        return None
    summ = summary(d, metric, methods); val_summ = summary(d, "val_top1", methods)
    std_best, std_val = strongest(d, protocol["groups"]["standard"]); ad_best, ad_val = strongest(d, protocol["groups"]["adaptive"])
    comps = []
    def cmp(other, role):
        if other is None: return
        pr = paired_ok(d, "apu", other, data_plan_verified)
        c = compare(by_seed(d, "apu", metric), by_seed(d, other, metric), pr)
        comps.append(dict(role=role, comparison=f"apu - {other}", **{k: (v if not isinstance(v, tuple) else str(v)) for k, v in c.items()}))
    cmp("apu_frozen", "primary")
    cmp(std_best, "secondary: strongest validated standard")
    cmp(ad_best, "secondary: strongest validated adaptive")
    cmp("apu_const", "secondary: learned constant power")
    comps = pd.DataFrame(comps)
    if len(comps):
        sec = comps.role.str.startswith("secondary") & comps.p.notna()
        comps["p_holm"] = np.nan
        if sec.any(): comps.loc[sec, "p_holm"] = holm(comps.loc[sec, "p"].values)
        comps.loc[comps.role == "primary", "p_holm"] = comps.loc[comps.role == "primary", "p"]
    # exploratory: fixed-profile investigation (frozen vs standard baselines)
    fixed_inv = []
    for other in protocol["groups"]["standard"] + protocol["groups"]["adaptive"]:
        c = compare(by_seed(d, "apu_frozen", metric), by_seed(d, other, metric), paired_ok(d, "apu_frozen", other, data_plan_verified))
        fixed_inv.append(dict(comparison=f"apu_frozen - {other}", **{k: (v if not isinstance(v, tuple) else str(v)) for k, v in c.items()}))
    return dict(tag=tag, issues=issues, summary=summ, val_summary=val_summ, comparisons=comps, strongest_standard=std_best,
                strongest_adaptive=ad_best, fixed_profile_investigation=pd.DataFrame(fixed_inv), frame=d)


def first_passage(d, frac, budget_epochs):
    ref = d[(d.method == "relu") & (d.status == "completed")].val_top1
    if not len(ref): return None, None
    thr = frac * float(ref.mean()); rows = []
    for _, r in d.iterrows():
        ep = next((h["epoch"] for h in r.history if h.get("val_top1") is not None and h["val_top1"] >= thr), None)
        rows.append(dict(method=r.method, seed=r.seed, status=r.status, epoch=ep, censored=ep is None))
    per = pd.DataFrame(rows); out = []
    for m, g in per.groupby("method"):
        e = sorted([x if x is not None else math.inf for x in g.epoch])
        med = e[len(e) // 2] if len(e) % 2 else (e[len(e) // 2 - 1] + e[len(e) // 2]) / 2
        out.append(dict(method=m, threshold=thr, budget=budget_epochs, reached=int((~g.censored).sum()), n=len(g),
                        median_epoch=(f">{budget_epochs} (censored)" if not math.isfinite(med) else med),
                        epochs=",".join("censored" if x is None else str(int(x)) for x in g.sort_values("seed").epoch)))
    return pd.DataFrame(out), per


def generalization(d):
    rows = []
    for m, g in d[d.status == "completed"].groupby("method"):
        gap = (g.clean_train_top1 - g.val_top1).dropna()
        rows.append(dict(method=m, clean_train_top1=g.clean_train_top1.mean(), val_top1=g.val_top1.mean(), gap_mean=gap.mean(),
                         gap_sd=gap.std(ddof=1) if len(gap) > 1 else float("nan"), n=len(gap), weight_decay=sorted(set(g.wd))))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- decision (plan section 7)
def decide(res1, res2, cost_overhead_pct, alpha=0.05):
    def flags(res):
        if res is None: return None
        c = res["comparisons"].set_index("role")
        def sig_pos(role):
            if role not in c.index: return False
            r = c.loc[role]; p = r["p_holm"] if role != "primary" else r["p"]
            return bool(np.isfinite(r["lo"]) and r["lo"] > 0 and np.isfinite(p) and p < alpha)
        prim = c.loc["primary"] if "primary" in c.index else None
        similar = prim is not None and np.isfinite(prim["lo"]) and prim["lo"] <= 0 <= prim["hi"]
        return dict(beats_frozen=sig_pos("primary"), similar_to_frozen=bool(similar),
                    beats_standard=sig_pos("secondary: strongest validated standard"),
                    beats_adaptive=sig_pos("secondary: strongest validated adaptive"),
                    primary_diff=float(prim["diff"]) if prim is not None else float("nan"))
    f1, f2 = flags(res1), flags(res2)
    full = lambda f: f is not None and f["beats_frozen"] and f["beats_standard"] and f["beats_adaptive"]
    applied = []
    if full(f1) and full(f2): applied.append("A")
    if f1 and f1["similar_to_frozen"]: applied.append("B")
    if full(f1) and not full(f2): applied.append("C")
    improves = f1 is not None and (f1["beats_standard"] or f1["beats_adaptive"])
    if improves and cost_overhead_pct is not None and cost_overhead_pct > 0: applied.append("D")
    if not any(k in applied for k in ("A", "C")): applied.append("E")
    return dict(stage1=f1, stage2=f2, applied=applied, cost_overhead_pct=cost_overhead_pct,
                extensions_triggered="A" in applied)


# ---------------------------------------------------------------- figures
def _style():
    import matplotlib as mpl
    mpl.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "font.size": 8, "axes.titlesize": 8.5,
                         "axes.labelsize": 8, "legend.fontsize": 6.5, "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25, "savefig.dpi": 300, "pdf.fonttype": 42,
                         "savefig.bbox": "tight"})


def _save(fig, out, name):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.pdf"); fig.savefig(out / f"{name}.png", dpi=200)


def fig_activation_overview(out, learned=None):
    """Plots the actual module implementations at initialisation (and learned layer profiles if given)."""
    import torch, matplotlib.pyplot as plt
    from .activations import ACT_SPECS, make_activation
    _style(); z = torch.linspace(-4, 4, 1601, requires_grad=True)
    fig, ax = plt.subplots(1, 3 if learned else 2, figsize=(7.0, 2.3))
    for m in ["relu", "gelu", "prelu", "acon_c", "apu_const", "apu_frozen"]:
        spec = dict(ACT_SPECS[m]); mod = make_activation(spec, 1, 1)
        if hasattr(mod, "reset_parameters_keyed"): mod.reset_parameters_keyed(0, "a1")
        y = mod(z.view(1, 1, -1, 1)).flatten(); g, = torch.autograd.grad(y.sum(), z)
        lab = "APU at init (= frozen profile)" if m == "apu_frozen" else METHOD_LABEL[m] + (" (seed 0, channel 0)" if m == "acon_c" else "")
        ax[0].plot(z.detach(), y.detach(), color=COLORS[m], label=lab, lw=1.1); ax[1].plot(z.detach(), g, color=COLORS[m], lw=1.1)
    ax[0].set_title("f(z) at initialisation"); ax[1].set_title("f'(z) at initialisation"); ax[1].set_ylim(-0.6, 2.0)
    for a in ax[:2]: a.set_xlabel("z")
    ax[0].legend(frameon=False)
    if learned:
        for name, (zz, yy) in learned.items():
            ax[2].plot(zz, yy, lw=1.1, label=name)
        ax[2].plot(zz, np.maximum(zz, 0), "k--", lw=0.8, label="ReLU"); ax[2].set_title("Learned APU per layer (seed 0)"); ax[2].legend(frameon=False)
    _save(fig, out, "fig1_activation_overview"); return fig


def fig_accuracy(out, arms):
    """arms: list of (title, frame). Individual seeds + mean + 95% CI; failed runs annotated, never plotted as values."""
    import matplotlib.pyplot as plt
    _style(); fig, axes = plt.subplots(1, len(arms), figsize=(3.5 * len(arms), 2.6), squeeze=False)
    for ax, (title, d) in zip(axes[0], arms):
        ms = [m for m in METHOD_ORDER if m in set(d.method)]
        for i, m in enumerate(ms):
            g = d[d.method == m]; ok = g[(g.status == "completed") & g.test_top1.notna()]
            x = ok.test_top1.values
            ax.scatter(np.full(len(x), i) + np.linspace(-0.12, 0.12, max(len(x), 1))[:len(x)], x, s=10, color=COLORS.get(m, "k"),
                       marker=MARKERS.get(m, "o"), alpha=0.75, zorder=3)
            if len(x) > 1:
                h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / math.sqrt(len(x))
                ax.errorbar(i + 0.28, x.mean(), yerr=h, fmt="_", color="k", capsize=2, lw=1)
            nf = int((g.status == "failed").sum())
            if nf: ax.annotate(f"{nf} failed", (i, ax.get_ylim()[0]), fontsize=6, ha="center", color="red")
        ax.set_xticks(range(len(ms))); ax.set_xticklabels([METHOD_LABEL.get(m, m) for m in ms], rotation=35, ha="right")
        ax.set_ylabel("test top-1 (%)"); ax.set_title(title)
    _save(fig, out, "fig2_shallow_accuracy_seeds"); return fig


def fig_learned_vs_frozen(out, arms, paired_flags):
    import matplotlib.pyplot as plt
    _style(); fig, axes = plt.subplots(1, len(arms), figsize=(2.6 * len(arms), 2.5), squeeze=False)
    for ax, (title, d), paired in zip(axes[0], arms, paired_flags):
        a = by_seed(d, "apu_frozen", "test_top1"); b = by_seed(d, "apu", "test_top1"); c = by_seed(d, "apu_const", "test_top1")
        if paired:
            for s in sorted(set(a) & set(b)):
                ax.plot([0, 1], [a[s], b[s]], color="0.6", lw=0.8, marker="o", ms=3)
        else:
            ax.scatter(np.zeros(len(a)), list(a.values()), s=10, color=COLORS["apu_frozen"]); ax.scatter(np.ones(len(b)), list(b.values()), s=10, color=COLORS["apu"])
        ax.scatter(np.full(len(c), 2), list(c.values()), s=10, color=COLORS["apu_const"], marker="s")
        ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["frozen\nprofile", "learned\n(full APU)", "learned\nconst p"])
        ax.set_title(title + (" (paired by seed)" if paired else " (unpaired)")); ax.set_ylabel("test top-1 (%)")
    _save(fig, out, "fig3_learned_vs_frozen"); return fig


def fig_accuracy_vs_cost(out, acc_summary, cost, cost_col, cost_label):
    import matplotlib.pyplot as plt
    j = acc_summary.merge(cost, on="method", how="inner")
    if not len(j): return None
    _style(); fig, ax = plt.subplots(figsize=(3.4, 2.5))
    for _, r in j.iterrows():
        ax.errorbar(r[cost_col], r["mean"], yerr=(r["mean"] - r["ci95_lo"]) if np.isfinite(r["ci95_lo"]) else None,
                    xerr=[[r[cost_col] - r[cost_col + "_q25"]], [r[cost_col + "_q75"] - r[cost_col]]] if cost_col + "_q25" in r else None,
                    fmt=MARKERS.get(r.method, "o"), color=COLORS.get(r.method, "k"), label=METHOD_LABEL.get(r.method, r.method), ms=5, capsize=2)
    ax.set_xlabel(cost_label); ax.set_ylabel("test top-1 (%)"); ax.legend(frameon=False, fontsize=6)
    _save(fig, out, f"fig4_accuracy_vs_{cost_col}"); return fig


def fig_val_curves(out, d, title):
    """Measured validation curves only (no interpolation); failed runs drawn up to their last epoch."""
    import matplotlib.pyplot as plt
    _style(); fig, ax = plt.subplots(figsize=(3.4, 2.5))
    for m in [x for x in METHOD_ORDER if x in set(d.method)]:
        for _, r in d[d.method == m].iterrows():
            e = [h["epoch"] for h in r.history if "val_top1" in h]; v = [h["val_top1"] for h in r.history if "val_top1" in h]
            ax.plot(e, v, color=COLORS.get(m, "k"), lw=0.6, alpha=0.6, label=METHOD_LABEL.get(m, m) if r.seed == min(d[d.method == m].seed) else None)
    ax.set_xlabel("epoch"); ax.set_ylabel("validation top-1 (%)"); ax.set_title(title); ax.legend(frameon=False, fontsize=5.5)
    _save(fig, out, f"supp_val_curves_{title.replace(' ', '_').replace('/', '_')}"); return fig


def fig_exponents(out, d, title):
    import matplotlib.pyplot as plt
    g = d[(d.method == "apu") & (d.status == "completed")]
    if not len(g): return None
    _style(); fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.4))
    for _, r in g.iterrows():
        eps = sorted(int(e) for e in r.diagnostics)
        layers = sorted({l for e in eps for l in r.diagnostics[str(e)]["activations"]})
        for l in layers:
            A = [r.diagnostics[str(e)]["activations"][l].get("amplitude_A") for e in eps]
            ax[0].plot(eps, A, lw=0.7, alpha=0.7, label=f"{l} s{r.seed}" if r.seed == g.seed.min() else None)
        last = r.diagnostics[str(eps[-1])]["activations"]
        for i, l in enumerate(layers):
            q = last[l]["exponent"]["quantiles"]
            ax[1].plot([float(k) for k in q], list(q.values()), lw=0.7, alpha=0.7, color=f"C{i}")
    ax[0].set_xlabel("epoch"); ax[0].set_ylabel("effective amplitude A"); ax[0].legend(frameon=False, fontsize=5.5)
    ax[1].set_xlabel("quantile"); ax[1].set_ylabel("p(z) at final epoch"); ax[0].set_title(title); ax[1].set_title("exponent quantiles per layer")
    _save(fig, out, f"supp_exponents_{title.replace(' ', '_').replace('/', '_')}"); return fig


def table_to_tex(df, path, caption):
    try:
        tex = df.to_latex(index=False, float_format=lambda x: f"{x:.2f}", escape=True)
    except Exception:
        tex = df.to_string()
    atomic_write_text(path, f"% generated by {script_id()} on {now()}\n% {caption}\n" + tex)
