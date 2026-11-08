"""Engineering and correctness checks run before any long training (plan sections 1-3).
Each check returns dict(name, passed, details). `run_all` writes engineering/checks.json."""
import io, math, copy, shutil, traceback
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

from .config import digest_of, Store, DigestMismatchError, load_foreign_result, atomic_write_json, now, read_json
from .activations import (APU, apu_reference, ACT_SPECS, make_activation, count_params, ACONC, PadeActivation, APU_V1,
                          activation_param_names)
from .models import create_model, backbone_digest, backbone_state, build_model
from .data import DS_META, KeyedLoader
from .evaluate import probe_activations, evaluate


def _res(name, passed, **details):
    return dict(name=name, passed=bool(passed), details=details)


# ---------------------------------------------------------------- activation mathematics
def _apu(mode="full", **kw):
    args = dict(APU_V1); args.update(kw); args["mode"] = mode
    return APU(**{k: args[k] for k in ("mode", "p0", "a_max", "init_A", "init_wz", "init_bz", "p_min", "p_max", "eps")})


def _ref_args(m, dtype):
    t = lambda name: getattr(m, name).detach().to(dtype) if hasattr(m, name) else None
    return dict(mode=m.mode, p0=m.p0, a_max=m.a_max, eps=m.eps, p_min=m.p_min, p_max=m.p_max, raw_A=t("raw_A"), w_z=t("w_z"),
                b_z=t("b_z"), a=t("a"), w_p=t("w_p"))


def check_activation_math(device, measured_range=None):
    out = []
    z = torch.cat([torch.linspace(-6, 6, 2001), torch.tensor([0.0, 1e-6, -1e-6, 1e-3, -1e-3, 50.0, -50.0])])
    modes = ["fixed", "const", "linear", "sigmoid", "full"]
    # forward values: production (float32) vs float64 reference
    errs = {}
    for mode in modes:
        m = _apu(mode, p0=(0.9 if mode == "fixed" else 1.0))
        y = m(z).double(); r = apu_reference(z.double(), **_ref_args(m, torch.float64))
        errs[mode] = float(((y.detach() - r).abs() / (r.abs() + 1e-6)).max())
    out.append(_res("APU forward matches float64 reference (relative error < 1e-5)", all(e < 1e-5 for e in errs.values()), rel_err=errs))
    # identity cases
    m = _apu("full"); 
    with torch.no_grad(): m.raw_A.zero_()
    zz = torch.randn(10000) * 5
    out.append(_res("identity at A = 0, p0 = 1 (exact)", torch.equal(m(zz), zz)))
    mc = APU(mode="const", p0=1.0)
    out.append(_res("learned constant power initialised at p = 1 is the identity (exact)", torch.equal(mc(zz), zz),
                    note="the const control starts as the identity and cannot match the non-constant initial full profile"))
    full0 = _apu("full")
    out.append(_res("full APU initial profile is non-constant in z", float(full0.exponent(zz).detach().std()) > 1e-3,
                    p_range=[float(full0.exponent(zz).detach().min()), float(full0.exponent(zz).detach().max())]))
    # sign preservation
    sp = {}
    for mode in modes:
        y = _apu(mode, p0=(0.9 if mode == "fixed" else 1.0))(zz)
        sp[mode] = bool(torch.equal(torch.sign(y), torch.sign(zz)))
    out.append(_res("sign preservation sign(f(z)) = sign(z)", all(sp.values()), per_mode=sp))
    # broadcasting: per-channel parameters along dim 1 (conv) and dim -1 (tokens) vs elementwise loop
    pc = APU(mode="full", num_channels=4, channel_dim=1)
    with torch.no_grad(): pc.w_z.copy_(torch.tensor([0.1, 0.5, 1.0, 2.0])); pc.b_z.copy_(torch.tensor([0.0, 0.3, -0.3, 1.0]))
    x = torch.randn(2, 4, 3, 3); y = pc(x); ok = True
    for c in range(4):
        sc = APU(mode="full")
        with torch.no_grad(): sc.raw_A.copy_(pc.raw_A[c:c + 1]); sc.w_z.copy_(pc.w_z[c:c + 1]); sc.b_z.copy_(pc.b_z[c:c + 1])
        ok &= torch.allclose(y[:, c], sc(x[:, c]), atol=1e-6)
    pt = APU(mode="full", num_channels=4, channel_dim=-1)
    pt.load_state_dict(pc.state_dict()); xt = x.permute(0, 2, 3, 1); ok &= torch.allclose(pt(xt), y.permute(0, 2, 3, 1), atol=1e-6)
    out.append(_res("broadcasting of per-channel parameters (dim 1 and dim -1)", ok))
    # serialization round trip
    buf = io.BytesIO(); torch.save(pc.state_dict(), buf); buf.seek(0)
    pc2 = APU(mode="full", num_channels=4, channel_dim=1); pc2.load_state_dict(torch.load(buf))
    out.append(_res("state_dict serialization round trip", torch.equal(pc2(x), y)))
    # device transfer
    if device.type == "cuda":
        g = copy.deepcopy(pc).to(device); yd = g(x.to(device)).cpu()
        out.append(_res("CPU and GPU outputs agree (atol 1e-5)", torch.allclose(yd.detach(), y.detach(), atol=1e-5), max_abs=float((yd - y).detach().abs().max())))
    else:
        out.append(_res("CPU and GPU outputs agree", True, note="no GPU in this session; not tested"))
    # float64 gradient check of the dtype-preserving reference (positive, negative, near-zero inputs; parameter limits)
    gc = {}
    settings = {"defaults": dict(), "A near a_max (raw_A=4)": dict(raw_A=4.0), "large w_z (8)": dict(w_z=8.0), "large b_z (5)": dict(b_z=5.0),
                "small eps (1e-6)": dict(eps=1e-6), "negative A (raw_A=-3)": dict(raw_A=-3.0)}
    zin = torch.tensor([-3.0, -0.7, -1e-3, -1e-5, 0.0, 1e-5, 1e-3, 0.7, 3.0], dtype=torch.float64)
    for label, s in settings.items():
        raw = torch.tensor([s.get("raw_A", math.atanh(0.3 / 0.5))], dtype=torch.float64, requires_grad=True)
        w = torch.tensor([s.get("w_z", 0.5)], dtype=torch.float64, requires_grad=True)
        b = torch.tensor([s.get("b_z", 0.0)], dtype=torch.float64, requires_grad=True)
        zi = zin.clone().requires_grad_(True); eps = s.get("eps", 1e-3)
        f = lambda zz_, r_, w_, b_: apu_reference(zz_, mode="full", eps=eps, raw_A=r_, w_z=w_, b_z=b_)
        try:
            gc[label] = bool(torch.autograd.gradcheck(f, (zi, raw, w, b), eps=1e-7, atol=1e-5, rtol=1e-4))
        except Exception as e:
            gc[label] = f"failed: {repr(e)[:150]}"
    out.append(_res("float64 finite-difference gradient check of the reference (incl. near-zero inputs and parameter limits)",
                    all(v is True for v in gc.values()), per_setting=gc))
    # production (float32) gradients vs reference (float64) gradients
    m = _apu("full"); zp = torch.tensor([-3.0, -0.7, -1e-3, 0.0, 1e-3, 0.7, 3.0], requires_grad=True)
    y = m(zp).sum(); gz, gw = torch.autograd.grad(y, [zp, m.w_z])
    zr = zp.detach().double().requires_grad_(True); a = _ref_args(m, torch.float64); a["w_z"] = a["w_z"].clone().requires_grad_(True)
    yr = apu_reference(zr, **a).sum(); rz, rw = torch.autograd.grad(yr, [zr, a["w_z"]])
    e1 = float(((gz.double() - rz).abs() / (rz.abs() + 1e-4)).max()); e2 = float(((gw.double() - rw).abs() / (rw.abs() + 1e-4)).max())
    out.append(_res("production float32 autograd agrees with float64 reference autograd (rel < 1e-3)", e1 < 1e-3 and e2 < 1e-3,
                    grad_z_rel_err=e1, grad_wz_rel_err=e2))
    # finite outputs and gradients over the stress range and the measured range
    ranges = {"stress [-1e4, 1e4]": (-1e4, 1e4)}
    if measured_range:
        ranges["measured " + measured_range.get("source", "")[:40]] = (measured_range["min"], measured_range["max"])
    fin = {}
    for label, (lo, hi) in ranges.items():
        zz_ = torch.linspace(lo, hi, 200001, device=device).requires_grad_(True)
        for mode in modes:
            for lim, kw in {"default": {}, "raw_A=4": dict(init_A=0.499)}.items():
                if lim != "default" and mode != "full": continue
                mm = _apu(mode, p0=(0.9 if mode == "fixed" else 1.0), **kw).to(device)
                yy = mm(zz_); g, = torch.autograd.grad(yy.sum(), zz_)
                fin[f"{label} | {mode} | {lim}"] = bool(torch.isfinite(yy).all() and torch.isfinite(g).all())
    out.append(_res("finite outputs and gradients over stress and measured pre-activation ranges", all(fin.values()), per_case=fin,
                    measured_range=measured_range or "unavailable (no historical exponent cache found)"))
    # monotonicity is not guaranteed: report the minimum derivative of the default profile over [-6, 6]
    zz_ = torch.linspace(-6, 6, 12001, requires_grad=True); g, = torch.autograd.grad(_apu("full")(zz_).sum(), zz_)
    out.append(_res("report: minimum derivative of the default APU on [-6, 6] (monotonicity is not guaranteed in general)", True,
                    min_derivative=float(g.min()), max_derivative=float(g.max())))
    # ACON-C identities
    ac = ACONC(4)
    with torch.no_grad(): ac.p1.fill_(1.0); ac.p2.fill_(0.0); ac.beta.fill_(1.0)
    x = torch.randn(3, 4, 5, 5)
    ok1 = torch.allclose(ac(x), torch.nn.functional.silu(x), atol=1e-6)
    with torch.no_grad(): ac.p2.fill_(0.01); ac.beta.fill_(1e4)
    ok2 = torch.allclose(ac(x), torch.maximum(x, 0.01 * x), atol=1e-3)
    out.append(_res("ACON-C identities (SiLU at p1=1,p2=0,beta=1; max(p1 x, p2 x) for large beta)", ok1 and ok2, silu=ok1, max_limit=ok2,
                    note="re-implementation; verify against the authors' code before submission"))
    # Pade
    pa = PadeActivation(); xx = torch.linspace(-3, 3, 2001)
    err = float((pa(xx) - torch.nn.functional.leaky_relu(xx, 0.01)).abs().max())
    xs = torch.linspace(-1e3, 1e3, 20001, requires_grad=True); g, = torch.autograd.grad(pa(xs).sum(), xs)
    out.append(_res("safe Pade activation: initial fit to leaky ReLU on [-3, 3] (max error < 0.1) and finite on [-1e3, 1e3]",
                    err < 0.1 and bool(torch.isfinite(g).all()), max_fit_error=err))
    return out


# ---------------------------------------------------------------- parameter counts and the frozen control
HISTORICAL_FULL_COUNTS = {("cifar10", "resnet18", "relu"): 11173962, ("cifar10", "resnet18", "apu"): 11174013,
                          ("cifar100", "resnet18", "relu"): 11220132, ("cifar100", "resnet18", "apu"): 11220183}


def check_param_counts(historical_df=None):
    out = []; meta10 = DS_META["cifar10"]; meta100 = DS_META["cifar100"]
    exp = {("simplecnn", "apu"): 9, ("resnet18", "apu"): 51, ("vit_tiny", "apu"): 36}
    got = {}
    for (arch, act), n in exp.items():
        m = create_model(dict(arch=arch, depth=2) if arch == "simplecnn" else dict(arch=arch), ACT_SPECS[act], meta10, 0)
        got[f"{arch}/{act}"] = count_params(m)["activation_trainable"]; del m
    out.append(_res("extra learned activation parameters: 9 (SimpleCNN), 51 (ResNet-18), 36 (ViT-Tiny)",
                    got == {f"{a}/{b}": n for (a, b), n in exp.items()}, counts=got))
    fc = {}
    for (ds, arch, act), n in HISTORICAL_FULL_COUNTS.items():
        m = create_model(dict(arch=arch), ACT_SPECS[act], DS_META[ds], 0); fc[f"{ds}/{arch}/{act}"] = (count_params(m)["total"], n); del m
    hist_cmp = {}
    if historical_df is not None and len(historical_df):
        for (ds, arch, act), g in historical_df[historical_df.params.notna()].groupby(["dataset", "arch", "act"]):
            if ds not in ("cifar10", "cifar100") or act not in ACT_SPECS or arch not in ("simplecnn", "resnet18", "vit_tiny"): continue
            m = create_model(dict(arch=arch, depth=2) if arch == "simplecnn" else dict(arch=arch), ACT_SPECS[act], DS_META[ds], 0)
            hist_cmp[f"{ds}/{arch}/{act}"] = (count_params(m)["trainable"], int(g.params.iloc[0])); del m
    allc = {**fc, **hist_cmp}
    out.append(_res("full-model parameter counts match the historical study", all(a == b for a, b in allc.values()),
                    counts={k: dict(now=a, historical=b) for k, (a, b) in allc.items()},
                    note="historical JSON counts compared where the historical folder is available"))
    per = {}
    for act in ["relu", "gelu", "prelu", "apu_frozen", "apu_const", "apu", "acon_c", "pade", "apu_pc"]:
        per[act] = count_params(create_model(dict(arch="simplecnn", depth=2), ACT_SPECS[act], meta10, 0))
    out.append(_res("SimpleCNN parameter counts per method (frozen control adds 0 trainable)", per["apu_frozen"]["activation_trainable"] == 0
                    and per["apu_frozen"]["trainable"] == per["relu"]["trainable"], per_method=per))
    return out


def check_frozen_control(device):
    meta = DS_META["cifar10"]; out = []
    fz = create_model(dict(arch="simplecnn", depth=2), ACT_SPECS["apu_frozen"], meta, 0).to(device)
    full = create_model(dict(arch="simplecnn", depth=2), ACT_SPECS["apu"], meta, 0).to(device)
    x = torch.randn(16, 3, 32, 32, generator=torch.Generator().manual_seed(1)).to(device)
    with torch.no_grad():
        same = torch.equal(fz(x), full(x))
    out.append(_res("frozen control has exactly the initial function of the full APU (same backbone init, same output)", same and
                    backbone_digest(fz) == backbone_digest(full)))
    from .train import make_optimizer
    opt, names = make_optimizer(fz, dict(name="adamw", lr=1e-2, betas=[0.9, 0.999], weight_decay=1e-2))
    act_names = activation_param_names(fz)
    before = {n: p.detach().clone() for n, p in fz.named_parameters()}
    y = torch.randint(0, 10, (16,), generator=torch.Generator().manual_seed(2)).to(device)
    for _ in range(3):
        opt.zero_grad(); nn.functional.cross_entropy(fz(x), y).backward(); opt.step()
    act_same = all(torch.equal(before[n], p) for n, p in fz.named_parameters() if n in act_names)
    bb_changed = all(not torch.equal(before[n], p) for n, p in fz.named_parameters() if n not in act_names)
    in_opt = {id(p) for g in opt.param_groups for p in g["params"]}
    no_act_in_opt = all(id(p) not in in_opt for n, p in fz.named_parameters() if n in act_names)
    out.append(_res("frozen activation parameters receive no updates; every backbone parameter is updated", act_same and bb_changed and no_act_in_opt,
                    activation_unchanged=act_same, backbone_changed=bb_changed, activation_params_not_in_optimizer=no_act_in_opt))
    return out


# ---------------------------------------------------------------- digests and alignment
def check_digests(tmp_root, cfg):
    out = []; d0 = digest_of(cfg)
    c1 = copy.deepcopy(cfg); c1["activation"]["init_wz"] = 0.25
    c2 = copy.deepcopy(cfg); c2["split"]["val_indices_sha256"] = "0" * 64; c2["split"]["split_id"] = "x"
    c3 = copy.deepcopy(cfg)
    out.append(_res("changing an activation argument or the split changes the digest; identical config gives the same digest",
                    digest_of(c1) != d0 and digest_of(c2) != d0 and digest_of(c3) == d0))
    st = Store(Path(tmp_root) / "digest_check"); d = st.register(cfg)
    p = st.config_path(d); stored = read_json(p); stored["config"]["seed"] = 999; atomic_write_json(p, stored)
    raised = False
    try: st.load_cached(cfg)
    except DigestMismatchError: raised = True
    old = Path(tmp_root) / "digest_check" / "historical_style.json"; atomic_write_json(old, dict(run_id="cifar10-simplecnn-apu-s0-e200", final={}))
    raised2 = False
    try: load_foreign_result(st, old, cfg)
    except DigestMismatchError: raised2 = True
    out.append(_res("a stored config that no longer matches, or a historical result file, fails visibly when loaded", raised and raised2,
                    tampered_config_raises=raised, historical_format_raises=raised2))
    shutil.rmtree(Path(tmp_root) / "digest_check", ignore_errors=True)
    return out


def check_alignment(device, data):
    """data: (x, y, train_idx). Backbone init and data plans must not depend on the activation."""
    meta = DS_META["cifar10"]; out = []
    dg = {m: backbone_digest(create_model(dict(arch="simplecnn", depth=2), ACT_SPECS[m], meta, 3))
          for m in ["relu", "gelu", "prelu", "apu", "apu_frozen", "apu_const", "acon_c", "pade"]}
    other_seed = backbone_digest(create_model(dict(arch="simplecnn", depth=2), ACT_SPECS["relu"], meta, 4))
    ok_init = len(set(dg.values())) == 1 and other_seed != dg["relu"]
    rd = {m: backbone_digest(create_model(dict(arch="resnet18"), ACT_SPECS[m], meta, 3)) for m in ["relu", "apu", "acon_c"]}
    ok_init &= len(set(rd.values())) == 1
    # global RNG consumption before construction must not matter
    torch.manual_seed(12345); torch.rand(1000)
    ok_init &= backbone_digest(create_model(dict(arch="simplecnn", depth=2), ACT_SPECS["apu"], meta, 3)) == dg["relu"]
    out.append(_res("backbone initialisation is identical across activations for a seed and differs across seeds", ok_init))
    x, y, tr = data
    l1 = KeyedLoader(x, y, tr[:4096], 128, device, meta["mean"], meta["std"], True, True, seed=3)
    l2 = KeyedLoader(x, y, tr[:4096], 128, device, meta["mean"], meta["std"], True, True, seed=3)
    l3 = KeyedLoader(x, y, tr[:4096], 128, device, meta["mean"], meta["std"], True, True, seed=4)
    ok = l1.plan_digest(0) == l2.plan_digest(0) and l1.plan_digest(5) == l2.plan_digest(5) and l1.plan_digest(0) != l3.plan_digest(0) \
         and l1.plan_digest(0) != l1.plan_digest(1)
    l1.set_epoch(0); l2.set_epoch(0)
    b1 = next(iter(l1)); torch.manual_seed(999); torch.rand(10); b2 = next(iter(l2))
    ok &= torch.equal(b1[0], b2[0]) and torch.equal(b1[2], b2[2])
    out.append(_res("sample order and augmentation are keyed by (seed, epoch): identical across methods, independent of the global RNG",
                    ok, note="the loader never sees the activation, so equal keys give equal batches for every method"))
    return out, bool(ok and ok_init)


def check_bn_and_modes(device):
    meta = DS_META["cifar10"]; m = create_model(dict(arch="resnet18"), ACT_SPECS["apu"], meta, 0).to(device)
    x = torch.randn(64, 3, 32, 32, generator=torch.Generator().manual_seed(0)).to(device)
    m.train()
    for _ in range(2): m(x)                                     # non-trivial running statistics
    y = torch.zeros(64, dtype=torch.long, device=device)
    loader = [(x, y, torch.arange(64, device=device))]
    before = {k: v.clone() for k, v in m.state_dict().items() if "running" in k}
    stats, bn_ok, mode_ok = probe_activations(m, loader, device, "fp32")
    evaluate(m, loader, device, "fp32")
    after = {k: v for k, v in m.state_dict().items() if "running" in k}
    same = all(torch.equal(before[k], after[k]) for k in before)
    return [_res("probes and evaluation leave BatchNorm running statistics unchanged and restore train mode",
                 bn_ok and mode_ok and same and m.training, probe_reported_unchanged=bn_ok, mode_restored=mode_ok, evaluate_unchanged=same)]


# ---------------------------------------------------------------- short training checks
def _final_state(store, d):
    r = store.record(d); return torch.load(store.root / r["evaluation_checkpoint"]["path"], map_location="cpu", weights_only=False)["model"], r


def _max_diff(sa, sb):
    return max(float((sa[k].float() - sb[k].float()).abs().max()) for k in sa)


def check_resume_and_passive(tmp_root, base_cfg, device, data_dir, split_dir):
    from .train import train_run
    out = []; root = Path(tmp_root)
    for sub in ("resume_A", "resume_B", "diag_off"):
        shutil.rmtree(root / sub, ignore_errors=True)
    cfg = copy.deepcopy(base_cfg)
    A = Store(root / "resume_A"); B = Store(root / "resume_B")
    ra = train_run(A, cfg, device, data_dir, split_dir, verbose=False, strict=True)
    rb1 = train_run(B, cfg, device, data_dir, split_dir, stop_after_epoch=cfg["epochs"] - 1, verbose=False, strict=True)
    rb = train_run(B, cfg, device, data_dir, split_dir, verbose=False, strict=True)
    d = digest_of(cfg)
    sa, _ = _final_state(A, d); sb, _ = _final_state(B, d)
    exact = all(torch.equal(sa[k], sb[k]) for k in sa); md = _max_diff(sa, sb)
    ha = [h["val_top1"] for h in ra["history"]]; hb = [h["val_top1"] for h in rb["history"]]
    out.append(_res("epoch-boundary resume reproduces the uninterrupted run exactly (deterministic configuration)",
                    rb1["status"] == "interrupted" and rb["status"] == "completed" and rb.get("resumed_from_epoch") == cfg["epochs"] - 1 and exact and ha == hb,
                    interrupted_status=rb1["status"], resumed_from_epoch=rb.get("resumed_from_epoch"), bitwise_equal=exact, max_abs_param_diff=md,
                    val_history_uninterrupted=ha, val_history_resumed=hb, determinism=rb.get("determinism"),
                    supported_mode="epoch-boundary resumption only; mid-epoch resumption is not supported"))
    c_off = copy.deepcopy(cfg); c_off["diagnostics"]["epochs"] = []
    C = Store(root / "diag_off"); rc = train_run(C, c_off, device, data_dir, split_dir, verbose=False, strict=True)
    sc, _ = _final_state(C, digest_of(c_off)); exact2 = all(torch.equal(sa[k], sc[k]) for k in sa)
    hc = [h["val_top1"] for h in rc["history"]]
    out.append(_res("diagnostics are passive: training with probes on and off gives the same trajectory", exact2 and ha == hc,
                    bitwise_equal=exact2, max_abs_param_diff=_max_diff(sa, sc), diag_epochs_on=cfg["diagnostics"]["epochs"]))
    diag_ok = all(v.get("bn_unchanged_by_probe") for v in ra.get("diagnostics", {}).values())
    out.append(_res("no validation example entered an optimisation batch; test partition never loaded during training",
                    ra["status"] == "completed" and ra.get("test_partition_loaded") is False, samples_seen=ra.get("samples_seen"),
                    note="every step checks batch indices against the validation index set; a leak would have failed the run"))
    out.append(_res("probes during training left BatchNorm statistics unchanged", diag_ok))
    return out


def run_all(path, sections):
    res = []
    for name, fn in sections:
        try:
            r = fn()
            res.extend(r if isinstance(r, list) else [r])
        except Exception as e:
            res.append(_res(f"{name} (raised)", False, error=repr(e)[:300], traceback=traceback.format_exc()[-2000:]))
    rep = dict(created=now(), all_passed=all(r["passed"] for r in res), results=res)
    atomic_write_json(path, rep)
    return rep
