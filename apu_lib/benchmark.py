"""Cost measurement in a fresh process (plan section 5).

Usage: python -m apu_lib.benchmark --spec spec.json --out result.json

One process measures one (architecture, activation, execution mode) configuration, so retained tensors and compiled
graphs of other models cannot contaminate memory or timing. The accelerator is synchronised before and after every
timed block; each scope has warm-up iterations and several independent timing blocks, and medians plus the
interquartile range across blocks are reported. Block-to-block spread estimates measurement variation only, not
training-seed variation. Peak memory is reset before each scope and reported with that scope's name.
Compilation failures are reported as status 'failed' with failure_kind 'compile'; there is no silent eager fallback.
"""
import argparse, json, os, sys, time, traceback, contextlib
from pathlib import Path


def _q(xs):
    import numpy as np
    xs = np.asarray(xs, float)
    return dict(median=float(np.median(xs)), q25=float(np.percentile(xs, 25)), q75=float(np.percentile(xs, 75)),
                min=float(xs.min()), max=float(xs.max()), blocks=len(xs))


def run(spec):
    import torch, torch.nn as nn
    from apu_lib.config import environment_record, read_json
    from apu_lib.activations import ACT_SPECS, count_params
    from apu_lib.models import create_model
    from apu_lib.data import DS_META, load_arrays, load_split, KeyedLoader
    from apu_lib.evaluate import autocast_ctx
    from apu_lib.train import make_optimizer, set_determinism

    device = torch.device(spec["device"]); precision = spec["precision"]; cuda = device.type == "cuda"
    set_determinism(spec.get("deterministic", True))
    meta = DS_META[spec["dataset"]]
    act_spec = dict(ACT_SPECS[spec["act"]]); act_spec["name"] = spec["act"]
    model = create_model(spec["model"], act_spec, meta, 0).to(device)
    cnn = spec["model"]["arch"] in ("simplecnn", "resnet18")
    if cuda and cnn: model = model.to(memory_format=torch.channels_last)
    out = dict(spec=spec, env=environment_record(), params=count_params(model), status="running")
    def sync():
        if cuda: torch.cuda.synchronize()
    def reset_mem():
        if cuda: sync(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    def peak():
        return torch.cuda.max_memory_allocated() / 2**20 if cuda else None
    W, K, M = spec["warmup"], spec["blocks"], spec["iters_per_block"]
    f = model
    if spec["mode"] == "compiled":
        try:
            import torch._dynamo as dyn; dyn.reset()
            f = torch.compile(model)
        except Exception as e:
            out.update(status="failed", failure_kind="compile", failure_reason=repr(e)[:500]); return out
    def timed(fn, first_label):
        t = time.time(); fn(); sync(); first = time.time() - t
        for _ in range(W): fn()
        sync(); blocks = []
        for _ in range(K):
            sync(); t = time.perf_counter()
            for _ in range(M): fn()
            sync(); blocks.append((time.perf_counter() - t) * 1000 / M)
        return dict(first_call_s=first, ms=_q(blocks))
    B = spec["batch"]
    try:
        res = {}
        for bs in sorted({1, B}):
            x = torch.randn(bs, meta["in_ch"], meta["img"], meta["img"], device=device)
            if cuda and cnn: x = x.contiguous(memory_format=torch.channels_last)
            model.eval(); reset_mem()
            def inf():
                with torch.no_grad(), autocast_ctx(device, precision): f(x)
            r = timed(inf, "inference"); r["peak_mem_mb"] = peak(); r["peak_mem_scope"] = f"inference batch {bs}, no_grad, eval mode"
            res[f"inference_bs{bs}"] = r
        model.train()
        opt, _ = make_optimizer(model, spec["optimizer"]); crit = nn.CrossEntropyLoss()
        x = torch.randn(B, meta["in_ch"], meta["img"], meta["img"], device=device)
        if cuda and cnn: x = x.contiguous(memory_format=torch.channels_last)
        y = torch.randint(0, meta["num_classes"], (B,), device=device)
        reset_mem()
        def step():
            opt.zero_grad(set_to_none=True)
            with autocast_ctx(device, precision): loss = crit(f(x), y)
            loss.backward(); opt.step()
        r = timed(step, "train_step"); r["peak_mem_mb"] = peak(); r["peak_mem_scope"] = f"full training step batch {B} with {spec['optimizer']['name']}"
        res["train_step"] = r
        if spec.get("epoch_reps", 0) > 0:
            xs, ys = load_arrays(spec["dataset"], spec["data_dir"], "train")
            split = read_json(spec["split_json"]); tr, _ = load_split(Path(spec["split_json"]).parent, split)
            if spec.get("train_limit"): tr = tr[: spec["train_limit"]]
            tl = KeyedLoader(xs, ys, tr, B, device, meta["mean"], meta["std"], train=True, augment=True, seed=0)
            reset_mem(); ep_times = []
            for e in range(spec["epoch_reps"] + 1):              # first epoch is warm-up
                tl.set_epoch(e); sync(); t = time.perf_counter()
                for xb, yb, _ in tl:
                    if cuda and cnn: xb = xb.contiguous(memory_format=torch.channels_last)
                    opt.zero_grad(set_to_none=True)
                    with autocast_ctx(device, precision): loss = crit(f(xb), yb)
                    loss.backward(); opt.step()
                sync()
                if e > 0: ep_times.append(time.perf_counter() - t)
            res["epoch_end_to_end"] = dict(s=_q(ep_times), n_train=len(tr), peak_mem_mb=peak(),
                                           peak_mem_scope="training epochs incl. GPU data pipeline and augmentation")
        out.update(status="completed", results=res, compile_outcome=("compiled" if spec["mode"] == "compiled" else "eager"))
    except Exception as e:
        kind = "compile" if spec["mode"] == "compiled" and ("dynamo" in repr(e).lower() or "inductor" in repr(e).lower() or "triton" in repr(e).lower()) else "benchmark"
        out.update(status="failed", failure_kind=kind, failure_reason=repr(e)[:500], traceback=traceback.format_exc()[-3000:])
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--spec"); ap.add_argument("--out"); a = ap.parse_args()
    spec = json.load(open(a.spec))
    try:
        res = run(spec)
    except Exception as e:
        res = dict(spec=spec, status="failed", failure_kind="benchmark", failure_reason=repr(e)[:500], traceback=traceback.format_exc()[-3000:])
    from apu_lib.config import atomic_write_json
    atomic_write_json(a.out, res)
    print("BENCH_STATUS", res["status"])
