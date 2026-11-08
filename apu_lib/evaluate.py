"""Evaluation, passive diagnostics and the gated one-time test evaluation (plan sections 2 and 5).

Every function here runs the model in evaluation mode without gradient tracking and restores the previous mode.
Diagnostics never draw from a training random stream (probe batches come from a fixed validation subset), and
BatchNorm running statistics are checked to be unchanged by probes.
"""
import contextlib, math, time
import numpy as np
import torch
import torch.nn.functional as F

from .activations import act_modules, APU
from .config import ProtocolError, atomic_write_json, read_json, sha256_file, now, digest_of

QUANTILES = (0.001, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 0.999)


@contextlib.contextmanager
def eval_mode(model):
    was = model.training
    model.eval()
    try:
        yield
    finally:
        model.train(was)


def autocast_ctx(device, precision):
    if precision == "bf16-autocast" and device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


@torch.no_grad()
def evaluate(model, loader, device, precision):
    with eval_mode(model):
        n = 0; loss = 0.0; c1 = 0; c5 = 0
        for x, y, _ in loader:
            with autocast_ctx(device, precision):
                out = model(x)
            out = out.float()
            loss += F.cross_entropy(out, y, reduction="sum").item()
            top = out.topk(min(5, out.shape[1]), 1).indices
            c1 += (top[:, 0] == y).sum().item(); c5 += (top == y[:, None]).any(1).sum().item(); n += y.numel()
    return dict(loss=loss / n, top1=100.0 * c1 / n, top5=100.0 * c5 / n, n=n)


def _bn_state(model):
    return {k: v.detach().clone() for k, v in model.state_dict().items() if "running_" in k or "num_batches_tracked" in k}


def _quant(t, max_n=200_000):
    t = t.detach().float().flatten()
    finite = torch.isfinite(t); nonfinite = int((~finite).sum().item()); t = t[finite]
    if t.numel() == 0:
        return dict(nonfinite=nonfinite, n=0)
    if t.numel() > max_n:                                   # deterministic stride subsample (no random stream)
        t = t[:: math.ceil(t.numel() / max_n)]
    q = torch.quantile(t, torch.tensor(QUANTILES, device=t.device)).tolist()
    return dict(n=int(t.numel()), nonfinite=nonfinite, min=t.min().item(), max=t.max().item(), mean=t.mean().item(),
                std=t.std().item() if t.numel() > 1 else 0.0, quantiles=dict(zip([str(q_) for q_ in QUANTILES], q)))


@torch.no_grad()
def probe_activations(model, probe_loader, device, precision, max_batches=None):
    """Per activation module: pre-activation and output quantiles, non-finite counts, exponent distribution and
    effective amplitude for APU modules. Returns (stats, bn_unchanged, mode_restored)."""
    bn_before = _bn_state(model); was_training = model.training
    store = {}
    def hook(name):
        def f(m, inp, out):
            z = inp[0].detach(); o = out.detach()
            s = store.setdefault(name, dict(z=[], o=[], p=[]))
            s["z"].append(z.float().flatten()); s["o"].append(o.float().flatten())
            if isinstance(m, APU):
                s["p"].append(m.exponent(z.float()).flatten())
        return f
    handles = [m.register_forward_hook(hook(n)) for n, m in act_modules(model)]
    try:
        with eval_mode(model):
            for b, (x, _, _) in enumerate(probe_loader):
                if max_batches is not None and b >= max_batches: break
                with autocast_ctx(device, precision):
                    model(x)
    finally:
        for h in handles: h.remove()
    out = {}
    for n, m in act_modules(model):
        s = store.get(n)
        if not s: continue
        rec = dict(preact=_quant(torch.cat(s["z"])), output=_quant(torch.cat(s["o"])))
        if s["p"]:
            rec["exponent"] = _quant(torch.cat(s["p"]))
            if m.mode == "full":
                rec["amplitude_A"] = m.amplitude().float().mean().item()
            rec["params"] = {pn: p.detach().float().cpu().flatten().tolist()[:8] for pn, p in m.named_parameters(recurse=False)}
        out[n] = rec
    bn_after = _bn_state(model)
    bn_unchanged = all(torch.equal(bn_before[k], bn_after[k]) for k in bn_before)
    return out, bn_unchanged, model.training == was_training


class GradNormMeter:
    """Accumulates per-parameter gradient norms over the steps of one epoch (read-only on gradients)."""
    def __init__(self, model):
        self.names = [n for n, p in model.named_parameters() if p.requires_grad]
        self.params = [p for n, p in model.named_parameters() if p.requires_grad]
        self.sum = None; self.max = None; self.steps = 0
    @torch.no_grad()
    def update(self):
        gs = [p.grad if p.grad is not None else torch.zeros_like(p) for p in self.params]
        norms = torch.stack(torch._foreach_norm(gs)).float()
        self.sum = norms if self.sum is None else self.sum + norms
        self.max = norms if self.max is None else torch.maximum(self.max, norms)
        self.steps += 1
    def result(self):
        if not self.steps: return {}
        mean = (self.sum / self.steps).tolist(); mx = self.max.tolist()
        return {n: dict(mean=a, max=b) for n, a, b in zip(self.names, mean, mx)}


# ---------------------------------------------------------------- gated test evaluation
def evaluate_test(store, digest, lock, build_fn, test_loader_fn):
    """Evaluate the recorded evaluation checkpoint of a completed run on the official test partition exactly once.
    Refuses unless a protocol lock exists and lists this digest. The result is stored next to the run record."""
    if lock is None:
        raise ProtocolError("no protocol lock: test evaluation is not allowed yet")
    if digest not in lock["final_fit_digests"]:
        raise ProtocolError(f"run {digest[:16]} is not a locked final fit; test evaluation refused")
    tp = store.test_path(digest)
    if tp.exists():
        t = read_json(tp)
        if t.get("digest") != digest:
            raise ProtocolError("stored test metrics belong to another digest")
        return t
    r = store.record(digest)
    if not r or r.get("status") != "completed":
        raise ProtocolError(f"run {digest[:16]} is not completed; cannot evaluate on test")
    ck = r["evaluation_checkpoint"]; path = store.root / ck["path"]
    if sha256_file(path) != ck["sha256"]:
        raise ProtocolError(f"checkpoint checksum mismatch for {digest[:16]}")
    cfg = store.config(digest)
    model, device, precision = build_fn(cfg)
    state = torch.load(path, map_location=device, weights_only=False)
    if state.get("digest") != digest:
        raise ProtocolError("checkpoint belongs to another configuration")
    model.load_state_dict(state["model"])
    t0 = time.time(); m = evaluate(model, test_loader_fn(cfg), device, precision)
    res = dict(digest=digest, lock_digest=lock["lock_digest"], evaluated=now(), checkpoint=ck, test_top1=m["top1"],
               test_top5=m["top5"], test_loss=m["loss"], test_n=m["n"], seconds=time.time() - t0)
    atomic_write_json(tp, res)
    return res
