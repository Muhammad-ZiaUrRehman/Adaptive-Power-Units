"""Training engine (plan sections 1, 2 and 5).

One call of `train_run` trains one immutable configuration and writes its record as it goes:
planned -> running -> completed | failed (failure_kind) | interrupted. Resumption happens only at completed epoch
boundaries and restores model, optimizer, scheduler, gradient-scaler state, epoch and step counters, the split id
and all random-generator states. Sample order and augmentation are keyed by (seed, epoch), so the sampling state of
an epoch boundary is fully described by the epoch number. The test partition is never loaded here.
"""
import math, os, random, time, traceback, warnings, contextlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import (digest_of, now, atomic_torch_save, sha256_file, validate_completed, DigestMismatchError)
from .activations import ACT_SPECS, activation_param_names, count_params
from .models import create_model, backbone_digest
from .data import DS_META, load_arrays, load_split, KeyedLoader, probe_positions, tensor_checksum
from .evaluate import evaluate, probe_activations, GradNormMeter, autocast_ctx


class SimulatedInterruption(Exception):
    """Raised by the engineering resume check after the checkpoint of a chosen epoch boundary is written."""


HISTORICAL_SHALLOW_RECIPE = dict(optimizer="adamw", lr=1e-3, betas=[0.9, 0.999], weight_decay=1e-4, warmup_epochs=1,
                                 label_smoothing=0.0, batch_size=128, epochs=200)


def make_config(*, dataset_version, split, arch, act_name, seed, epochs, lr, weight_decay, precision, compile,
                deterministic, diag_epochs, probe_n, source_rev, depth=2, batch_size=128, betas=(0.9, 0.999),
                warmup_epochs=1, label_smoothing=0.0, optimizer="adamw", ckpt_every=5, checkpoint_rule="final_epoch",
                train_limit=None, val_limit=None, augment=True):
    spec = dict(ACT_SPECS[act_name]); spec["name"] = act_name
    return dict(
        dataset=dataset_version,
        split={k: split[k] for k in ("dataset", "split_id", "val_fraction", "stratified", "split_seed", "n_train", "n_val",
                                     "train_indices_sha256", "val_indices_sha256")},
        fit_partition="train_minus_validation", train_limit=train_limit, val_limit=val_limit,
        model=dict(arch=arch, depth=depth) if arch == "simplecnn" else dict(arch=arch),
        activation=spec,
        optimizer=dict(name=optimizer, lr=float(lr), betas=list(betas), weight_decay=float(weight_decay),
                       param_groups="decay: trainable backbone tensors with ndim > 1; no decay: biases, norm and all activation parameters"),
        schedule=dict(kind="linear-warmup+cosine", per="step", warmup_epochs=int(min(warmup_epochs, max(1, epochs // 10)))),
        loss=dict(name="cross_entropy", label_smoothing=float(label_smoothing)),
        augmentation=dict(random_crop_pad=4, horizontal_flip=0.5) if augment else dict(none=True),
        batch_size=int(batch_size), epochs=int(epochs), eval_batch_size=512,
        precision=precision, compile=bool(compile), deterministic=bool(deterministic),
        seed=int(seed), rng_scheme="keyed-v1: init(seed,init,module) act(seed,act-init,module) order(seed,order,epoch) aug(seed,aug,epoch) probe(split_seed,probe)",
        checkpoint_rule=checkpoint_rule, ckpt_every=int(ckpt_every),
        diagnostics=dict(epochs=sorted(int(e) for e in diag_epochs if e <= epochs), probe_examples=int(probe_n)),
        source_revision=source_rev,
    )


def set_determinism(deterministic):
    torch.use_deterministic_algorithms(bool(deterministic), warn_only=True)
    torch.backends.cudnn.deterministic = bool(deterministic)
    torch.backends.cudnn.benchmark = not deterministic
    return "deterministic-algorithms(warn_only)" if deterministic else "nondeterministic(cudnn.benchmark)"


def _rng_state():
    st = dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state())
    if torch.cuda.is_available():
        st["cuda"] = torch.cuda.get_rng_state_all()
    return st


def _set_rng_state(st):
    """Generator states must be CPU ByteTensors, whatever device the checkpoint was mapped to."""
    random.setstate(st["python"]); np.random.set_state(st["numpy"])
    torch.set_rng_state(st["torch"].cpu().to(torch.uint8))
    if "cuda" in st and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([t.cpu().to(torch.uint8) for t in st["cuda"]])


def _seed_global(seed):
    random.seed(seed); np.random.seed(seed % (2 ** 32)); torch.manual_seed(seed)


def reset_compile_state():
    try:
        import torch._dynamo as dyn
        dyn.reset()
        for k in ("recompile_limit", "cache_size_limit", "accumulated_recompile_limit", "accumulated_cache_size_limit"):
            if hasattr(dyn.config, k): setattr(dyn.config, k, max(getattr(dyn.config, k), 256))
    except Exception:
        pass


def param_groups(model, wd):
    act = activation_param_names(model); decay, no_decay = [], []
    names = dict(decay=[], no_decay=[])
    for n, p in model.named_parameters():
        if not p.requires_grad: continue
        if p.ndim > 1 and n not in act:
            decay.append(p); names["decay"].append(n)
        else:
            no_decay.append(p); names["no_decay"].append(n)
    return [dict(params=decay, weight_decay=wd), dict(params=no_decay, weight_decay=0.0)], names


def make_optimizer(model, ocfg):
    groups, names = param_groups(model, ocfg["weight_decay"])
    if ocfg["name"] == "adamw":
        opt = torch.optim.AdamW(groups, lr=ocfg["lr"], betas=tuple(ocfg["betas"]))
    elif ocfg["name"] == "sgd":
        opt = torch.optim.SGD(groups, lr=ocfg["lr"], momentum=0.9, nesterov=True)
    else:
        raise ValueError(ocfg["name"])
    return opt, names


def make_scheduler(opt, steps_per_epoch, epochs, warmup_epochs):
    warm, total = max(warmup_epochs * steps_per_epoch, 1), max(epochs * steps_per_epoch, 2)
    def f(step):
        if step < warm: return (step + 1) / warm
        return 0.5 * (1 + math.cos(math.pi * (step - warm) / max(total - warm, 1)))
    return torch.optim.lr_scheduler.LambdaLR(opt, f)


_DS_VERIFIED = set()


def build_data(cfg, device, data_dir, split_dir):
    name = cfg["dataset"]["name"]; meta = DS_META[name]
    x, y = load_arrays(name, data_dir, "train")
    if name not in _DS_VERIFIED:
        if tensor_checksum(x, y) != cfg["dataset"]["train_sha256"]:
            raise RuntimeError(f"{name}: training tensors do not match the dataset version in the configuration")
        _DS_VERIFIED.add(name)
    tr_idx, va_idx = load_split(split_dir, cfg["split"])
    if np.intersect1d(tr_idx, va_idx).size:
        raise RuntimeError("training and validation indices overlap")
    if cfg.get("train_limit"): tr_idx = tr_idx[: cfg["train_limit"]]
    if cfg.get("val_limit"): va_idx = va_idx[: cfg["val_limit"]]
    aug = "none" not in cfg["augmentation"]
    tl = KeyedLoader(x, y, tr_idx, cfg["batch_size"], device, meta["mean"], meta["std"], train=True, augment=aug, seed=cfg["seed"])
    vl = KeyedLoader(x, y, va_idx, cfg["eval_batch_size"], device, meta["mean"], meta["std"], train=False, augment=False)
    pl = KeyedLoader(x, y, va_idx[probe_positions(len(va_idx), cfg["diagnostics"]["probe_examples"], cfg["split"]["split_seed"])],
                     256, device, meta["mean"], meta["std"], train=False, augment=False)
    cl = KeyedLoader(x, y, tr_idx, cfg["eval_batch_size"], device, meta["mean"], meta["std"], train=False, augment=False)
    return meta, tl, vl, pl, cl, torch.as_tensor(va_idx).to(device)


def build_model_for(cfg, device):
    meta = DS_META[cfg["dataset"]["name"]]
    model = create_model(cfg["model"], cfg["activation"], meta, cfg["seed"]).to(device)
    if device.type == "cuda" and cfg["model"]["arch"] in ("simplecnn", "resnet18"):
        model = model.to(memory_format=torch.channels_last)
    return model


def train_run(store, cfg, device, data_dir, split_dir, tags=(), stop_after_epoch=None, retry_failed=False, verbose=True,
              print_every=10, strict=False):
    d = store.register(cfg)
    rec = store.record(d)
    tags = sorted(set(rec.get("tags", [])) | set(tags))
    if rec["status"] == "completed":
        reason = validate_completed(store, d, rec)
        if reason is None:
            return store.write_record(d, dict(tags=tags))
        rec = store.write_record(d, dict(status="planned", tags=tags, invalidated=dict(time=now(), reason=reason)))
        if verbose: print(f"[{d[:16]}] previous completion invalid ({reason}); retraining")
    if rec["status"] == "failed" and not retry_failed:
        return store.write_record(d, dict(tags=tags))
    if rec["status"] == "running":                     # a previous session stopped without closing the record
        rec = store.write_record(d, dict(status="interrupted", interrupted=dict(time=now(), reason="stale 'running' record found at start")))

    ck_dir = store.ckpt_dir(d); ck_dir.mkdir(parents=True, exist_ok=True); last = ck_dir / "last.pt"
    det_mode = set_determinism(cfg["deterministic"])
    store.write_record(d, dict(status="running", tags=tags, started=rec.get("started", now()), session_started=now(),
                               determinism_mode_requested=det_mode))
    label = f"{cfg['dataset']['name']}/{cfg['model']['arch']}/{cfg['activation']['name']}/s{cfg['seed']}/lr{cfg['optimizer']['lr']}/wd{cfg['optimizer']['weight_decay']}"
    t_wall = time.time()
    det_warnings = []
    try:
        with warnings.catch_warnings(record=True) as wlist:
            warnings.simplefilter("always")
            result = _train(store, d, cfg, device, data_dir, split_dir, last, ck_dir, stop_after_epoch, verbose, print_every, label)
            det_warnings = sorted({str(w.message)[:300] for w in wlist if "determinis" in str(w.message).lower()})
        result["determinism"] = dict(requested=cfg["deterministic"], mode=det_mode, nondeterministic_op_warnings=det_warnings,
                                     actual=("deterministic" if cfg["deterministic"] and not det_warnings else
                                             "deterministic-requested-with-warnings" if cfg["deterministic"] else "nondeterministic"))
        result["wall_seconds_this_session"] = time.time() - t_wall
        return store.write_record(d, result)
    except SimulatedInterruption as e:
        return store.write_record(d, dict(status="interrupted", interrupted=dict(time=now(), reason=str(e))))
    except KeyboardInterrupt:
        store.write_record(d, dict(status="interrupted", interrupted=dict(time=now(), reason="KeyboardInterrupt")))
        raise
    except Exception as e:
        kind = getattr(e, "failure_kind", "exception")
        r = store.write_record(d, dict(status="failed", failure_kind=kind, failure_reason=repr(e)[:500], failed=now(),
                                       traceback=traceback.format_exc()[-4000:], **getattr(e, "extra", {})))
        if verbose: print(f"[{label}] FAILED ({kind}): {repr(e)[:200]}")
        if strict: raise
        return r


class _Fail(RuntimeError):
    def __init__(self, msg, kind, **extra):
        super().__init__(msg); self.failure_kind = kind; self.extra = extra


def _train(store, d, cfg, device, data_dir, split_dir, last, ck_dir, stop_after_epoch, verbose, print_every, label):
    _seed_global(cfg["seed"])
    meta, tl, vl, pl, cl, val_idx_dev = build_data(cfg, device, data_dir, split_dir)
    model = build_model_for(cfg, device)
    counts = count_params(model)
    bb_digest = backbone_digest(model)
    opt, group_names = make_optimizer(model, cfg["optimizer"])
    E = cfg["epochs"]; spe = len(tl)
    sched = make_scheduler(opt, spe, E, cfg["schedule"]["warmup_epochs"])
    crit = nn.CrossEntropyLoss(label_smoothing=cfg["loss"]["label_smoothing"])
    precision = cfg["precision"]
    use_scaler = precision == "fp16-autocast"
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler) if device.type == "cuda" else torch.amp.GradScaler("cpu", enabled=False)
    trainable = [p for p in model.parameters() if p.requires_grad]

    mc = model; compile_outcome = "not requested"; compile_seconds = None
    if cfg["compile"]:
        reset_compile_state()
        try:
            t0 = time.time(); mc = torch.compile(model)
            with torch.no_grad(), autocast_ctx(device, precision):
                was = model.training; model.eval(); mc(next(iter(vl))[0][:2]); model.train(was)
            compile_seconds = time.time() - t0; compile_outcome = "compiled"
        except Exception as e:
            raise _Fail(f"torch.compile failed (no silent eager fallback): {repr(e)[:300]}", "compile")

    hist, diag, start, step, train_seconds, eval_seconds, diag_seconds = [], {}, 0, 0, 0.0, 0.0, 0.0
    resumed_from = None
    if last.exists():
        ck = torch.load(last, map_location="cpu", weights_only=False)   # load_state_dict moves tensors to the model/optimizer device
        if ck.get("digest") != d or ck.get("split_id") != cfg["split"]["split_id"]:
            raise _Fail("resume checkpoint belongs to another configuration or split", "checkpoint")
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        scaler.load_state_dict(ck["scaler"]); _set_rng_state(ck["rng"])
        hist, diag, start, step = ck["hist"], ck["diag"], ck["epoch"], ck["step"]
        train_seconds, eval_seconds, diag_seconds = ck["times"]
        resumed_from = start
        if verbose: print(f"[{label}] resumed at epoch boundary {start}")

    if device.type == "cuda": torch.cuda.reset_peak_memory_stats()
    peak_mem = None; samples_total = sum(h["samples"] for h in hist)
    diag_epochs = set(cfg["diagnostics"]["epochs"])
    for epoch in range(start, E):
        ep = epoch + 1
        tl.set_epoch(epoch); model.train()
        gm = GradNormMeter(model) if ep in diag_epochs else None
        t0 = time.time()
        loss_sum = torch.zeros((), device=device); corr = torch.zeros((), device=device, dtype=torch.long)
        nonfinite = torch.zeros((), device=device, dtype=torch.long); leak = torch.zeros((), device=device, dtype=torch.long)
        gns = []; n_seen = 0
        for x, y, idx in tl:
            if device.type == "cuda" and cfg["model"]["arch"] in ("simplecnn", "resnet18"):
                x = x.contiguous(memory_format=torch.channels_last)
            opt.zero_grad(set_to_none=True)
            with autocast_ctx(device, precision):
                out = mc(x); loss = crit(out, y)
            if use_scaler:
                scaler.scale(loss).backward(); scaler.unscale_(opt)
            else:
                loss.backward()
            if gm is not None: gm.update()
            gn = torch.nn.utils.clip_grad_norm_(trainable, float("inf")) if trainable else torch.zeros((), device=device)
            if use_scaler: scaler.step(opt); scaler.update()
            else: opt.step()
            sched.step(); step += 1
            lf = loss.detach().float()
            nonfinite += (~torch.isfinite(lf)).long() + (~torch.isfinite(gn.detach())).long()
            loss_sum += lf * y.numel(); corr += (out.detach().argmax(1) == y).sum()
            leak += torch.isin(idx, val_idx_dev).sum(); gns.append(gn.detach().float()); n_seen += y.numel()
        if device.type == "cuda": torch.cuda.synchronize()
        t_train = time.time() - t0; train_seconds += t_train
        if peak_mem is None and device.type == "cuda": peak_mem = torch.cuda.max_memory_allocated() / 2**20
        if leak.item() != 0:
            raise _Fail(f"{leak.item()} validation examples entered optimisation batches", "data")
        g = torch.stack(gns) if gns else torch.zeros(1); gf = g[torch.isfinite(g)]
        train_loss = loss_sum.item() / max(n_seen, 1)
        entry = dict(epoch=ep, lr=opt.param_groups[0]["lr"], samples=n_seen, train_loss_aug=train_loss,
                     train_acc_aug=100.0 * corr.item() / max(n_seen, 1), grad_norm_mean=gf.mean().item() if gf.numel() else float("nan"),
                     grad_norm_max=gf.max().item() if gf.numel() else float("nan"), nonfinite_events=int(nonfinite.item()), time_train=t_train)
        samples_total += n_seen
        if nonfinite.item() > 0 or not math.isfinite(train_loss):
            hist.append(entry)
            raise _Fail(f"non-finite loss or gradient in epoch {ep} ({int(nonfinite.item())} events)", "numerical",
                        history=hist, failed_epoch=ep, param_counts=counts, label=label, seed=cfg["seed"], activation=cfg["activation"])
        t1 = time.time(); ev = evaluate(model, vl, device, precision); t_eval = time.time() - t1; eval_seconds += t_eval
        entry.update(val_top1=ev["top1"], val_top5=ev["top5"], val_loss=ev["loss"], time_eval=t_eval)
        if ep in diag_epochs:
            t2 = time.time()
            ce = evaluate(model, cl, device, precision)
            stats, bn_ok, mode_ok = probe_activations(model, pl, device, precision)
            if not (bn_ok and mode_ok):
                raise _Fail("diagnostic probe changed BatchNorm statistics or the model mode", "exception")
            diag[str(ep)] = dict(clean_train_top1=ce["top1"], clean_train_loss=ce["loss"], val_top1=ev["top1"], val_loss=ev["loss"],
                                 generalization_gap_top1=ce["top1"] - ev["top1"], activations=stats, grad_norms=gm.result(),
                                 bn_unchanged_by_probe=bn_ok, model_mode="eval during probe; restored to train")
            entry.update(clean_train_top1=ce["top1"])
            diag_seconds += time.time() - t2
        hist.append(entry)
        if verbose and (ep == 1 or ep % print_every == 0 or ep == E):
            print(f"[{label}] ep {ep}/{E} loss {train_loss:.3f} val {ev['top1']:.2f} gn {entry['grad_norm_mean']:.3f} {t_train:.1f}s", flush=True)
        if ep < E and (ep % cfg["ckpt_every"] == 0 or ep == stop_after_epoch):
            atomic_torch_save(last, dict(digest=d, split_id=cfg["split"]["split_id"], epoch=ep, step=step, model=model.state_dict(),
                                         opt=opt.state_dict(), sched=sched.state_dict(), scaler=scaler.state_dict(), rng=_rng_state(),
                                         hist=hist, diag=diag, times=(train_seconds, eval_seconds, diag_seconds),
                                         sampling_state=dict(scheme="keyed (seed, epoch)", next_epoch=ep)))
            store.write_record(d, dict(last_checkpoint_epoch=ep, epochs_done=ep))
        if stop_after_epoch is not None and ep == stop_after_epoch:
            raise SimulatedInterruption(f"stopped after epoch {ep} for the resume check")

    final = ck_dir / "final.pt"
    atomic_torch_save(final, dict(digest=d, epoch=E, model=model.state_dict()))
    sha = sha256_file(final)
    chk = torch.load(final, map_location=device, weights_only=False)            # readable and restorable
    probe_model = build_model_for(cfg, device); probe_model.load_state_dict(chk["model"])
    f = hist[-1]
    fm = dict(val_top1=f["val_top1"], val_top5=f["val_top5"], val_loss=f["val_loss"], train_loss_aug=f["train_loss_aug"],
              train_acc_aug=f["train_acc_aug"], clean_train_top1=f.get("clean_train_top1"))
    if not all(math.isfinite(v) for v in (fm["val_top1"], fm["val_loss"])):
        raise _Fail("final metrics are not finite", "metrics")
    if last.exists(): last.unlink()
    return dict(status="completed", completed=now(), label=label, config_digest=d, source_revision=cfg["source_revision"],
                dataset_checksums=dict(train=cfg["dataset"]["train_sha256"], test=cfg["dataset"]["test_sha256"]),
                split_checksums=dict(train=cfg["split"]["train_indices_sha256"], val=cfg["split"]["val_indices_sha256"]),
                seed=cfg["seed"], activation=cfg["activation"], param_counts=counts, backbone_init_sha256=bb_digest,
                optimizer=cfg["optimizer"], optimizer_groups=dict(decay=len(group_names["decay"]), no_decay=len(group_names["no_decay"]),
                no_decay_names=group_names["no_decay"]), schedule=cfg["schedule"], precision=precision,
                grad_scaler="enabled" if use_scaler else "disabled (bf16 or fp32)", compile_outcome=compile_outcome,
                compile_seconds=compile_seconds, epochs_done=E, samples_seen=samples_total, steps=step, resumed_from_epoch=resumed_from,
                times=dict(train_s=train_seconds, eval_s=eval_seconds, diagnostics_s=diag_seconds,
                           median_epoch_train_s=float(np.median([h["time_train"] for h in hist]))),
                peak_mem_mb=peak_mem, peak_mem_scope="torch.cuda.max_memory_allocated during epoch 1 of this session (model, optimizer, GPU-resident data)",
                evaluation_checkpoint=dict(path=str(final.relative_to(store.root)).replace("\\", "/"), sha256=sha, epoch=E,
                                           rule=cfg["checkpoint_rule"]),
                final_metrics=fm, history=hist, diagnostics=diag, test_partition_loaded=False)


def run_many(store, jobs, device, data_dir, split_dir, verbose=True, strict=False, print_every=10):
    """jobs: list of (cfg, tags). Runs in the given order (callers order them seed-first). Returns status counts."""
    from collections import Counter
    try:
        from tqdm.auto import tqdm
    except Exception:
        tqdm = lambda x, **k: x
    c = Counter()
    todo = []
    for cfg, tags in jobs:
        d = store.register(cfg); r = store.record(d)
        if r["status"] == "completed" and validate_completed(store, d, r) is None:
            store.write_record(d, dict(tags=sorted(set(r.get("tags", [])) | set(tags)))); c["cached"] += 1
        elif r["status"] == "failed":
            store.write_record(d, dict(tags=sorted(set(r.get("tags", [])) | set(tags)))); c["failed (kept)"] += 1
        else:
            todo.append((cfg, tags))
    print(f"{len(jobs)} jobs: {c['cached']} already completed, {c['failed (kept)']} failed earlier (kept), {len(todo)} to run")
    for cfg, tags in tqdm(todo, desc="runs"):
        r = train_run(store, cfg, device, data_dir, split_dir, tags=tags, verbose=verbose, strict=strict, print_every=print_every)
        c[r["status"]] += 1
    return dict(c)
