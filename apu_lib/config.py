"""Configuration, digests, provenance and run records (plan section 1 and 6).

Every run is described by one immutable configuration dictionary. Its digest is the SHA-256 of a canonical
serialization (sorted keys, no whitespace, floats written with repr). Records, checkpoints and test metrics are keyed
by that digest. A cached record is only reused when its stored configuration has exactly the same digest; anything
else raises DigestMismatchError instead of silently reusing or overwriting a result.
"""
import hashlib, json, os, platform, subprocess, sys, time, datetime, traceback, io
from pathlib import Path

SCHEMA = "apu-run-v2"
STATUSES = ("planned", "running", "completed", "failed", "interrupted")
FAILURE_KINDS = ("numerical", "compile", "benchmark", "exception", "data", "checkpoint", "metrics")
TRAINING_MODULES = ("config.py", "activations.py", "models.py", "data.py", "train.py", "evaluate.py")


class DigestMismatchError(RuntimeError):
    """A stored result does not belong to the requested configuration."""


class ProtocolError(RuntimeError):
    """An action would violate the frozen protocol (for example test access before the lock)."""


# ---------------------------------------------------------------- canonical serialization and hashing
def _norm(o):
    if isinstance(o, dict):
        return {str(k): _norm(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_norm(v) for v in o]
    if isinstance(o, float):
        if o != o or o in (float("inf"), float("-inf")):
            raise ValueError("non-finite float in configuration")
        return o
    if isinstance(o, (str, int, bool)) or o is None:
        return o
    if hasattr(o, "item"):                      # numpy / torch scalars
        return _norm(o.item())
    raise TypeError(f"configuration value of type {type(o)} is not serializable: {o!r}")


def canonical_json(obj):
    return json.dumps(_norm(obj), sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def digest_of(obj):
    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def key_seed(*parts):
    """64-bit seed derived from a key such as (seed, 'order', epoch). Used for every independent random stream."""
    h = hashlib.sha256("|".join(str(p) for p in parts).encode("utf-8")).digest()
    return int.from_bytes(h[:8], "little") & ((1 << 63) - 1)


# ---------------------------------------------------------------- atomic writes
def atomic_write_bytes(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    with open(tmp, "wb") as f:
        f.write(data); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_write_json(path, obj):
    atomic_write_bytes(path, json.dumps(obj, indent=2, sort_keys=True, default=_json_default).encode("utf-8"))


def atomic_write_text(path, text):
    atomic_write_bytes(path, text.encode("utf-8"))


def atomic_torch_save(path, obj):
    import torch
    buf = io.BytesIO(); torch.save(obj, buf); atomic_write_bytes(path, buf.getvalue())


def _json_default(o):
    if hasattr(o, "item"):
        return o.item()
    if hasattr(o, "tolist"):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    return str(o)


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def now():
    return datetime.datetime.now().isoformat(timespec="seconds")


# ---------------------------------------------------------------- provenance
def source_revision(lib_dir):
    """Hash of the training-relevant source files plus the git commit if the folder is a repository."""
    lib_dir = Path(lib_dir); files = {}
    for name in TRAINING_MODULES:
        p = lib_dir / name
        files[name] = sha256_file(p) if p.exists() else "missing"
    code_digest = digest_of(files)
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=lib_dir, capture_output=True, text=True, timeout=10)
        git = r.stdout.strip() if r.returncode == 0 and r.stdout.strip() else "unavailable"
        if git != "unavailable":
            d = subprocess.run(["git", "status", "--porcelain"], cwd=lib_dir, capture_output=True, text=True, timeout=10)
            git += "+dirty" if d.stdout.strip() else ""
    except Exception:
        git = "unavailable"
    return dict(code_sha256=code_digest, files=files, git_commit=git)


def environment_record(deterministic_mode=None):
    import torch, numpy, scipy, pandas, matplotlib
    try:
        import torchvision; tv = torchvision.__version__
    except Exception:
        tv = "unavailable"
    rec = dict(date=now(), python=platform.python_version(), executable=sys.executable, platform=platform.platform(),
               cpu=platform.processor() or "unavailable", cpu_count=os.cpu_count(), torch=torch.__version__, torchvision=tv,
               numpy=numpy.__version__, scipy=scipy.__version__, pandas=pandas.__version__, matplotlib=matplotlib.__version__,
               cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
               cuda_available=torch.cuda.is_available(), cublas_workspace_config=os.environ.get("CUBLAS_WORKSPACE_CONFIG", "unset"),
               cudnn_benchmark=torch.backends.cudnn.benchmark, cudnn_deterministic=torch.backends.cudnn.deterministic,
               deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
               deterministic_warn_only=torch.is_deterministic_algorithms_warn_only_enabled(),
               deterministic_mode=deterministic_mode)
    try:
        import psutil; rec["ram_gb"] = round(psutil.virtual_memory().total / 2**30, 1)
    except Exception:
        rec["ram_gb"] = "unavailable"
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        rec.update(gpu=p.name, gpu_mem_gb=round(p.total_memory / 2**30, 2), gpu_capability=list(torch.cuda.get_device_capability(0)),
                   bf16_supported=torch.cuda.is_bf16_supported())
        try:
            r = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], capture_output=True, text=True, timeout=20)
            rec["driver"] = r.stdout.strip() or "unavailable"
        except Exception:
            rec["driver"] = "unavailable"
    else:
        rec.update(gpu=None, driver="unavailable")
    return rec


# ---------------------------------------------------------------- directory layout and run store
class Store:
    """root/
         configs/<id>.json        immutable configuration (written once)
         runs/<id>.json           run record (status, metrics, provenance)
         runs/<id>.test.json      one-time test evaluation (written only after the protocol lock)
         checkpoints/<id>/        last.pt (resume state), final.pt (evaluation checkpoint)
         splits/ protocol/ engineering/ benchmarks/ diagnostics/ audit/ logs/ paper_assets/
       <id> is the first 16 hex characters of the digest; the full digest is stored and verified inside every file."""
    SUBDIRS = ("configs", "runs", "checkpoints", "splits", "protocol", "engineering", "benchmarks", "diagnostics", "audit",
               "logs", "paper_assets", "paper_assets/figures", "paper_assets/tables")

    def __init__(self, root):
        self.root = Path(root)
        for d in self.SUBDIRS:
            (self.root / d).mkdir(parents=True, exist_ok=True)

    @staticmethod
    def rid(digest):
        return digest[:16]

    def config_path(self, d): return self.root / "configs" / f"{self.rid(d)}.json"
    def record_path(self, d): return self.root / "runs" / f"{self.rid(d)}.json"
    def test_path(self, d): return self.root / "runs" / f"{self.rid(d)}.test.json"
    def ckpt_dir(self, d): return self.root / "checkpoints" / self.rid(d)

    # -- configurations
    def register(self, cfg):
        """Write the immutable configuration; verify an existing one is identical. Returns the digest."""
        d = digest_of(cfg); p = self.config_path(d)
        if p.exists():
            stored = read_json(p)
            if stored.get("digest") != d or digest_of(stored.get("config")) != d:
                raise DigestMismatchError(f"configuration file {p} does not match digest {d}; refusing to reuse it")
        else:
            atomic_write_json(p, dict(schema=SCHEMA, digest=d, created=now(), config=cfg))
        if not self.record_path(d).exists():
            self.write_record(d, dict(status="planned"))
        return d

    def config(self, d):
        stored = read_json(self.config_path(d))
        if digest_of(stored["config"]) != d:
            raise DigestMismatchError(f"stored configuration for {d[:16]} was modified")
        return stored["config"]

    # -- records
    def record(self, d):
        p = self.record_path(d)
        if not p.exists():
            return None
        r = read_json(p)
        if r.get("digest") != d:
            raise DigestMismatchError(f"record {p} belongs to digest {r.get('digest')}, expected {d}")
        return r

    def write_record(self, d, updates):
        r = self.record(d) or dict(schema=SCHEMA, digest=d, run_id=self.rid(d), history_of_status=[])
        status = updates.get("status", r.get("status"))
        if status not in STATUSES:
            raise ValueError(f"unknown status {status}")
        if updates.get("status") and updates["status"] != r.get("status"):
            r.setdefault("history_of_status", []).append(dict(status=updates["status"], time=now()))
        r.update(updates); r["updated"] = now()
        atomic_write_json(self.record_path(d), r)
        return r

    def load_cached(self, cfg):
        """Return a completed, valid record for exactly this configuration, None if not completed.
        Raises DigestMismatchError when stored data claims this digest but holds a different configuration."""
        d = digest_of(cfg)
        if not self.config_path(d).exists():
            return None
        if digest_of(self.config(d)) != d or canonical_json(self.config(d)) != canonical_json(cfg):
            raise DigestMismatchError(f"stored configuration for {d[:16]} differs from the requested one")
        r = self.record(d)
        if r and r.get("status") == "completed" and validate_completed(self, d, r) is None:
            return r
        return None

    def all_records(self):
        out = []
        for p in sorted((self.root / "runs").glob("*.json")):
            if p.name.endswith(".test.json"):
                continue
            r = read_json(p); out.append(r)
        return out


def load_foreign_result(store, path, cfg):
    """Loading a result file that was not produced for `cfg` (for example a historical v1 JSON) must fail visibly."""
    r = read_json(path)
    d = digest_of(cfg)
    if r.get("digest") != d or r.get("schema") != SCHEMA:
        raise DigestMismatchError(f"{path} has digest {r.get('digest', 'none (historical format)')} and schema "
                                  f"{r.get('schema', 'none')}; it cannot stand in for configuration {d[:16]}")
    return r


def validate_completed(store, d, r=None):
    """None if the record may be called completed, otherwise the reason. Completed requires a readable final
    checkpoint whose checksum matches the record and finite final metrics."""
    import math
    r = r or store.record(d)
    if r is None:
        return "no record"
    ck = r.get("evaluation_checkpoint")
    if not ck:
        return "no evaluation checkpoint recorded"
    p = store.root / ck["path"]
    if not p.exists():
        return f"evaluation checkpoint {p} missing"
    if ck.get("sha256") and sha256_file(p) != ck["sha256"]:
        return "evaluation checkpoint checksum mismatch"
    m = r.get("final_metrics") or {}
    for k in ("val_top1", "val_loss"):
        v = m.get(k)
        if v is None or not math.isfinite(v):
            return f"final metric {k} missing or non-finite"
    return None
