"""Datasets, integrity checks, fixed stratified splits and GPU loaders with keyed random streams (plan section 2).

Random streams are independent and keyed:
    sample order   (seed, 'order', epoch)
    augmentation   (seed, 'aug', epoch)   one crop offset pair and one flip bit per training example, indexed by the
                                          example's position in the training subset, so every method sees the same
                                          augmented example at the same step for a given seed
    probes         (split_seed, 'probe')  a fixed validation subset, never drawn from a training stream
Downloads are synchronous: a run starts only after the archive MD5 and the extracted-file integrity checks pass.
"""
import os, time, json, tarfile, hashlib, urllib.request
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

from .config import key_seed, sha256_bytes, atomic_write_json, read_json, digest_of

DS_META = {
    "cifar10":  dict(num_classes=10,  in_ch=3, img=32, patch=4, mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616)),
    "cifar100": dict(num_classes=100, in_ch=3, img=32, patch=4, mean=(0.5071, 0.4865, 0.4409), std=(0.2673, 0.2564, 0.2762)),
    "fake10":   dict(num_classes=10,  in_ch=3, img=32, patch=4, mean=(0.5, 0.5, 0.5), std=(0.25, 0.25, 0.25)),
    "fake100":  dict(num_classes=100, in_ch=3, img=32, patch=4, mean=(0.5, 0.5, 0.5), std=(0.25, 0.25, 0.25)),
}


def _gen(*parts):
    g = torch.Generator(); g.manual_seed(key_seed(*parts)); return g


def _md5(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""): h.update(b)
    return h.hexdigest()


def fetch(url, dest, md5=None, retries=30, timeout=60):
    """Resumable download with retries; returns True only when dest exists and matches md5."""
    dest = Path(dest); dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and (md5 is None or _md5(dest) == md5):
        return True
    part = dest.with_name(dest.name + ".part")
    for attempt in range(1, retries + 1):
        try:
            have = part.stat().st_size if part.exists() else 0
            hdr = {"User-Agent": "Mozilla/5.0"}
            if have: hdr["Range"] = f"bytes={have}-"
            with urllib.request.urlopen(urllib.request.Request(url, headers=hdr), timeout=timeout) as r:
                if have and getattr(r, "status", 200) != 206:
                    have = 0; part.unlink(missing_ok=True)
                total = have + int(r.headers.get("Content-Length", 0)); last = -1
                with open(part, "ab" if have else "wb") as f:
                    while True:
                        chunk = r.read(1 << 18)
                        if not chunk: break
                        f.write(chunk); have += len(chunk)
                        pct = int(100 * have / total) if total else 0
                        if total and pct // 10 > last // 10:
                            last = pct; print(f"  {dest.name}: {pct}%", flush=True)
            if md5 is not None and _md5(part) != md5:
                part.unlink(missing_ok=True); raise IOError("md5 mismatch")
            part.replace(dest); return True
        except Exception as e:
            print(f"  download attempt {attempt}/{retries} for {dest.name} failed: {repr(e)[:120]}", flush=True)
            time.sleep(min(5 * attempt, 60))
    return False


def prepare_dataset(name, data_dir):
    """Synchronous download + integrity verification. Returns a dataset-version record (raises on failure)."""
    import torchvision
    data_dir = Path(data_dir)
    if name.startswith("fake"):
        return dict(name=name, source="synthetic keyed random tensors", archive_md5=None)
    cls = torchvision.datasets.CIFAR10 if name == "cifar10" else torchvision.datasets.CIFAR100
    if not fetch(cls.url, data_dir / cls.filename, cls.tgz_md5):
        raise RuntimeError(f"{name}: download failed; place {cls.filename} in {data_dir} and re-run")
    for train in (True, False):                     # extracts if needed; torchvision then checks every file's md5
        ds = cls(str(data_dir), train=train, download=True)
        if not ds._check_integrity():
            raise RuntimeError(f"{name}: extracted files failed the integrity check")
    return dict(name=name, source=f"torchvision.datasets.{cls.__name__} {torchvision.__version__}", archive=cls.filename,
                archive_md5=cls.tgz_md5, archive_md5_verified=True, extracted_files_verified=True)


_ARR = {}


def load_arrays(name, data_dir, partition):
    """Whole partition as uint8 (N, C, H, W) and int64 labels. partition in {'train', 'test'}."""
    key = (name, partition)
    if key in _ARR:
        return _ARR[key]
    if name.startswith("fake"):
        meta = DS_META[name]; n = 1024 if partition == "train" else 256
        g = _gen("fake", name, partition)
        x = torch.randint(0, 256, (n, 3, 32, 32), generator=g, dtype=torch.uint8)
        y = torch.arange(n) % meta["num_classes"]
        y = y[torch.randperm(n, generator=g)]
    else:
        import torchvision
        cls = torchvision.datasets.CIFAR10 if name == "cifar10" else torchvision.datasets.CIFAR100
        ds = cls(str(data_dir), train=(partition == "train"), download=False)
        x = torch.as_tensor(np.asarray(ds.data)).permute(0, 3, 1, 2).contiguous()
        y = torch.as_tensor(np.asarray(ds.targets)).long()
    _ARR[key] = (x, y)
    return _ARR[key]


def tensor_checksum(x, y):
    return sha256_bytes(x.numpy().tobytes() + y.numpy().astype("<i8").tobytes())


def dataset_version(name, data_dir, prepared):
    xtr, ytr = load_arrays(name, data_dir, "train"); xte, yte = load_arrays(name, data_dir, "test")
    return dict(prepared, train_n=len(ytr), test_n=len(yte), train_sha256=tensor_checksum(xtr, ytr), test_sha256=tensor_checksum(xte, yte))


def make_split(name, data_dir, split_dir, val_fraction=0.1, split_seed=20261030):
    """Fixed stratified validation split of the training partition. Saved to disk; returns the split record."""
    _, y = load_arrays(name, data_dir, "train")
    y_np = y.numpy(); classes = np.unique(y_np)
    val = []
    for c in classes:
        idx = np.nonzero(y_np == c)[0]
        k = int(round(val_fraction * len(idx)))
        perm = torch.randperm(len(idx), generator=_gen(split_seed, "split", name, int(c))).numpy()
        val.append(idx[perm[:k]])
    val = np.sort(np.concatenate(val)).astype(np.int64)
    train = np.setdiff1d(np.arange(len(y_np)), val).astype(np.int64)
    assert len(np.intersect1d(train, val)) == 0 and len(train) + len(val) == len(y_np)
    rec = dict(dataset=name, val_fraction=val_fraction, stratified=True, split_seed=split_seed, n_train=int(len(train)),
               n_val=int(len(val)), train_indices_sha256=sha256_bytes(train.astype("<i8").tobytes()),
               val_indices_sha256=sha256_bytes(val.astype("<i8").tobytes()),
               val_per_class={int(c): int((y_np[val] == c).sum()) for c in classes})
    rec["split_id"] = digest_of({k: rec[k] for k in ("dataset", "val_fraction", "split_seed", "train_indices_sha256", "val_indices_sha256")})[:16]
    split_dir = Path(split_dir); split_dir.mkdir(parents=True, exist_ok=True)
    np.savez(split_dir / f"{name}_{rec['split_id']}.npz", train=train, val=val)
    atomic_write_json(split_dir / f"{name}_{rec['split_id']}.json", rec)
    return rec


def load_split(split_dir, rec):
    z = np.load(Path(split_dir) / f"{rec['dataset']}_{rec['split_id']}.npz")
    train, val = z["train"], z["val"]
    if sha256_bytes(train.astype("<i8").tobytes()) != rec["train_indices_sha256"] or sha256_bytes(val.astype("<i8").tobytes()) != rec["val_indices_sha256"]:
        raise RuntimeError("saved split indices do not match their recorded checksums")
    return train, val


class KeyedLoader:
    """Batches from a GPU-resident uint8 tensor. Training mode uses the keyed order/augmentation plan of each epoch;
    evaluation mode iterates in index order without augmentation. Yields (x, y, dataset_indices)."""
    def __init__(self, x, y, indices, batch, device, mean, std, train, augment, seed=None, pad=4, limit=None):
        idx = np.asarray(indices, dtype=np.int64)
        if limit is not None:
            idx = idx[:limit]
        self.indices = torch.as_tensor(idx)
        self.x = x[self.indices].to(device); self.y = y[self.indices].to(device); self.idx_dev = self.indices.to(device)
        self.batch, self.device, self.train, self.augment, self.seed, self.pad = batch, device, train, augment, seed, pad
        self.mean = torch.tensor(mean, device=device).view(1, -1, 1, 1); self.std = torch.tensor(std, device=device).view(1, -1, 1, 1)
        self.epoch = None
    def __len__(self):
        n = len(self.indices)
        return n // self.batch if self.train else -(-n // self.batch)
    def set_epoch(self, epoch):
        self.epoch = epoch
    def epoch_plan(self, epoch):
        n = len(self.indices)
        order = torch.randperm(n, generator=_gen(self.seed, "order", epoch))
        g = _gen(self.seed, "aug", epoch)
        off = torch.randint(0, 2 * self.pad + 1, (n, 2), generator=g)
        flip = torch.rand(n, generator=g) < 0.5
        return order, off, flip
    def plan_digest(self, epoch):
        o, off, fl = self.epoch_plan(epoch)
        return sha256_bytes(o.numpy().tobytes() + off.numpy().tobytes() + fl.numpy().tobytes())
    def _aug(self, xb, off, flip):
        B, C, H, W = xb.shape; p = self.pad
        xp = F.pad(xb, (p, p, p, p))
        rows = off[:, 0:1] + torch.arange(H, device=xb.device); cols = off[:, 1:2] + torch.arange(W, device=xb.device)
        out = xp[torch.arange(B, device=xb.device)[:, None, None, None], torch.arange(C, device=xb.device)[None, :, None, None],
                 rows[:, None, :, None], cols[:, None, None, :]]
        return torch.where(flip[:, None, None, None], out.flip(-1), out)
    def __iter__(self):
        n = len(self.indices)
        if self.train:
            if self.epoch is None:
                raise RuntimeError("set_epoch() must be called before iterating a training loader")
            order, off, flip = self.epoch_plan(self.epoch)
            order, off, flip = order.to(self.device), off.to(self.device), flip.to(self.device)
        else:
            order = torch.arange(n, device=self.device)
        for k in range(len(self)):
            pos = order[k * self.batch:(k + 1) * self.batch]
            xb = self.x[pos].float() / 255.0
            if self.train and self.augment:
                xb = self._aug(xb, off[pos], flip[pos])
            yield (xb - self.mean) / self.std, self.y[pos], self.idx_dev[pos]


def probe_positions(n_val, n_probe, split_seed):
    """Fixed validation subset (positions within the validation index list) for activation diagnostics."""
    perm = torch.randperm(n_val, generator=_gen(split_seed, "probe")).numpy()
    return np.sort(perm[:min(n_probe, n_val)])
