"""Activation functions (plan section 3).

Adaptive Power Unit, as implemented and as used in the historical study:

    s     = z^2 + eps
    p(z)  = p0 + A * tanh(w * z + b),   A = a_max * tanh(a)
    f(z)  = z * s^((p(z) - 1) / 2)       (approximately sign(z) * |z|^p(z))

a, w and b are learned scalars shared by all elements of one activation module ("module" sharing). Historical
defaults (named configuration "apu_v1"): p0 = 1, a_max = 0.5, eps = 1e-3, initial A = 0.3, w = 0.5, b = 0.
With these defaults p(z) lies in (0.5, 1.5); exponents above one are allowed. Bounded exponents do NOT guarantee
monotonicity, globally bounded gradients or successful training: with p > 1 activations can grow across residual
blocks, and the learned-exponent variants can diverge. Per-channel parameters ("channel" sharing) are a different
method configuration with their own parameter counts.

Other modes kept for the historical ablation:
    fixed    p = p0, not learned
    const    p = p_min + (p_max - p_min) * sigmoid(a), one learned scalar (bounded in (p_min, p_max))
    linear   p = clamp(p0 + w_p * (w_z * z + b_z), 0.1, 4)   (clamped to [0.1, 4]; not unbounded)
    sigmoid  p = p_min + (p_max - p_min) * sigmoid(w_z * z + b_z)
    full     the APU above

The forward pass computes the power in float32 (also under autocast). A dtype-preserving reference implementation
(`apu_reference`) exists for float64 derivative checks; it is validated against the production path separately.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import key_seed

APU_V1 = dict(p0=1.0, a_max=0.5, eps=1e-3, init_A=0.3, init_wz=0.5, init_bz=0.0, p_min=0.5, p_max=1.5)


def _keyed_gen(*parts):
    g = torch.Generator(); g.manual_seed(key_seed(*parts)); return g


class APU(nn.Module):
    def __init__(self, mode="full", p0=1.0, a_max=0.5, init_A=0.3, init_wz=0.5, init_bz=0.0, p_min=0.5, p_max=1.5,
                 eps=1e-3, num_channels=None, channel_dim=1, trainable=True):
        super().__init__()
        self.mode, self.p0, self.a_max, self.p_min, self.p_max, self.eps = mode, float(p0), float(a_max), float(p_min), float(p_max), float(eps)
        self.channel_dim, self.trainable = channel_dim, trainable
        self.init_args = dict(mode=mode, p0=p0, a_max=a_max, init_A=init_A, init_wz=init_wz, init_bz=init_bz, p_min=p_min,
                              p_max=p_max, eps=eps, num_channels=num_channels, channel_dim=channel_dim, trainable=trainable)
        shape = (1,) if num_channels is None else (int(num_channels),)
        P_ = lambda v: nn.Parameter(torch.full(shape, float(v)), requires_grad=trainable)
        if mode == "fixed":
            pass
        elif mode == "const":
            frac = min(max((p0 - p_min) / (p_max - p_min), 1e-6), 1 - 1e-6)
            self.a = P_(math.log(frac / (1 - frac)))
        elif mode == "linear":
            self.w_p, self.w_z, self.b_z = P_(0.1), P_(0.1), P_(0.0)
        elif mode == "sigmoid":
            self.w_z, self.b_z = P_(init_wz), P_(init_bz)
        elif mode == "full":
            if not abs(init_A) < a_max:
                raise ValueError("init_A must satisfy |init_A| < a_max")
            self.raw_A = P_(math.atanh(init_A / a_max))
            self.w_z, self.b_z = P_(init_wz), P_(init_bz)
        else:
            raise ValueError(mode)

    def _b(self, t, z):
        if t.numel() == 1 or z.dim() == 1:
            return t
        shape = [1] * z.dim(); shape[self.channel_dim] = -1
        return t.view(shape)

    def exponent(self, z):
        m = self.mode
        if m == "fixed":
            return torch.full_like(z, self.p0)
        if m == "const":
            return (self.p_min + (self.p_max - self.p_min) * torch.sigmoid(self._b(self.a, z))).expand_as(z)
        if m == "linear":
            return (self.p0 + self._b(self.w_p, z) * (self._b(self.w_z, z) * z + self._b(self.b_z, z))).clamp(0.1, 4.0)
        if m == "sigmoid":
            return self.p_min + (self.p_max - self.p_min) * torch.sigmoid(self._b(self.w_z, z) * z + self._b(self.b_z, z))
        A = self.a_max * torch.tanh(self._b(self.raw_A, z))
        return self.p0 + A * torch.tanh(self._b(self.w_z, z) * z + self._b(self.b_z, z))

    def amplitude(self):
        return (self.a_max * torch.tanh(self.raw_A)).detach() if self.mode == "full" else None

    def forward(self, z):
        dt = z.dtype
        with torch.autocast(device_type=z.device.type, enabled=False):
            z32 = z.float()
            p = self.exponent(z32)
            out = z32 * (z32 * z32 + self.eps).pow(0.5 * (p - 1.0))
        return out.to(dt)

    def extra_repr(self):
        return f"mode={self.mode}, p0={self.p0}, a_max={self.a_max}, eps={self.eps}, trainable={self.trainable}"


def apu_reference(z, mode="full", p0=1.0, a_max=0.5, eps=1e-3, p_min=0.5, p_max=1.5, raw_A=None, w_z=None, b_z=None,
                  a=None, w_p=None):
    """Dtype-preserving reference (no float32 cast). Parameters are tensors of z's dtype, already broadcastable."""
    if mode == "fixed":
        p = torch.full_like(z, p0)
    elif mode == "const":
        p = (p_min + (p_max - p_min) * torch.sigmoid(a)).expand_as(z)
    elif mode == "linear":
        p = (p0 + w_p * (w_z * z + b_z)).clamp(0.1, 4.0)
    elif mode == "sigmoid":
        p = p_min + (p_max - p_min) * torch.sigmoid(w_z * z + b_z)
    else:
        p = p0 + a_max * torch.tanh(raw_A) * torch.tanh(w_z * z + b_z)
    return z * (z * z + eps).pow(0.5 * (p - 1.0))


class ESReLU(nn.Module):
    """Historical baseline: z for z > 0, alpha * (exp(-gamma z^2) - 1) otherwise."""
    def __init__(self, alpha=0.1, gamma=1.0):
        super().__init__(); self.alpha, self.gamma = alpha, gamma
    def forward(self, x):
        return torch.where(x > 0, x, self.alpha * (torch.exp(-self.gamma * x * x) - 1))


class ACONC(nn.Module):
    """ACON-C (Ma et al., "Activate or Not: Learning Customized Activation", CVPR 2021):
        f(x) = (p1 - p2) * x * sigmoid(beta * (p1 - p2) * x) + p2 * x
    with per-channel learnable p1, p2, beta. Initialisation follows the authors' public code (p1, p2 ~ N(0, 1),
    beta = 1); here the normal draws come from a keyed generator so they do not shift any other random stream.
    Identities checked in the engineering cell: p1 = 1, p2 = 0, beta = 1 gives SiLU; large beta approaches
    max(p1 x, p2 x). Re-check against the primary source before submission."""
    def __init__(self, num_channels, channel_dim=1):
        super().__init__()
        self.channel_dim, self.C = channel_dim, int(num_channels)
        self.p1 = nn.Parameter(torch.zeros(self.C)); self.p2 = nn.Parameter(torch.zeros(self.C)); self.beta = nn.Parameter(torch.ones(self.C))
        self.init_args = dict(num_channels=num_channels, channel_dim=channel_dim, init="p1,p2~N(0,1) keyed; beta=1")

    def reset_parameters_keyed(self, seed, name):
        g = _keyed_gen(seed, "act-init", name)
        with torch.no_grad():
            self.p1.copy_(torch.randn(self.C, generator=g)); self.p2.copy_(torch.randn(self.C, generator=g)); self.beta.fill_(1.0)

    def _b(self, t, x):
        if x.dim() == 1:
            return t
        shape = [1] * x.dim(); shape[self.channel_dim] = -1
        return t.view(shape)

    def forward(self, x):
        p1, p2, beta = self._b(self.p1, x), self._b(self.p2, x), self._b(self.beta, x)
        d = (p1 - p2) * x
        return d * torch.sigmoid(beta * d) + p2 * x


_PADE_CACHE = {}


def pade_init_coefficients(target="leaky_relu", m=5, n=4, lo=-3.0, hi=3.0, steps=3000):
    """Least-squares fit of a safe Pade activation (numerator degree m, denominator Q(x) = 1 + sum_k |b_k| |x|^k,
    Molina et al., ICLR 2020) to the target on [lo, hi]. Deterministic (float64, fixed grid, Adam then LBFGS)."""
    key = (target, m, n, lo, hi)
    if key in _PADE_CACHE:
        return _PADE_CACHE[key]
    x = torch.linspace(lo, hi, 2001, dtype=torch.float64)
    y = F.leaky_relu(x, 0.01) if target == "leaky_relu" else F.gelu(x)
    a = torch.zeros(m + 1, dtype=torch.float64); a[1] = 0.5; a[2] = 0.3
    b = torch.full((n,), 0.1, dtype=torch.float64)
    a.requires_grad_(True); b.requires_grad_(True)
    def model():
        num = sum(a[k] * x ** k for k in range(m + 1))
        den = 1 + sum(b[k - 1].abs() * x.abs() ** k for k in range(1, n + 1))
        return num / den
    opt = torch.optim.Adam([a, b], lr=1e-2)
    for _ in range(steps):
        opt.zero_grad(); loss = ((model() - y) ** 2).mean(); loss.backward(); opt.step()
    opt2 = torch.optim.LBFGS([a, b], lr=0.5, max_iter=500, line_search_fn="strong_wolfe")
    def closure():
        opt2.zero_grad(); l = ((model() - y) ** 2).mean(); l.backward(); return l
    opt2.step(closure)
    with torch.no_grad():
        err = (model() - y).abs().max().item()
    res = (a.detach().clone(), b.detach().clone(), err)
    _PADE_CACHE[key] = res
    return res


class PadeActivation(nn.Module):
    """Safe Pade activation unit, module-shared coefficients (m + 1 + n learned scalars per module)."""
    def __init__(self, m=5, n=4, target="leaky_relu"):
        super().__init__()
        a, b, err = pade_init_coefficients(target, m, n)
        self.num = nn.Parameter(a.float().clone()); self.den = nn.Parameter(b.float().clone())
        self.m, self.n, self.fit_max_error = m, n, err
        self.init_args = dict(m=m, n=n, target=target, fit_range=[-3.0, 3.0], fit_max_abs_error=err)

    def forward(self, x):
        dt = x.dtype
        with torch.autocast(device_type=x.device.type, enabled=False):
            x = x.float()
            num = self.num[self.m] * torch.ones_like(x)
            for k in range(self.m - 1, -1, -1):              # Horner
                num = num * x + self.num[k]
            ax = x.abs(); den = torch.ones_like(x); pw = torch.ones_like(x)
            for k in range(self.n):
                pw = pw * ax; den = den + self.den[k].abs() * pw
            out = num / den
        return out.to(dt)


# ---------------------------------------------------------------- named method specifications
# Every spec is a plain dict that goes verbatim into the run configuration (and therefore into the digest).
ACT_SPECS = {
    "relu":       dict(kind="relu"),
    "gelu":       dict(kind="gelu"),
    "silu":       dict(kind="silu"),
    "mish":       dict(kind="mish"),
    "leakyrelu":  dict(kind="leakyrelu", negative_slope=0.01),
    "prelu":      dict(kind="prelu", init=0.25, sharing="module"),
    "esrelu":     dict(kind="esrelu", alpha=0.1, gamma=1.0),
    # historical full APU ("apu_v1"): learned, module-shared
    "apu":        dict(kind="apu", mode="full", sharing="module", trainable=True, **APU_V1),
    # matched frozen profile: identical initial function, activation parameters not learned
    "apu_frozen": dict(kind="apu", mode="full", sharing="module", trainable=False, **APU_V1),
    # learned constant power initialised at p = 1 (starts as the identity; cannot match the full initial profile)
    "apu_const":  dict(kind="apu", mode="const", sharing="module", trainable=True, **{**APU_V1, "p0": 1.0}),
    # per-channel full APU: a different method configuration (not part of the core study)
    "apu_pc":     dict(kind="apu", mode="full", sharing="channel", trainable=True, **APU_V1),
    "apu_fixed2": dict(kind="apu", mode="fixed", sharing="module", trainable=False, **{**APU_V1, "p0": 2.0}),
    "apu_linear": dict(kind="apu", mode="linear", sharing="module", trainable=True, **APU_V1),
    "apu_sigmoid": dict(kind="apu", mode="sigmoid", sharing="module", trainable=True, **APU_V1),
    "acon_c":     dict(kind="acon_c", sharing="channel", init="p1,p2~N(0,1) keyed; beta=1"),
    "pade":       dict(kind="pade", m=5, n=4, target="leaky_relu", sharing="module"),
}
STANDARD = ("relu", "gelu")
ADAPTIVE = ("prelu", "acon_c")
CONTROLS = ("apu_frozen", "apu_const")
CORE_METHODS = ("relu", "gelu", "prelu", "apu_frozen", "apu_const", "apu", "acon_c")


def make_activation(spec, num_channels, channel_dim=1):
    k = spec["kind"]
    if k == "relu": m = nn.ReLU()
    elif k == "gelu": m = nn.GELU()
    elif k == "silu": m = nn.SiLU()
    elif k == "mish": m = nn.Mish()
    elif k == "leakyrelu": m = nn.LeakyReLU(spec["negative_slope"])
    elif k == "prelu":
        if spec.get("sharing", "module") != "module":
            raise ValueError("only module-shared PReLU is used")
        m = nn.PReLU(init=spec["init"])
    elif k == "esrelu": m = ESReLU(spec["alpha"], spec["gamma"])
    elif k == "apu":
        args = {kk: spec[kk] for kk in ("mode", "p0", "a_max", "init_A", "init_wz", "init_bz", "p_min", "p_max", "eps")}
        m = APU(**args, num_channels=(num_channels if spec["sharing"] == "channel" else None), channel_dim=channel_dim,
                trainable=spec["trainable"])
    elif k == "acon_c": m = ACONC(num_channels, channel_dim)
    elif k == "pade": m = PadeActivation(spec["m"], spec["n"], spec["target"])
    else:
        raise ValueError(f"unknown activation kind {k}")
    m._is_act = True; m._act_spec = spec; m._cd = channel_dim
    return m


def act_modules(model):
    return [(n, m) for n, m in model.named_modules() if getattr(m, "_is_act", False)]


def activation_param_names(model):
    names = set()
    for n, m in act_modules(model):
        for pn, _ in m.named_parameters(recurse=False):
            names.add(f"{n}.{pn}" if n else pn)
    return names


def init_activation_params(model, seed):
    """Activation-specific initialisation with its own keyed stream (never shifts backbone initialisation)."""
    for n, m in act_modules(model):
        if hasattr(m, "reset_parameters_keyed"):
            m.reset_parameters_keyed(seed, n)


def count_params(model):
    act_names = activation_param_names(model)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    act_all = sum(p.numel() for n, p in model.named_parameters() if n in act_names)
    act_trainable = sum(p.numel() for n, p in model.named_parameters() if n in act_names and p.requires_grad)
    return dict(total=total, trainable=trainable, activation_total=act_all, activation_trainable=act_trainable,
                backbone=total - act_all, n_activation_modules=len(act_modules(model)))
