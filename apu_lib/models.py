"""Models. The activation replaces the baseline nonlinearity; nothing else changes between methods.

Backbone initialisation is done after construction by `init_backbone(model, seed)`: every backbone module draws from
its own generator keyed by (seed, 'init', module name). Construction order, activation parameters and the global
random state therefore cannot change the backbone weights, which makes seed-paired comparisons possible.
The init rules reproduce the historical ones: PyTorch defaults for SimpleCNN and Linear layers, Kaiming-normal
(fan_out, ReLU gain) for ResNet convolutions, truncated normal (std 0.02) for ViT linear layers and embeddings.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import key_seed, digest_of
from .activations import make_activation, act_modules, activation_param_names, init_activation_params


def _gen(*parts):
    g = torch.Generator(); g.manual_seed(key_seed(*parts)); return g


def _uniform_(t, bound, g):
    with torch.no_grad():
        t.copy_((torch.rand(t.shape, generator=g, dtype=torch.float64) * 2 - 1).mul_(bound).to(t.dtype))


def _normal_(t, std, g, trunc=None):
    with torch.no_grad():
        v = torch.randn(t.shape, generator=g, dtype=torch.float64) * std
        if trunc is not None:
            v = v.clamp(-trunc, trunc)
        t.copy_(v.to(t.dtype))


class SimpleCNN(nn.Module):
    """Historical layout: conv-act, conv-act, [extra 3x3 same-padding conv-act blocks], max-pool, fc-act, fc.
    depth = number of conv layers (2 = the original model)."""
    init_rule = "pytorch_default"
    def __init__(self, act, meta, depth=2):
        super().__init__()
        c, img, nc = meta["in_ch"], meta["img"], meta["num_classes"]
        self.conv1, self.conv2 = nn.Conv2d(c, 32, 3, 1), nn.Conv2d(32, 64, 3, 1)
        self.a1, self.a2 = act(32, 1), act(64, 1)
        self.extra = nn.ModuleList([nn.Conv2d(64, 64, 3, 1, 1) for _ in range(depth - 2)])
        self.extra_act = nn.ModuleList([act(64, 1) for _ in range(depth - 2)])
        feat = 64 * ((img - 4) // 2) ** 2
        self.fc1 = nn.Linear(feat, 128); self.a3 = act(128, 1); self.fc2 = nn.Linear(128, nc)
    def forward(self, x):
        x = self.a2(self.conv2(self.a1(self.conv1(x))))
        for conv, a in zip(self.extra, self.extra_act):
            x = a(conv(x))
        x = torch.flatten(F.max_pool2d(x, 2), 1)
        return self.fc2(self.a3(self.fc1(x)))


class BasicBlock(nn.Module):
    def __init__(self, inp, planes, stride, act):
        super().__init__()
        self.conv1 = nn.Conv2d(inp, planes, 3, stride, 1, bias=False); self.bn1 = nn.BatchNorm2d(planes); self.act1 = act(planes, 1)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False); self.bn2 = nn.BatchNorm2d(planes); self.act2 = act(planes, 1)
        self.shortcut = nn.Sequential()
        if stride != 1 or inp != planes:
            self.shortcut = nn.Sequential(nn.Conv2d(inp, planes, 1, stride, bias=False), nn.BatchNorm2d(planes))
    def forward(self, x):
        out = self.act1(self.bn1(self.conv1(x)))
        return self.act2(self.bn2(self.conv2(out)) + self.shortcut(x))


class ResNet18(nn.Module):
    init_rule = "resnet_kaiming_fan_out"
    def __init__(self, act, meta, width=64):
        super().__init__()
        w = width; self.inp = w
        self.stem = nn.Sequential(nn.Conv2d(meta["in_ch"], w, 3, 1, 1, bias=False), nn.BatchNorm2d(w), act(w, 1))
        self.layer1 = self._make(w, 1, act); self.layer2 = self._make(w * 2, 2, act)
        self.layer3 = self._make(w * 4, 2, act); self.layer4 = self._make(w * 8, 2, act)
        self.fc = nn.Linear(w * 8, meta["num_classes"])
    def _make(self, planes, stride, act):
        layers = []
        for s in (stride, 1):
            layers.append(BasicBlock(self.inp, planes, s, act)); self.inp = planes
        return nn.Sequential(*layers)
    def forward(self, x):
        x = self.layer4(self.layer3(self.layer2(self.layer1(self.stem(x)))))
        return self.fc(torch.flatten(F.adaptive_avg_pool2d(x, 1), 1))


class Attention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__(); self.h = heads; self.qkv = nn.Linear(dim, dim * 3); self.proj = nn.Linear(dim, dim)
    def forward(self, x):
        B, N, D = x.shape
        q, k, v = self.qkv(x).reshape(B, N, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        return self.proj(F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(B, N, D))


class ViTBlock(nn.Module):
    def __init__(self, dim, heads, act, mlp_ratio=4):
        super().__init__()
        self.ln1, self.attn, self.ln2 = nn.LayerNorm(dim), Attention(dim, heads), nn.LayerNorm(dim)
        self.fc1, self.act, self.fc2 = nn.Linear(dim, dim * mlp_ratio), act(dim * mlp_ratio, -1), nn.Linear(dim * mlp_ratio, dim)
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.fc2(self.act(self.fc1(self.ln2(x))))


class ViT(nn.Module):
    init_rule = "vit_trunc_normal_0.02"
    def __init__(self, act, meta, dim=192, depth=12, heads=3):
        super().__init__()
        p = meta["patch"]; n = (meta["img"] // p) ** 2
        self.patch = nn.Conv2d(meta["in_ch"], dim, p, p)
        self.cls = nn.Parameter(torch.zeros(1, 1, dim)); self.pos = nn.Parameter(torch.zeros(1, n + 1, dim))
        self.blocks = nn.ModuleList([ViTBlock(dim, heads, act) for _ in range(depth)])
        self.norm, self.head = nn.LayerNorm(dim), nn.Linear(dim, meta["num_classes"])
    def forward(self, x):
        x = self.patch(x).flatten(2).transpose(1, 2)
        x = torch.cat([self.cls.expand(x.shape[0], -1, -1), x], 1) + self.pos
        for b in self.blocks:
            x = b(x)
        return self.head(self.norm(x)[:, 0])


def build_model(model_cfg, act_spec, meta):
    act = lambda C, cd: make_activation(act_spec, C, cd)
    arch = model_cfg["arch"]
    if arch == "simplecnn":
        return SimpleCNN(act, meta, depth=model_cfg.get("depth", 2))
    if arch == "resnet18":
        return ResNet18(act, meta, width=model_cfg.get("width", 64))
    if arch == "vit_tiny":
        return ViT(act, meta, dim=192, depth=12, heads=3)
    raise ValueError(arch)


def init_backbone(model, seed):
    """Re-initialise every backbone parameter and buffer from keyed generators. Returns the set of initialised names."""
    rule = getattr(model, "init_rule", "pytorch_default")
    act_mod_names = {n for n, _ in act_modules(model)}
    done = set()
    for name, m in model.named_modules():
        if name in act_mod_names or any(name.startswith(a + ".") for a in act_mod_names if a):
            continue
        g = _gen(seed, "init", name)
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            fan_in = m.weight[0].numel()
            if isinstance(m, nn.Conv2d) and rule == "resnet_kaiming_fan_out":
                fan_out = m.weight.shape[0] * m.weight[0][0].numel()
                _normal_(m.weight, math.sqrt(2.0 / fan_out), g)
            elif isinstance(m, nn.Linear) and rule == "vit_trunc_normal_0.02":
                _normal_(m.weight, 0.02, g, trunc=2.0)
            else:                                            # PyTorch default (kaiming_uniform a=sqrt(5))
                _uniform_(m.weight, 1.0 / math.sqrt(fan_in), g)
            done.add(f"{name}.weight")
            if m.bias is not None:
                if isinstance(m, nn.Linear) and rule == "vit_trunc_normal_0.02":
                    with torch.no_grad(): m.bias.zero_()
                else:
                    _uniform_(m.bias, 1.0 / math.sqrt(fan_in), g)
                done.add(f"{name}.bias")
        elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d, nn.LayerNorm)):
            with torch.no_grad():
                if m.weight is not None: m.weight.fill_(1.0); done.add(f"{name}.weight")
                if m.bias is not None: m.bias.zero_(); done.add(f"{name}.bias")
                if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                    m.running_mean.zero_(); m.running_var.fill_(1.0); m.num_batches_tracked.zero_()
    if isinstance(model, ViT):
        _normal_(model.pos, 0.02, _gen(seed, "init", "pos"), trunc=2.0); _normal_(model.cls, 0.02, _gen(seed, "init", "cls"), trunc=2.0)
        done |= {"pos", "cls"}
    act_names = activation_param_names(model)
    missing = {n for n, _ in model.named_parameters()} - act_names - done
    if missing:
        raise RuntimeError(f"backbone parameters without keyed initialisation: {sorted(missing)}")
    return done


def create_model(model_cfg, act_spec, meta, seed):
    """Build, initialise backbone and activation parameters from independent keyed streams, apply freezing."""
    model = build_model(model_cfg, act_spec, meta)
    init_backbone(model, seed)
    init_activation_params(model, seed)
    return model


def backbone_state(model):
    act_names = activation_param_names(model)
    return {k: v for k, v in model.state_dict().items() if k not in act_names}


def backbone_digest(model):
    import hashlib
    h = hashlib.sha256()
    for k, v in sorted(backbone_state(model).items()):
        h.update(k.encode()); h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
