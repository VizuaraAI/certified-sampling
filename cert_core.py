"""Head variants, error bounds and candidate sets for Exp01 of topic 07. PyTorch only, so it runs on a laptop CPU or a GPU.

A certificate has a low-bit head and a bound. For a hidden state h, row v gets an interval [L_v, U_v] that must hold the
reference logit. A sampling rule then names the rows it cannot exclude from those intervals alone. That set R is what a
BF16 recompute has to read.

    python3 -I cert_core.py     runs the self tests on the CPU (see selftest below)
"""

import math

import torch

U32 = 2.0 ** -24                                # unit roundoff of FP32
HOEFF_DELTA = 1e-15
HOEFF = math.sqrt(math.log(2.0 / HOEFF_DELTA) / 2.0)   # about 4.197
GROUP = 128
ABS_SLACK = 1e-4                                # for FP32 rounding of the interval ends themselves
INFLATE = 1.0 + 1e-5                            # for FP32 rounding in the sums that form each bound
A_INT4, A_INT4_DITHER, A_INT8 = 7.0, 6.5, 127.0
Q_INT4, Q_INT8 = 7, 127
RULES = [
    dict(name="greedy", kind="greedy"),
    dict(name="gumbel_t0.6", kind="gumbel", T=0.6),
    dict(name="gumbel_t1.0", kind="gumbel", T=1.0),
    dict(name="topk50_t1.0", kind="topk", T=1.0, k=50),
    dict(name="minp0.05_t1.0", kind="minp", T=1.0, p=0.05),
]
RULE = {r["name"]: r for r in RULES}


# ------------------------------------------------------------------ numbers as stored

def up16(x):
    """Round up to an FP16 value: a stored scale or bound is never smaller than the number it stands for."""
    return (x * (1.0 + 2.0 ** -10) + 6e-8).half().float()


def e4m3_table(dev):
    vals = [m / 8.0 * 2.0 ** -6 for m in range(1, 8)]
    for e in range(-6, 9):
        for m in range(8):
            if e == 8 and m == 7:
                continue
            vals.append((1.0 + m / 8.0) * 2.0 ** e)
    return torch.tensor(sorted(vals), device=dev, dtype=torch.float32)


# ------------------------------------------------------------------ head variants

def _meta(W_view, what_view):
    res = W_view - what_view
    return up16(res.abs().amax(2)), up16(res.norm(dim=2))


def int_head(W, A, qmax, group, dither=None):
    """Symmetric integer head with FP16 scales per group. With dither (values in [-1/2, 1/2)) it is subtractive."""
    V, d = W.shape
    G = d // group
    X = W.view(V, G, group)
    s = up16(X.abs().amax(2, keepdim=True) / A)
    if dither is None:
        q = torch.round(X / s)
        what = q * s
    else:
        D = dither.view(1, G, group)
        q = torch.round(X / s + D)
        what = (q - D) * s
    assert q.abs().max().item() <= qmax, "a code fell outside the range"
    e, r = _meta(X, what)
    return dict(W=what.view(V, d), s=s.squeeze(2), e=e, r=r, group=group)


def nvfp4_head(W, group=GROUP):
    """NVFP4 emulation: E2M1 values, E4M3 block scales per 16 elements, one FP32 tensor scale."""
    V, d = W.shape
    dev = W.device
    table = e4m3_table(dev)
    grid = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=dev)
    mids = (grid[1:] + grid[:-1]) / 2.0
    ts = W.abs().max() / (6.0 * 448.0)
    X = W.view(V, d // 16, 16)
    want = (X.abs().amax(2, keepdim=True) / (6.0 * ts)).clamp(max=448.0)
    sb = table[torch.searchsorted(table, want).clamp(max=len(table) - 1)] * ts
    mag = (X.abs() / sb).clamp(max=6.0)
    what = torch.sign(X) * grid[torch.bucketize(mag, mids)] * sb
    G = d // group
    e, r = _meta(W.view(V, G, group), what.view(V, G, group))
    return dict(W=what.view(V, d), s=None, e=e, r=r, group=group)


def hadamard(n, dev):
    H = torch.ones(1, 1, device=dev, dtype=torch.float32)
    while H.shape[0] < n:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return H / math.sqrt(n)


class Rotation:
    """Random permutation, random signs, then a Hadamard transform on blocks of 512. Orthogonal, so dot products keep."""

    def __init__(self, d, seed, dev, block=512):
        assert d % block == 0
        g = torch.Generator().manual_seed(seed)
        self.perm = torch.randperm(d, generator=g).to(dev)
        self.sign = (torch.randint(0, 2, (d,), generator=g).float() * 2 - 1).to(dev)
        self.H = hadamard(block, dev)
        self.block, self.d = block, d

    def __call__(self, x):
        n = x.shape[0]
        y = (x[:, self.perm] * self.sign).view(n, self.d // self.block, self.block)
        return (y @ self.H).view(n, self.d)


def dither_pattern(d, seed, dev):
    g = torch.Generator().manual_seed(seed)
    return (torch.rand(d, generator=g, dtype=torch.float64) - 0.5).float().to(dev)


# ------------------------------------------------------------------ bounds, N states by V rows

def group_norms(H, group):
    N, d = H.shape
    X = H.view(N, d // group, group)
    return X.abs().sum(2), X.norm(dim=2)


def bound_worst(head, H1):
    return (head["e"] @ H1.T).T * INFLATE


def bound_resid(head, H2):
    return (head["r"] @ H2.T).T * INFLATE


def bound_dither(head, H2):
    return HOEFF * torch.sqrt(((head["s"] * head["s"]) @ (H2 * H2).T).T) * INFLATE


def predicted_sd(head, H2):
    """Standard deviation of the dithered error if each rounding error is uniform on [-s/2, s/2]."""
    return torch.sqrt(((head["s"] * head["s"]) @ (H2 * H2).T).T / 12.0)


def fp_slack(H, rownorm):
    """Rounding of FP32 sums, in the reference and in the first pass: at most about d u sum |w_j h_j| each."""
    d = H.shape[1]
    return 3.0 * d * U32 * torch.outer(H.norm(dim=1), rownorm)


def phi(x, cap):
    return x if cap is None else cap * torch.tanh(x / cap)


def interval(approx, b, cap):
    return phi(approx - b, cap) - ABS_SLACK, phi(approx + b, cap) + ABS_SLACK


# ------------------------------------------------------------------ candidate sets and tokens

def rule_set(L, U, rule, g):
    """Rows the bounds cannot exclude. Returns the mask R, and what the token step needs to finish the decision."""
    kind = rule["kind"]
    if kind == "greedy":
        return U >= L.amax(1, keepdim=True), None
    T = rule["T"]
    zL, zU = L / T + g, U / T + g
    if kind == "gumbel":
        return zU >= zL.amax(1, keepdim=True), None
    if kind == "topk":
        k = rule["k"]
        Lk = L.topk(k, dim=1).values[:, -1:]
        Uk1 = U.topk(k + 1, dim=1).values[:, -1:]
        C, D = U >= Lk, L > Uk1
        G = None
    else:
        th = T * math.log(rule["p"])
        maxL, maxU = L.amax(1, keepdim=True), U.amax(1, keepdim=True)
        C, D = U >= maxL + th, L >= maxU + th
        G = U >= maxL
    best = torch.where(D, zL, -math.inf).amax(1, keepdim=True)
    race = C & (zU >= best)
    amb = (race & ~D).any(1, keepdim=True)
    R = race | (amb & C & ~D) if kind == "topk" else race | (amb & G)
    return R, dict(C=C, D=D, G=G)


def ref_token(ref, rule, g):
    kind = rule["kind"]
    if kind == "greedy":
        return ref.argmax(1)
    T = rule["T"]
    z = ref / T + g
    if kind == "gumbel":
        return z.argmax(1)
    if kind == "topk":
        m = torch.zeros_like(ref, dtype=torch.bool).scatter(1, ref.topk(rule["k"], dim=1).indices, True)
    else:
        m = ref >= ref.amax(1, keepdim=True) + T * math.log(rule["p"])
    return torch.where(m, z, -math.inf).argmax(1)


def cert_token(ref, R, aux, rule, g):
    """Finish the decision reading exact values only where the rule allows it, and return the token."""
    kind = rule["kind"]
    if kind == "greedy":
        return torch.where(R, ref, -math.inf).argmax(1)
    T = rule["T"]
    z = ref / T + g
    if kind == "gumbel":
        return torch.where(R, z, -math.inf).argmax(1)
    C, D, G = aux["C"], aux["D"], aux["G"]
    if kind == "topk":
        y = torch.where(D, math.inf, torch.where(C, ref, -math.inf))
        m = torch.zeros_like(ref, dtype=torch.bool).scatter(1, y.topk(rule["k"], dim=1).indices, True)
    else:
        top = torch.where(G, ref, -math.inf).amax(1, keepdim=True)
        m = D | (C & (ref >= top + T * math.log(rule["p"])))
    return torch.where(R & m, z, -math.inf).argmax(1)


def refine(L, U, S, L8, U8):
    """Cascade: rows on the 4-bit shortlist S take the intersection with their INT8 interval."""
    return torch.where(S, torch.maximum(L, L8), L), torch.where(S, torch.minimum(U, U8), U)


# ------------------------------------------------------------------ counter-based Gumbel noise

GOLDEN, MIX1, MIX2 = -7046029254386353131, -4658895280553007687, -7723592293110705685   # SplitMix64 constants as int64


def _mix(z):
    z = (z ^ ((z >> 30) & ((1 << 34) - 1))) * MIX1
    z = (z ^ ((z >> 27) & ((1 << 37) - 1))) * MIX2
    return z ^ ((z >> 31) & ((1 << 33) - 1))


def state_key(prompt, step, mode):
    return prompt * (1 << 20) + step * 2 + mode


def gumbel(keys, V, dev):
    """Noise for (key, token id): N keys by V ids. The same key and id give the same noise on any machine."""
    seed = _mix(keys.to(dev).long() * GOLDEN + GOLDEN)
    ids = torch.arange(1, V + 1, device=dev, dtype=torch.int64)
    z = _mix(seed.view(-1, 1) + ids.view(1, -1) * GOLDEN)
    u = (((z >> 40) & ((1 << 23) - 1)).float() + 0.5) / float(1 << 23)
    return -torch.log(-torch.log(u))


def _mix_py(z):
    M = (1 << 64) - 1
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & M
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & M
    return z ^ (z >> 31)


# ------------------------------------------------------------------ byte model, as fractions of the BF16 head

def first_pass_bytes(head, d):
    if head == "int8":
        return 0.5 + 1.0 / d
    if head == "nvfp4":
        return 0.25 + 8.0 / (16 * 16)
    return 0.25 + 1.0 / GROUP


def bound_meta_bytes(head, bound, d):
    if bound in ("oracle", "dither"):
        return 0.0
    return 1.0 / d if head == "int8" else 1.0 / GROUP


def total_bytes(head, bound, d, V, R, S=None):
    """First pass, bound metadata, one row norm per row, INT8 rows on the shortlist, BF16 rows in R."""
    x = first_pass_bytes(head, d) + bound_meta_bytes(head, bound, d) + (0.0 if bound == "oracle" else 1.0 / d)
    if S is not None:
        x = x + S / V * (0.5 + 2.0 / d)
    return x + R / V


# ------------------------------------------------------------------ self tests

def naive_set(L, U, rule, g):
    """One state, plain Python, written from the definitions and independent of rule_set."""
    V = len(L)
    kind = rule["kind"]
    if kind == "greedy":
        m = max(L)
        return {v for v in range(V) if U[v] >= m}
    T = rule["T"]
    zL, zU = [L[v] / T + g[v] for v in range(V)], [U[v] / T + g[v] for v in range(V)]
    if kind == "gumbel":
        m = max(zL)
        return {v for v in range(V) if zU[v] >= m}
    if kind == "topk":
        k = rule["k"]
        Lk, Uk1 = sorted(L, reverse=True)[k - 1], sorted(U, reverse=True)[k]
        C = {v for v in range(V) if U[v] >= Lk}
        D = {v for v in range(V) if L[v] > Uk1}
    else:
        th = T * math.log(rule["p"])
        C = {v for v in range(V) if U[v] >= max(L) + th}
        D = {v for v in range(V) if L[v] >= max(U) + th}
    best = max((zL[v] for v in D), default=-math.inf)
    race = {v for v in C if zU[v] >= best}
    if race <= D:
        return race
    if kind == "topk":
        return race | (C - D)
    return race | {v for v in range(V) if U[v] >= max(L)}


def selftest(dev="cpu"):
    torch.manual_seed(7)
    # Known byte fractions from the review.
    assert abs(first_pass_bytes("int4", 4096) - 0.2578) < 1e-3 and abs(first_pass_bytes("nvfp4", 4096) - 0.28125) < 1e-9
    assert abs(HOEFF - 4.197) < 1e-3
    # Noise matches a plain Python SplitMix64 on the same key and id.
    keys = torch.tensor([state_key(3, 5, 1), state_key(0, 0, 0)], device=dev)
    z = gumbel(keys, 40, dev)
    for i, k in enumerate(keys.tolist()):
        seed = _mix_py((k * (GOLDEN & (2 ** 64 - 1)) + (GOLDEN & (2 ** 64 - 1))) & (2 ** 64 - 1))
        for t in (0, 17, 39):
            x = _mix_py((seed + (t + 1) * (GOLDEN & (2 ** 64 - 1))) & (2 ** 64 - 1))
            u = ((x >> 40 & ((1 << 23) - 1)) + 0.5) / (1 << 23)
            assert abs(-math.log(-math.log(u)) - z[i, t].item()) < 1e-5
    # Gumbel noise has mean 0.5772 and variance pi^2/6.
    big = gumbel(torch.arange(200, device=dev), 5000, dev)
    assert abs(big.mean().item() - 0.5772) < 0.01 and abs(big.var().item() - math.pi ** 2 / 6) < 0.05
    # The rotation is orthogonal.
    rot = Rotation(1024, 11, dev)
    a, b = torch.randn(30, 1024, device=dev), torch.randn(20, 1024, device=dev)
    assert (a @ b.T - rot(a) @ rot(b).T).abs().max().item() < 2e-3
    # Heads and bounds on synthetic heavy-tailed data. d = 1024 so all groupings divide.
    V, d, N = 3000, 1024, 40
    W = torch.randn(V, d, device=dev) * 0.02 * torch.exp(torch.randn(V, d, device=dev) * 0.5)
    H = torch.randn(N, d, device=dev) * torch.exp(torch.randn(N, d, device=dev))
    ref = H @ W.T
    rownorm = W.norm(dim=1)
    tau = fp_slack(H, rownorm)
    D = dither_pattern(d, 5, dev)
    heads = dict(int8=int_head(W, A_INT8, Q_INT8, d), int4=int_head(W, A_INT4, Q_INT4, GROUP),
                 int4d=int_head(W, A_INT4_DITHER, Q_INT4, GROUP, D), nvfp4=nvfp4_head(W))
    rot = Rotation(d, 11, dev)
    heads["int4h"] = int_head(rot(W), A_INT4, Q_INT4, GROUP)
    g = gumbel(torch.arange(N, device=dev), V, dev)
    for name, hd in heads.items():
        Hq = rot(H) if name == "int4h" else H
        H1, H2 = group_norms(Hq, hd["group"])
        err = (Hq @ hd["W"].T - ref).abs()
        bw, br = bound_worst(hd, H1), bound_resid(hd, H2)
        assert (err <= bw + tau).all() and (err <= br + tau).all(), name
        if name == "int4d":
            sd = predicted_sd(hd, H2)
            ratio = (((Hq @ hd["W"].T - ref) / sd) ** 2).mean().item()
            assert 0.9 < ratio < 1.1, f"dither variance ratio {ratio}"
            assert (err <= bound_dither(hd, H2) + tau).all()
        # The planted bug must be caught: two percent of the bound does not hold.
        assert (err > 0.02 * br).any(), name
    # The vectorised sets match the plain definitions, the winner is always inside, and the token is exact.
    approx = H @ heads["int4"]["W"].T
    b = bound_resid(heads["int4"], group_norms(H, GROUP)[1]) + tau
    for cap in (None, 30.0):
        L, U = interval(approx, b, cap)
        rf = phi(ref, cap)
        for rule in RULES:
            R, aux = rule_set(L, U, rule, g)
            for i in range(N):
                n = naive_set(L[i].tolist(), U[i].tolist(), rule, g[i].tolist())
                assert n == set(torch.nonzero(R[i]).flatten().tolist()), (rule["name"], i)
            assert (cert_token(rf, R, aux, rule, g) == ref_token(rf, rule, g)).all(), rule["name"]
    # Narrow bounds, so that definite sets exist, also give exact tokens, and tighter bounds give smaller sets.
    for scale in (0.002, 0.02, 0.2):
        ap = ref + 0.5 * scale * b * (2 * torch.rand_like(ref) - 1)
        L, U = interval(ap, scale * b, None)
        for rule in RULES:
            R, aux = rule_set(L, U, rule, g)
            assert (cert_token(ref, R, aux, rule, g) == ref_token(ref, rule, g)).all(), (scale, rule["name"])
            for i in range(N):
                n = naive_set(L[i].tolist(), U[i].tolist(), rule, g[i].tolist())
                assert n == set(torch.nonzero(R[i]).flatten().tolist()), (scale, rule["name"], i)
    # A cascade only ever shrinks the set.
    L, U = interval(approx, b, None)
    L8, U8 = interval(H @ heads["int8"]["W"].T, bound_resid(heads["int8"], group_norms(H, d)[1]) + tau, None)
    for rule in RULES:
        S, _ = rule_set(L, U, rule, g)
        L2, U2 = refine(L, U, S, L8, U8)
        R2, aux = rule_set(L2, U2, rule, g)
        assert not (R2 & ~S).any(), rule["name"]
        assert (cert_token(ref, R2, aux, rule, g) == ref_token(ref, rule, g)).all(), rule["name"]
    # Exact and empty heads bracket the measure.
    zero = torch.zeros_like(ref)
    Lz, Uz = interval(zero, torch.outer(H.norm(dim=1), rownorm) + tau, None)
    assert rule_set(Lz, Uz, RULES[0], g)[0].float().mean().item() > 0.2
    Le, Ue = interval(ref, 0 * ref, None)
    assert (rule_set(Le, Ue, RULES[0], g)[0].sum(1) == 1).float().mean().item() > 0.99
    # A planted fault in the exactness check: a set that drops the winner must give a wrong token.
    R, aux = rule_set(L, U, RULES[0], g)
    R2 = R.clone()
    R2.scatter_(1, ref.argmax(1, keepdim=True), False)
    assert (cert_token(ref, R2, aux, RULES[0], g) != ref_token(ref, RULES[0], g)).all()
    print("selftest passed")
    return True


if __name__ == "__main__":
    selftest()
