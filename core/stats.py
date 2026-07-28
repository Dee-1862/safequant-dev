# Exact McNemar, Wilson CIs, and paired bootstrap recovery intervals.

#Front Matter
import math

import numpy as np


def mcnemar_exact(flags_a, flags_b):
    """Exact two-sided McNemar on paired binary flags (1 = complied)."""
    # Count discordant pairs (b: A yes B no, c: A no B yes)
    b = 0
    c = 0
    for x, y in zip(flags_a, flags_b):
        if x and not y:
            b += 1
        elif not x and y:
            c += 1
    nbc = b + c

    # Two-sided exact binomial under p = 0.5
    if nbc == 0:
        p = 1.0
    else:
        k = min(b, c)
        p = 2.0 * sum(math.comb(nbc, i) for i in range(k + 1)) * (0.5 ** nbc)
        p = min(1.0, p)
    return {"b": b, "c": c, "n_discordant": nbc, "p_value": p}


def paired_delta_asr(fp_flags, quant_flags, n_boot = 5000, seed = 0):
    """Paired ΔASR estimator with the correct (McNemar-style) standard error.
    Flags are per-prompt compliance labels on the SAME prompts (1 = complied with the harmful request = attack success, 0 = refused).
    Because the two variants see identical prompts, only discordant pairs carry information.
    b = #(fp refused, quant complied), quantization-induced compliance.
    c = #(fp complied, quant refused).
    ΔASR = 100 (b - c) / n == quant_asr - fp_asr.
    The paired SE uses the multinomial variance of that difference, not the p(1-p)/n form (which assumes independent samples and is conservative here).
    SE = 100/n * sqrt( (b + c) - (b - c)^2 / n ).
    Returns p_mcnemar, ci95 (paired bootstrap), fp_asr, quant_asr."""
    a = np.asarray(fp_flags, dtype = int)
    q = np.asarray(quant_flags, dtype = int)
    if a.shape != q.shape:
        raise ValueError(f"flag length mismatch: fp {a.shape} vs quant {q.shape}")
    n = int(a.size)
    if n == 0:
        raise ValueError("empty flag arrays")

    b = int(np.sum((a == 0) & (q == 1)))   # fp refused, quant complied (damage)
    c = int(np.sum((a == 1) & (q == 0)))   # fp complied, quant refused
    bc = b + c
    delta = 100.0 * (b - c) / n

    se_paired = 100.0 / n * math.sqrt(max(0.0, bc - (b - c) ** 2 / n))
    se_null = 100.0 * math.sqrt(bc) / n
    p_fp, p_q = float(a.mean()), float(q.mean())
    se_unpaired = 100.0 * math.sqrt(p_fp * (1 - p_fp) / n + p_q * (1 - p_q) / n)

    p_mc = mcnemar_exact(a.tolist(), q.tolist())["p_value"]

    # Paired bootstrap over prompt pairs (resample the +1/-1/0 per-prompt diffs).
    rng = np.random.default_rng(seed)
    diffs = q - a
    boot = np.empty(n_boot)
    for i in range(n_boot):
        boot[i] = 100.0 * diffs[rng.integers(0, n, n)].mean()
    ci = [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]

    return {
        "b": b, "c": c, "n": n, "delta_asr": delta,
        "se_paired": se_paired, "se_null": se_null, "se_unpaired": se_unpaired,
        "p_mcnemar": p_mc, "ci95": ci,
        "fp_asr": 100.0 * p_fp, "quant_asr": 100.0 * p_q,
    }


def holm_bonferroni(pvalues):
    """Holm step-down adjusted p-values (strong family-wise error control)."""
    m = len(pvalues)
    if m == 0:
        return []
    order = sorted(range(m), key = lambda i: pvalues[i])
    adj = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        # (m - rank) multiplier, enforced monotonic non-decreasing down the ladder
        running = max(running, (m - rank) * pvalues[i])
        adj[i] = min(1.0, running)
    return adj


def wilson_ci(k, n, z = 1.96):
    """Wilson score 95% CI for a proportion k/n."""
    # Empty trials -> degenerate interval
    if n == 0:
        return [0.0, 0.0]
    p = k / n

    # Wilson center and half-width
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [max(0.0, center - half), min(1.0, center + half)]


def bootstrap_recovery(flags_by_cond, qtz, n_boot = 2000, seed = 0):
    """Paired bootstrap CIs on read/write/both recovery percentages."""
    # Seeded RNG and float arrays
    rng = np.random.default_rng(seed)
    arrs = {c: np.asarray(v, dtype = float) for c, v in flags_by_cond.items()}
    n = len(next(iter(arrs.values())))
    rec = {"read": [], "write": [], "both": [], "read_minus_write": []}

    # Resample and accumulate recovery percentages
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        asr = {c: arrs[c][idx].mean() * 100 for c in arrs}
        dmg = asr[qtz] - asr["fp32"]
        if dmg == 0:
            continue
        rr = (asr[qtz] - asr["read_restored"]) / dmg * 100
        rw = (asr[qtz] - asr["write_restored"]) / dmg * 100
        rb = (asr[qtz] - asr["both_restored"]) / dmg * 100
        rec["read"].append(rr)
        rec["write"].append(rw)
        rec["both"].append(rb)
        rec["read_minus_write"].append(rr - rw)

    def ci(a):
        """2.5 / 97.5 percentile interval for a bootstrap sample list."""
        a = np.asarray(a)
        return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]

    # Replace sample lists with percentile CIs
    for k, v in rec.items():
        rec[k] = ci(v)
    return rec
