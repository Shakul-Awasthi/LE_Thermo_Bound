"""crn_cost.py -- numba engine for Figure 2 (thermodynamic-metric bound only).

Networks: WR, Brusselator, Oregonator, CS14, each reaction completed with its own
fuel/waste reservoir pair (any positive rate vector obeys local detailed balance).
Mixing: ideal (Gamma = 0) or nonideal activities a_i = x_i exp[(Gamma x)_i],
thermodynamic metric H = diag(1/x) + Gamma.

For one input the engine integrates the state (log concentrations) and the full
tangent matrix with an adaptive Dormand-Prince 5(4) scheme, orthonormalises the
tangent after every accepted step, and integrates along the leading tangent the
terms of the bound |lambda_1 + Delta_1| <= sqrt(sigma_ps B_1):
    sigma, sigma_ps, Delta, B, the directly differentiated Hessian stretching rate,
    and tr J (for the Liouville audit).
Classification: stable fixed point (Newton + spectrum), limit cycle (Poincare
recurrence, Newton shooting, Floquet multipliers), chaos candidate (two windows),
otherwise unresolved. Reported time = time_unit x raw time (rates divided by it).
"""
import time
from dataclasses import dataclass, asdict
import numpy as np
from numba import njit

VERSION = '4.0-numba'
# name: (reactant nu', product nu'', reference k (k+1,k-1,...), time unit, reference x, Gamma scale)
MODELS = {
    'wr': ([[1, 1, 0, 1, 0], [0, 1, 1, 0, 0], [0, 0, 0, 1, 1]],
           [[2, 0, 0, 0, 0], [0, 2, 0, 0, 0], [0, 0, 0, 0, 2]],
           [30, .25, 1, 1e-4, 10, 1e-3, 1, 1e-4, 16.5, .5], 30., [10, 10, 10], .08),
    'brusselator': ([[0, 2, 1, 1], [0, 1, 0, 0]], [[1, 3, 0, 0], [0, 0, 1, 0]],
                    [1, .001, 1, .001, 3, .001, 1, .001], 1., [1, 2], .3),
    'oregonator': ([[0, 1, 1, 2, 0], [1, 1, 0, 0, 0], [0, 0, 0, 0, 1]],
                   [[1, 0, 2, 0, 0], [0, 0, 0, 0, 1], [0, 0, 1, 0, 0]],
                   [1, .001, 40, .001, 1, .001, .2, .001, .1, .001], 1., [1, 1, 1], .3),
    'cs14': ([[0, 2, 1, 0, 1, 0], [1, 0, 0, 0, 1, 0], [0, 0, 1, 1, 0, 0]],
             [[1, 3, 0, 0, 1, 0], [1, 0, 0, 1, 0, 0], [0, 0, 2, 0, 0, 1]],
             [.2, .001, 1, .02, 1, .001, 1, .001, 1, .001, .01, .001], 1., [1, 1, 1], .2),
}
OBS = ('sigma', 'sigma_ps', 'Delta1', 'B1', 'stretch_H', 'trace_J')
NOB = 6


@dataclass
class Config:
    transient: float = 600.
    duration: float = 1600.
    blocks: int = 6
    rtol: float = 1e-8
    atol: float = 1e-11
    step_budget: int = 3_000_000     # Dormand-Prince steps per input (all windows); stiffer inputs are rejected
    confirmation: bool = True
    fp_residual: float = 1e-9
    fp_distance: float = 1e-5
    recurrence_tol: float = .03
    shooting_tol: float = 2e-8
    floquet_tol: float = 2e-4
    chaos_floor: float = 1e-4
    chaos_relative: float = .25
    n_record: int = 8192


# ------------------------------------------------------------ parameters ----
def reference(model='wr', nonideal=False):
    a, b, k, tu, x, g = MODELS[model]; n = len(x)
    return dict(model=model, k=list(k), Gamma=(np.eye(n) * g if nonideal else np.zeros((n, n))).tolist(),
                x0=list(x), d0=(np.ones(n) / np.sqrt(n)).tolist(), time_unit=tu,
                nonideal=bool(nonideal), branch='reference', seed=0)


def project_drive(p, d):
    """Scale every emergent-cycle affinity by d through the reverse constants."""
    p = dict(p); a, b, *_ = MODELS[p['model']]; S = np.array(b, float) - np.array(a, float)
    k = np.array(p['k'], float); Aff = np.log(k[::2] / k[1::2])
    P = np.eye(S.shape[1]) - S.T @ np.linalg.solve(S @ S.T, S)
    k[1::2] *= np.exp((1 - d) * (P @ Aff)); p['k'] = k.tolist(); p['drive'] = float(d)
    return p


def sample(model, index, seed=20260927, nonideal=False):
    """Deterministic input: even index near the reference, odd index wide."""
    mi = list(MODELS).index(model)
    rng = np.random.default_rng(np.random.SeedSequence([seed, mi, int(nonideal), index]))
    p = reference(model); a, b, k, tu, x, g = MODELS[model]; n = len(x)
    near = index % 2 == 0; r = rng.uniform(1, 1.3 if near else 4.)
    p['k'] = (np.array(k) * np.exp(rng.uniform(-np.log(r), np.log(r), len(k)))).tolist()
    p = project_drive(p, rng.uniform(.95, 1.03) if near else rng.uniform(.3, 1.15))
    p.update(seed=seed, index=index, branch='near' if near else 'wide', nonideal=bool(nonideal))
    p['x0'] = (np.array(x) * np.exp(rng.uniform(-2.3, 2.3, n))).tolist()
    d = rng.normal(size=n); p['d0'] = (d / np.linalg.norm(d)).tolist()
    if nonideal:
        Q = np.linalg.qr(rng.normal(size=(n, n)))[0]
        p['Gamma'] = ((Q * rng.uniform(.05, 1, n) * g) @ Q.T).tolist()
    return p


def arrays(p):
    a, b, *_ = MODELS[p['model']]
    A = np.array(a, float); Bm = np.array(b, float)
    k = np.array(p['k'], float) / p['time_unit']; G = np.array(p['Gamma'], float)
    if np.any(k <= 0) or not np.allclose(G, G.T) or np.linalg.eigvalsh(G).min() < -1e-12:
        raise ValueError('rates must be positive and Gamma symmetric positive semidefinite')
    return A, Bm, np.log(k[::2]), np.log(k[1::2]), G


# ------------------------------------------------------------ kernels -------
@njit(cache=True, error_model='numpy')
def kinetics(z, A, Bm, lkp, lkm, G, x, jp, jm, j, t, aff, la):
    n, m = A.shape
    for i in range(n):
        if not (abs(z[i]) < 650.):
            return False
        x[i] = np.exp(z[i])
    for i in range(n):
        s = z[i]
        for l in range(n):
            s += G[i, l] * x[l]
        la[i] = s
    for r in range(m):
        lp = lkp[r]; lm = lkm[r]
        for i in range(n):
            lp += A[i, r] * la[i]; lm += Bm[i, r] * la[i]
        if not (abs(lp) < 650. and abs(lm) < 650.):
            return False
        jp[r] = np.exp(lp); jm[r] = np.exp(lm); d = lp - lm; aff[r] = d
        j[r] = jp[r] * (-np.expm1(-d)) if d >= 0 else jm[r] * np.expm1(d)
        t[r] = jp[r] + jm[r]
    return True


@njit(cache=True, error_model='numpy')
def jacobian(x, jp, jm, A, Bm, G, J, H):
    """J = S (diag(j+) nu'^T - diag(j-) nu''^T) H,  H = diag(1/x) + Gamma."""
    n, m = A.shape
    for i in range(n):
        for l in range(n):
            H[i, l] = G[i, l]
            J[i, l] = 0.
        H[i, i] += 1. / x[i]
    for r in range(m):
        for l in range(n):
            g = 0.
            for i in range(n):
                g += (jp[r] * A[i, r] - jm[r] * Bm[i, r]) * H[i, l]
            if g != 0.:
                for i in range(n):
                    J[i, l] += (Bm[i, r] - A[i, r]) * g


@njit(cache=True, error_model='numpy')
def observe(x, j, t, aff, A, Bm, H, J, f, v, out, Hv):
    """Bound terms for direction v; returns the stretching-identity residual."""
    n, m = A.shape
    for i in range(n):
        Hv[i] = 0.
        for l in range(n):
            Hv[i] += H[i, l] * v[l]
    n2 = 0.
    for i in range(n):
        n2 += v[i] * Hv[i]
    s = 1. / np.sqrt(n2)
    sig = 0.; sps = 0.; D = 0.; B = 0.; sjb = 0.
    for r in range(m):
        a = 0.; c = 0.; drift = 0.
        for i in range(n):
            S_ir = Bm[i, r] - A[i, r]
            a += S_ir * Hv[i] * s                                   # S_r^T H u
            c += 0.5 * (A[i, r] + Bm[i, r]) * Hv[i] * s             # u . grad ln omega_r
            drift += -0.5 * S_ir * (v[i] * s) ** 2 / (x[i] * x[i])  # metric drift
        b = a * c + drift
        D += 0.5 * t[r] * a * a; B += 0.5 * t[r] * b * b; sjb += j[r] * b
        sig += j[r] * aff[r]; sps += 2. * j[r] * j[r] / t[r]
    direct = 0.; tr = 0.
    for i in range(n):
        Ju = 0.
        for l in range(n):
            Ju += J[i, l] * v[l] * s
        direct += Hv[i] * s * Ju - 0.5 * f[i] * (v[i] * s) ** 2 / (x[i] * x[i])
        tr += J[i, i]
    out[0] = sig; out[1] = sps; out[2] = D; out[3] = B; out[4] = direct; out[5] = tr
    return abs(direct - (sjb - D)) / (1. + abs(direct) + D + abs(sjb))


@njit(cache=True, error_model='numpy')
def work(n, m):
    """Scratch arrays for deriv (allocated once per window)."""
    return (np.empty(n), np.empty(m), np.empty(m), np.empty(m), np.empty(m), np.empty(m), np.empty(n),
            np.empty((n, n)), np.empty((n, n)), np.empty(n), np.empty(n), np.empty(n))


@njit(cache=True, error_model='numpy')
def deriv(z, Y, A, Bm, lkp, lkm, G, dz, dY, dq, wk):
    n, m = A.shape
    x, jp, jm, j, t, aff, la, J, H, f, Hv, v = wk
    if not kinetics(z, A, Bm, lkp, lkm, G, x, jp, jm, j, t, aff, la):
        return False, 0.
    jacobian(x, jp, jm, A, Bm, G, J, H)
    for i in range(n):
        f[i] = 0.
        v[i] = Y[i, 0]
    for r in range(m):
        for i in range(n):
            f[i] += (Bm[i, r] - A[i, r]) * j[r]
    for i in range(n):
        dz[i] = f[i] / x[i]
    k = Y.shape[1]
    for i in range(n):
        for c in range(k):
            s = 0.
            for l in range(n):
                s += J[i, l] * Y[l, c]
            dY[i, c] = s
    res = observe(x, j, t, aff, A, Bm, H, J, f, v, dq, Hv)
    return True, res


# Dormand-Prince 5(4)
DPA = np.array([[0., 0, 0, 0, 0, 0], [1 / 5, 0, 0, 0, 0, 0], [3 / 40, 9 / 40, 0, 0, 0, 0],
                [44 / 45, -56 / 15, 32 / 9, 0, 0, 0], [19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729, 0, 0],
                [9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656, 0],
                [35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84]])
DPB = np.array([35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0.])
DPE = np.array([71 / 57600, 0, -71 / 16695, 71 / 1920, -17253 / 339200, 22 / 525, -1 / 40])


@njit(cache=True, error_model='numpy')
def mgs(Y, logs):
    """Modified Gram-Schmidt in place; adds log|R_ii| to logs."""
    n, k = Y.shape
    for c in range(k):
        for p in range(c):
            d = 0.
            for i in range(n):
                d += Y[i, p] * Y[i, c]
            for i in range(n):
                Y[i, c] -= d * Y[i, p]
        nr = 0.
        for i in range(n):
            nr += Y[i, c] * Y[i, c]
        nr = np.sqrt(nr)
        logs[c] += np.log(nr)
        for i in range(n):
            Y[i, c] /= nr


@njit(cache=True, error_model='numpy')
def hnorm(z, G, v):
    n = z.size; s = 0.
    for i in range(n):
        s += v[i] * v[i] * np.exp(-z[i])
        for l in range(n):
            s += v[i] * G[i, l] * v[l]
    return np.sqrt(s)


@njit(cache=True, error_model='numpy')
def window(z0, Y0, T, nblk, orth, A, Bm, lkp, lkm, G, rtol, atol, max_steps, nrec):
    """Integrate state + tangent columns over [0, T] in nblk blocks.
    orth: re-orthonormalise the columns after every step (Lyapunov mode); otherwise
    propagate the raw tangent matrix (monodromy mode).
    Returns status, z, Y, spectrum blocks, lambda_H blocks, observable blocks,
    recorded x samples, x_min, x_max, max identity residual, accepted steps, all steps.
    Status: 0 ok, 1 overflow / step-size underflow, 3 more than max_steps steps."""
    n = z0.size; k = Y0.shape[1]
    z = z0.copy(); Y = Y0.copy()
    spec = np.zeros((nblk, k)); lamH = np.zeros(nblk); obs = np.zeros((nblk, NOB)); btime = np.zeros(nblk)
    rec = np.empty((max(nrec, 1), n)); irec = 0; rdt = T / max(nrec, 1); trec = 0.
    xmin = np.full(n, 1e300); xmax = np.zeros(n)
    dummy = np.zeros(k)
    if orth:
        mgs(Y, dummy)
    wk = work(n, A.shape[1]); logs = np.zeros(k); rr6 = 0.
    Kz = np.empty((7, n)); KY = np.empty((7, n, k)); Kq = np.empty((7, NOB))
    zs = np.empty(n); Ys = np.empty((n, k)); z5 = np.empty(n); Y5 = np.empty((n, k))
    t = 0.; h = min(1e-3, T); blen = T / nblk; resid = 0.; nacc = 0; nstep = 0
    ok, r0 = deriv(z, Y, A, Bm, lkp, lkm, G, Kz[0], KY[0], Kq[0], wk)
    if not ok:
        return 1, z, Y, spec, lamH, obs, rec[:0], xmin, xmax, resid, nacc, nstep
    while t < T * (1. - 1e-14):
        nstep += 1
        if nstep >= max_steps:
            return 3, z, Y, spec, lamH, obs, rec[:irec], xmin, xmax, resid, nacc, nstep
        b = min(int(t / blen + 1e-12), nblk - 1)
        tb = (b + 1) * blen if b < nblk - 1 else T
        if t + h > tb:
            h = tb - t
        good = True
        for s in range(1, 7):
            for i in range(n):
                acc = 0.
                for q in range(s):
                    acc += DPA[s, q] * Kz[q, i]
                zs[i] = z[i] + h * acc
                for c in range(k):
                    acc = 0.
                    for q in range(s):
                        acc += DPA[s, q] * KY[q, i, c]
                    Ys[i, c] = Y[i, c] + h * acc
            ok, rr = deriv(zs, Ys, A, Bm, lkp, lkm, G, Kz[s], KY[s], Kq[s], wk)
            if not ok:
                good = False
                break
            if s == 6:
                rr6 = rr
                for i in range(n):
                    z5[i] = zs[i]
                    for c in range(k):
                        Y5[i, c] = Ys[i, c]
        if not good:
            h *= 0.25
            if h < 1e-14:
                return 1, z, Y, spec, lamH, obs, rec[:irec], xmin, xmax, resid, nacc, nstep
            continue
        err = 0.
        for i in range(n):
            e = 0.
            for q in range(7):
                e += DPE[q] * Kz[q, i]
            err += (h * e / (atol + rtol * max(abs(z[i]), abs(z5[i])))) ** 2
            for c in range(k):
                e = 0.
                for q in range(7):
                    e += DPE[q] * KY[q, i, c]
                err += (h * e / (atol + rtol * max(abs(Y[i, c]), abs(Y5[i, c])))) ** 2
        err = np.sqrt(err / (n * (1 + k)))
        if err > 1.:
            h *= max(0.2, 0.9 * err ** -0.2)
            continue
        # accepted: 5th-order quadrature of the bound terms (not error-controlled)
        for q in range(NOB):
            acc = 0.
            for st in range(7):
                acc += DPB[st] * Kq[st, q]
            obs[b, q] += h * acc
        btime[b] += h
        lamH[b] += np.log(hnorm(z5, G, Y5[:, 0]) / hnorm(z, G, Y[:, 0]))
        # samples of x at a fixed spacing (linear interpolation in log concentration)
        while irec < nrec and trec <= t + h:
            w = (trec - t) / h
            for i in range(n):
                rec[irec, i] = np.exp(z[i] + w * (z5[i] - z[i]))
            irec += 1; trec += rdt
        for i in range(n):
            z[i] = z5[i]
            for c in range(k):
                Y[i, c] = Y5[i, c]
            xi = np.exp(z[i])
            if xi < xmin[i]: xmin[i] = xi
            if xi > xmax[i]: xmax[i] = xi
        if orth:
            for c in range(k):
                logs[c] = 0.
            mgs(Y, logs)
            for c in range(k):
                spec[b, c] += logs[c]
        # FSAL: the last stage was evaluated at the new point. Orthonormalisation only
        # rescales the leading column, so the bound terms are unchanged; J Y is redone.
        Jn = wk[7]
        for i in range(n):
            Kz[0, i] = Kz[6, i]
            for c in range(k):
                acc = 0.
                for l in range(n):
                    acc += Jn[i, l] * Y[l, c]
                KY[0, i, c] = acc
        for q in range(NOB):
            Kq[0, q] = Kq[6, q]
        if rr6 > resid:
            resid = rr6
        t += h; nacc += 1
        h *= min(5., max(0.2, 0.9 * err ** -0.2)) if err > 1e-14 else 5.
    for bb in range(nblk):
        if btime[bb] > 0:
            for c in range(k):
                spec[bb, c] /= btime[bb]
            lamH[bb] /= btime[bb]
            for q in range(NOB):
                obs[bb, q] /= btime[bb]
    return 0, z, Y, spec, lamH, obs, rec[:irec], xmin, xmax, resid, nacc, nstep


@njit(cache=True, error_model='numpy')
def state(z, A, Bm, lkp, lkm, G):
    """x, f, J, residual scale at log state z (for Newton and fixed-point modes)."""
    n, m = A.shape
    x = np.empty(n); jp = np.empty(m); jm = np.empty(m); j = np.empty(m); t = np.empty(m); aff = np.empty(m)
    ok = kinetics(z, A, Bm, lkp, lkm, G, x, jp, jm, j, t, aff, np.empty(n))
    J = np.empty((n, n)); H = np.empty((n, n)); f = np.zeros(n); sc = np.ones(n)
    if ok:
        jacobian(x, jp, jm, A, Bm, G, J, H)
        for r in range(m):
            for i in range(n):
                f[i] += (Bm[i, r] - A[i, r]) * j[r]
                sc[i] += abs(Bm[i, r] - A[i, r]) * t[r]
    return ok, x, f, J, sc


@njit(cache=True, error_model='numpy')
def newton_fixed(z0, A, Bm, lkp, lkm, G, tol):
    n = z0.size; z = z0.copy()
    for it in range(200):
        ok, x, f, J, sc = state(z, A, Bm, lkp, lkm, G)
        if not ok:
            return z, 1e300
        res = np.max(np.abs(f / sc))
        if res < tol:
            return z, res
        Jz = J * x.reshape(1, n)                      # d f / d z
        dz = np.linalg.solve(Jz, -f)
        mx = np.max(np.abs(dz))
        if mx > 2.:
            dz *= 2. / mx
        lam = 1.
        for _ in range(30):
            ok2, x2, f2, J2, sc2 = state(z + lam * dz, A, Bm, lkp, lkm, G)
            if ok2 and np.max(np.abs(f2 / sc2)) < res:
                break
            lam *= 0.5
        z = z + lam * dz
    ok, x, f, J, sc = state(z, A, Bm, lkp, lkm, G)
    return z, np.max(np.abs(f / sc)) if ok else 1e300


@njit(cache=True, error_model='numpy')
def observe_at(z, v, A, Bm, lkp, lkm, G, out):
    n, m = A.shape
    x = np.empty(n); jp = np.empty(m); jm = np.empty(m); j = np.empty(m); t = np.empty(m); aff = np.empty(m)
    if not kinetics(z, A, Bm, lkp, lkm, G, x, jp, jm, j, t, aff, np.empty(n)):
        out[:] = np.nan
        return np.inf
    J = np.empty((n, n)); H = np.empty((n, n)); f = np.zeros(n)
    jacobian(x, jp, jm, A, Bm, G, J, H)
    for r in range(m):
        for i in range(n):
            f[i] += (Bm[i, r] - A[i, r]) * j[r]
    return observe(x, j, t, aff, A, Bm, H, J, f, v, out, np.empty(n))


# ------------------------------------------------------ classification -------
def _fixed(z, K, cfg):
    zs, res = newton_fixed(z, *K, cfg.fp_residual)
    if res > cfg.fp_residual:
        return None
    ok, x, f, J, sc = state(zs, *K)
    vals, vecs = np.linalg.eig(J)
    i = int(np.argmax(vals.real)); v = vecs[:, i]; out = np.empty(NOB)
    if abs(vals[i].imag) < 1e-10:
        observe_at(zs, np.ascontiguousarray(v.real), *K, out); ob = out.copy()
    else:                                              # rotation average of a complex pair
        old = None
        for cnt in (64, 128, 256, 512, 1024, 2048):
            acc = np.zeros(NOB)
            for ph in np.linspace(0, 2 * np.pi, cnt, endpoint=False):
                observe_at(zs, np.ascontiguousarray(np.real(v * np.exp(1j * ph))), *K, out); acc += out
            ob = acc / cnt
            if old is not None and np.max(np.abs(ob - old) / (1 + np.abs(ob))) < 2e-7:
                break
            old = ob
    return zs, vals, ob, res


def _cycle(z_end, rec, K, cfg, win):
    """Recurrence seed -> Newton shooting on (z0, T) -> Floquet test -> flow-tangent averages."""
    n = rec.shape[1]
    amp = np.ptp(rec, axis=0) / (1 + rec.mean(axis=0))
    if amp.max() < 1e-4:
        return None
    d = int(np.argmax(amp)); X = rec[:, d]; level = 0.5 * (X.max() + X.min())
    cr = np.where((X[:-1] < level) & (X[1:] >= level))[0]
    if len(cr) < 7:
        return None
    dt = cfg.duration / cfg.n_record
    fr = (level - X[cr]) / (X[cr + 1] - X[cr]); ct = (cr + fr) * dt
    cx = rec[cr] + fr[:, None] * (rec[cr + 1] - rec[cr]); T = None
    for q in range(1, min(16, (len(cr) - 1) // 5) + 1):
        per = (ct[q:] - ct[:-q])[-5 * q:]
        dist = np.max(np.abs(cx[q:] - cx[:-q]) / (1 + np.abs(cx[q:])), axis=1)[-5 * q:]
        if np.std(per) / np.mean(per) < cfg.recurrence_tol and dist.max() < cfg.recurrence_tol:
            T = float(np.median(per)); break
    if T is None:
        return None
    z = np.log(cx[-1]); T0 = T; rt, at = 1e-11, 1e-13
    for it in range(40):
        st, zT, M = win(z, np.eye(n), T, 1, False, rt, at, 0)[:3]
        okT, xT, fT, JT, scT = state(zT, *K)
        R = np.r_[zT - z, z[d] - np.log(level)]
        if np.max(np.abs(R)) < cfg.shooting_tol:
            break
        x0 = np.exp(z); Mz = (M / xT[:, None]) * x0[None, :]
        Jac = np.zeros((n + 1, n + 1)); Jac[:n, :n] = Mz - np.eye(n); Jac[:n, n] = fT / xT; Jac[n, d] = 1.
        try:
            step = np.linalg.solve(Jac, -R)
        except np.linalg.LinAlgError:
            return None
        z = z + step[:n]; T = T + step[n]
        if not (T0 / 2 < T < 2 * T0) or np.max(np.abs(np.exp(z) - cx[-1]) / (1 + cx[-1])) > .05:
            return None
    else:
        return None
    defect = float(np.max(np.abs(R)))
    st, zT, M = win(z, np.eye(n), T, 1, False, rt, at, 0)[:3]
    mu = np.linalg.eigvals(M); i0 = int(np.argmin(np.abs(mu - 1)))
    if abs(mu[i0] - 1) > cfg.floquet_tol or np.any(np.abs(np.delete(mu, i0)) >= 1 - cfg.floquet_tol):
        return None
    ok, x0, f0, J0, sc0 = state(z, *K)
    res = win(z, f0.reshape(n, 1).copy(), T, 1, True, cfg.rtol, cfg.atol, 0)
    return dict(obs=res[5][0], lamH_measured=float(res[4][0]), period=T, cycle_closure=defect,
                floquet_moduli=np.sort(np.abs(mu))[::-1], xcycle=np.exp(z), identity_residual=res[9])


def _chaotic(spec, cfg):
    mean = spec.mean(axis=0); se = spec.std(axis=0, ddof=1) / np.sqrt(len(spec))
    return bool(len(mean) >= 3 and mean[0] > max(cfg.chaos_floor, 4 * se[0])
                and np.ptp(spec[:, 0]) < max(8 * cfg.chaos_relative * mean[0], 4 * cfg.chaos_floor)
                and abs(mean[1]) < max(cfg.chaos_floor, 4 * se[1], .15 * mean[0])
                and mean[-1] < 0 and mean.sum() < 0)


class StepBudget(RuntimeError):
    pass


def clean(a):
    if isinstance(a, dict): return {k: clean(v) for k, v in a.items()}
    if isinstance(a, (list, tuple, np.ndarray)): return [clean(v) for v in a]
    if isinstance(a, np.bool_): return bool(a)
    if isinstance(a, np.integer): return int(a)
    if isinstance(a, (float, np.floating)): return float(a) if np.isfinite(a) else None
    return a


def warmup():
    """Compile (or load from the numba cache) every kernel once."""
    p = reference('wr'); p['k'][8] = 15.5
    run_point(p, Config(transient=1., duration=2., n_record=16, confirmation=False))


def basis(d):
    """Orthonormal basis whose first column is the unit vector along d."""
    d = np.asarray(d, float); d = d / np.linalg.norm(d)
    Q = np.linalg.qr(np.column_stack((d, np.eye(d.size))))[0][:, :d.size]
    return np.ascontiguousarray(Q * np.sign(Q[:, 0] @ d))


def run_point(p, cfg=None):
    """Integrate one input, classify it, and return the thermodynamic-bound terms."""
    cfg = cfg or Config(); t0 = time.monotonic()
    out = dict(parameters=p, config=asdict(cfg), version=VERSION, status='ok', cls='unresolved', selected=False)
    try:
        K = arrays(p); n = len(p['x0']); used = [0]

        def win(z, Y, T, nblk, orth, rtol, atol, nrec):
            """window() under the per-input step budget; raises on failure."""
            r = window(z, Y, T, nblk, orth, *K, rtol, atol, cfg.step_budget - used[0], nrec)
            used[0] += r[11]; out['steps'] = used[0]
            if r[0] == 3:
                raise StepBudget('more than %d integration steps (stiff input)' % cfg.step_budget)
            if r[0]:
                raise RuntimeError('integration failed (overflow or step size underflow)')
            return r

        z = np.log(np.array(p['x0'], float)); Y = basis(p['d0'])
        _, z, Y, *_ = win(z, Y, cfg.transient, 1, True, cfg.rtol, cfg.atol, 0)

        def measure(z, Y, T):
            return win(z, Y, T, cfg.blocks, True, cfg.rtol, cfg.atol, cfg.n_record)

        def classify(r):
            _, zE, YE, spec, lamH, obs, rec, xmin, xmax, resid, nacc, nst = r
            tail = rec[len(rec) // 2:]
            amplitude = np.max(np.ptp(tail, axis=0) / (1 + tail.mean(axis=0)))
            fp = _fixed(zE, K, cfg)
            if fp is not None and fp[1].real.max() < -1e-8 and amplitude < cfg.fp_distance \
                    and np.max(np.abs(np.exp(fp[0]) - np.exp(zE)) / (1 + np.exp(zE))) < cfg.fp_distance:
                return 'fixed', fp
            try:
                cy = _cycle(zE, rec, K, cfg, win)
            except StepBudget:
                raise
            except (RuntimeError, np.linalg.LinAlgError):      # shooting left the admissible region
                cy = None
            if cy is not None:
                return 'cycle', cy
            return 'unresolved', None

        first = run = measure(z, Y, cfg.duration)
        cls, info = classify(run); confirmed = False
        if cls == 'unresolved' and cfg.confirmation:
            run = measure(run[1], run[2], 2 * cfg.duration)
            cls, info = classify(run)
            a = first[3][:, 0]; b = run[3][:, 0]
            agree = abs(a.mean() - b.mean()) < max(cfg.chaos_relative * abs(b.mean()),
                                                   4 * np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b)))
            confirmed = _chaotic(first[3], cfg) and _chaotic(run[3], cfg) and agree
            if cls == 'unresolved' and confirmed:
                cls = 'chaos_candidate'
        _, zE, YE, spec, lamH, obs, rec, xmin, xmax, resid, nacc, nst = run
        ob = obs.mean(axis=0)
        out.update(spectrum=spec.mean(axis=0), spectrum_blocks=spec, lambda_H_blocks=lamH, lam1=spec[:, 0].mean(),
                   lambda_H=lamH.mean(), endpoint=np.exp(zE), xmin=xmin, xmax=xmax, identity_residual=resid,
                   spectrum_trace_error=abs(spec.mean(axis=0).sum() - ob[5]),
                   stretch_integral_error=abs(lamH.mean() - ob[4]), confirmation_passed=confirmed,
                   measured_duration=2 * cfg.duration if run is not first else cfg.duration)
        out.update(dict(zip(OBS, ob)))
        if cls == 'fixed':
            zs, vals, obf, res = info
            out.update(dict(zip(OBS, obf)))
            out.update(xstar=np.exp(zs), eigen_real=np.sort(vals.real)[::-1], lam1=vals.real.max(),
                       lambda_H=vals.real.max(), beta_max=np.abs(vals.imag).max(), root_residual=res,
                       coefficient_source='fixed_eigenmode')
        elif cls == 'cycle':
            out.update(dict(zip(OBS, info['obs'])))
            out.update(lam1=0., lambda_H=0., period=info['period'], cycle_closure=info['cycle_closure'],
                       floquet_moduli=info['floquet_moduli'], xcycle=info['xcycle'],
                       measured_cycle_lambda_H=info['lamH_measured'], coefficient_source='verified_cycle')
        else:
            out['coefficient_source'] = 'trajectory'
        out.update(cls=cls, selected=cls != 'unresolved')
        if max(out['spectrum_trace_error'], out['stretch_integral_error']) > 1e-5 * (1 + abs(out['lam1'])) \
                or resid > 1e-10:
            out.update(cls='unresolved', selected=False, status='audit_failed')
        B = out['B1']; out['motion_cost'] = out['Delta1'] ** 2 / B if B > 0 else None
        out['bound_gap'] = out['sigma_ps'] * B - (out['lambda_H'] + out['Delta1']) ** 2
    except StepBudget as err:
        out.update(status='step_budget', selected=False, error=str(err))
    except Exception as err:
        out.update(status='failed', selected=False, error=type(err).__name__ + ': ' + str(err))
    out['seconds'] = time.monotonic() - t0
    return clean(out)
