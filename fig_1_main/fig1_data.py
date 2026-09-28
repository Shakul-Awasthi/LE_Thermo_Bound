#!/usr/bin/env python3
"""
fig1_data.py -- data for Fig. 1: ideal reversible Willamowski-Rossler network.

  1. scan of k+5 at the reference drive (panels b, c):
     leading exponent lambda_1 and, along the leading tangent, Delta_1, B_1,
     sigma_ps, with the oscillation amplitude and the block spread of lambda_1;
  2. trace of the chaotic attractor at k+5 = 16.5 (panel a), with the
     instantaneous Hessian stretching rate lambda_H.

Output: fig1_data.npz.   Usage: python3 fig1_data.py [n_workers]
Rates are reported in units of k+1 (raw values divided by TU = 30).
"""
import sys, time
import numpy as np
from numba import njit

# ---- network: A1+X<->2X, X+Y<->2Y, A5+Y<->A2, X+Z<->A3, A4+Z<->2Z (chemostats absorbed)
NUP = np.array([[1., 1., 0., 1., 0.], [0., 1., 1., 0., 0.], [0., 0., 0., 1., 1.]])
NUM = np.array([[2., 0., 0., 0., 0.], [0., 2., 0., 0., 0.], [0., 0., 0., 0., 2.]])
S = NUM - NUP
NBAR = 0.5 * (NUP + NUM)
K0 = np.array([30., .25, 1., 1e-4, 10., 1e-3, 1., 1e-4, 16.5, .5])   # (k+1, k-1, ..., k+5, k-5)
TU = 30.
X0 = np.array([10., 10., 10.])
D0 = np.array([1., .3, -.2])


@njit(cache=True)
def fluxes(x, k, jp, jm):
    lna = np.log(x)
    for r in range(5):
        ep = 0.; em = 0.
        for i in range(3):
            ep += NUP[i, r] * lna[i]; em += NUM[i, r] * lna[i]
        jp[r] = k[2 * r] * np.exp(ep); jm[r] = k[2 * r + 1] * np.exp(em)


@njit(cache=True)
def rhs(x, dl, k, fx, fd):
    """f = S j and the tangent J dl, J = S (diag(j+) NUP^T - diag(j-) NUM^T) H, H = diag(1/x)."""
    jp = np.empty(5); jm = np.empty(5)
    fluxes(x, k, jp, jm)
    for i in range(3):
        fx[i] = 0.; fd[i] = 0.
    for r in range(5):
        dp = 0.; dm = 0.
        for i in range(3):
            dp += NUP[i, r] * dl[i] / x[i]; dm += NUM[i, r] * dl[i] / x[i]
        dj = jp[r] * dp - jm[r] * dm; j = jp[r] - jm[r]
        for i in range(3):
            fx[i] += S[i, r] * j; fd[i] += S[i, r] * dj


@njit(cache=True)
def observables(x, dl, k, out):
    """out = [sigma_ps, Delta, B, lambda_H (direct), ||dl||_H^2] for the tangent dl."""
    jp = np.empty(5); jm = np.empty(5)
    fluxes(x, k, jp, jm)
    n2 = 0.
    for i in range(3):
        n2 += dl[i] * dl[i] / x[i]
    nrm = np.sqrt(n2)
    sps = 0.; Delta = 0.; B = 0.
    for r in range(5):
        j = jp[r] - jm[r]; t = jp[r] + jm[r]
        sps += 2. * j * j / t
        a = 0.; c = 0.; drift = 0.
        for i in range(3):
            hu = dl[i] / (x[i] * nrm); u = dl[i] / nrm
            a += S[i, r] * hu                              # S_r^T H u
            c += NBAR[i, r] * hu                           # u . grad ln omega_r
            drift += -0.5 * S[i, r] * u * u / (x[i] * x[i])
        b = a * c + drift                                  # reaction gain, Eq. (10)
        Delta += 0.5 * t * a * a
        B += 0.5 * t * b * b
    fx = np.empty(3); fd = np.empty(3)
    rhs(x, dl, k, fx, fd)
    num = 0.
    for i in range(3):                                     # dl^T H J dl + 1/2 dl^T Hdot dl
        num += dl[i] / x[i] * fd[i] - 0.5 * fx[i] * dl[i] * dl[i] / (x[i] * x[i])
    out[0] = sps; out[1] = Delta; out[2] = B; out[3] = num / n2; out[4] = n2


@njit(cache=True)
def integrate(x0, d0, k, t_trans, t_avg, n_blocks, rtol=1e-8, atol=1e-11, h=4e-4):
    """Adaptive Dormand-Prince 5(4) for state + tangent; tangent renormalised in the
    H norm after every accepted step; trapezoidal block averages after the transient.
    Returns status, lam_blocks, avg (n_blocks x 3: sigma_ps, Delta, B), xmin, xmax."""
    A = np.array([[0., 0., 0., 0., 0., 0.],
                  [1/5, 0., 0., 0., 0., 0.],
                  [3/40, 9/40, 0., 0., 0., 0.],
                  [44/45, -56/15, 32/9, 0., 0., 0.],
                  [19372/6561, -25360/2187, 64448/6561, -212/729, 0., 0.],
                  [9017/3168, -355/33, 46732/5247, 49/176, -5103/18656, 0.],
                  [35/384, 0., 500/1113, 125/192, -2187/6784, 11/84]])
    E = np.array([71/57600, 0., -71/16695, 71/1920, -17253/339200, 22/525, -1/40])
    x = x0.copy(); dl = d0.copy()
    out = np.empty(5); out2 = np.empty(5)
    observables(x, dl, k, out); dl = dl / np.sqrt(out[4])
    lam = np.zeros(n_blocks); acc = np.zeros((n_blocks, 4))
    xmin = np.full(3, 1e300); xmax = np.full(3, -1e300)
    Kx = np.empty((7, 3)); Kd = np.empty((7, 3)); xt = np.empty(3); dt_ = np.empty(3)
    t = 0.; T = t_trans + t_avg; blk = t_avg / n_blocks
    while t < T:
        if t + h > T:
            h = T - t
        rhs(x, dl, k, Kx[0], Kd[0])
        ok = True
        for s in range(1, 7):
            for i in range(3):
                sx = 0.; sd = 0.
                for q in range(s):
                    sx += A[s, q] * Kx[q, i]; sd += A[s, q] * Kd[q, i]
                xt[i] = x[i] + h * sx; dt_[i] = dl[i] + h * sd
                if not (xt[i] > 0.) or xt[i] > 1e7:
                    ok = False
            if not ok:
                break
            rhs(xt, dt_, k, Kx[s], Kd[s])
        if not ok:
            h *= 0.25
            if h < 1e-14:
                return 1, lam, acc[:, :3], xmin, xmax
            continue
        err = 0.
        for i in range(3):
            ex = 0.; ed = 0.
            for q in range(7):
                ex += h * E[q] * Kx[q, i]; ed += h * E[q] * Kd[q, i]
            err += (ex / (atol + rtol * max(abs(x[i]), abs(xt[i])))) ** 2
            err += (ed / (atol + rtol * max(abs(dl[i]), abs(dt_[i])))) ** 2
        err = np.sqrt(err / 6)
        if err > 1.:
            h *= max(0.2, 0.9 * err ** -0.2)
            continue
        observables(xt, dt_, k, out2)
        tn = t + h
        if tn > t_trans:                                   # part of the step inside the window
            ta = max(t, t_trans); w = tn - ta
            b = min(int((ta - t_trans) / blk), n_blocks - 1)
            for q in range(3):
                acc[b, q] += 0.5 * w * (out[q] + out2[q])
            acc[b, 3] += w
            lam[b] += (w / h) * 0.5 * np.log(out2[4])      # H-norm log growth over the step
            for i in range(3):
                xmin[i] = min(xmin[i], xt[i]); xmax[i] = max(xmax[i], xt[i])
        for i in range(3):
            x[i] = xt[i]; dl[i] = dt_[i] / np.sqrt(out2[4])
        for q in range(4):
            out[q] = out2[q]
        t = tn
        h *= min(5., max(0.2, 0.9 * err ** -0.2)) if err > 1e-12 else 5.
    avg = np.empty((n_blocks, 3))
    for b in range(n_blocks):
        lam[b] /= acc[b, 3]
        for q in range(3):
            avg[b, q] = acc[b, q] / acc[b, 3]
    return 0, lam, avg, xmin, xmax


@njit(cache=True)
def trace(x0, d0, k, dt, t_trans, t_rec, every):
    """Fixed-step RK4 trace of state and aligned tangent; returns x and lambda_H samples."""
    x = x0.copy(); dl = d0.copy(); out = np.empty(5)
    fx = np.empty((4, 3)); fd = np.empty((4, 3)); xt = np.empty(3); dt_ = np.empty(3)
    n_tr = int(t_trans / dt); n_rec = int(t_rec / dt)
    X = np.empty((n_rec // every + 1, 3)); L = np.empty(n_rec // every + 1); m = 0
    for step in range(n_tr + n_rec):
        rhs(x, dl, k, fx[0], fd[0])
        for s, c in ((1, 0.5), (2, 0.5), (3, 1.0)):
            for i in range(3):
                xt[i] = x[i] + c * dt * fx[s - 1, i]; dt_[i] = dl[i] + c * dt * fd[s - 1, i]
            rhs(xt, dt_, k, fx[s], fd[s])
        for i in range(3):
            x[i] += dt * (fx[0, i] + 2 * fx[1, i] + 2 * fx[2, i] + fx[3, i]) / 6.
            dl[i] += dt * (fd[0, i] + 2 * fd[1, i] + 2 * fd[2, i] + fd[3, i]) / 6.
        observables(x, dl, k, out)
        dl = dl / np.sqrt(out[4])
        if step >= n_tr and (step - n_tr) % every == 0:
            X[m] = x; L[m] = out[3]; m += 1
    return X[:m], L[:m]


def scan_point(k5):
    k = K0.copy(); k[8] = k5
    st, lam, avg, xmin, xmax = integrate(X0, D0, k, 900., 3000., 6)
    if st:
        return None
    lam = lam / TU; m = avg.mean(axis=0) / TU
    amp = float(np.max((xmax - xmin) / (xmax + xmin)))
    return (k5, lam.mean(), m[1], m[2], m[0], amp, float(np.ptp(lam)))


if __name__ == "__main__":
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    ks = np.concatenate([np.arange(13.0, 16.3, 0.05), np.arange(16.3, 16.601, 0.01)])
    t0 = time.time()
    scan_point(16.5)                                        # compile once before forking
    if workers > 1:
        from multiprocessing import Pool
        with Pool(workers) as p:
            rows = p.map(scan_point, ks)
    else:
        rows = []
        for k5 in ks:
            rows.append(scan_point(k5)); print("k+5 = %.3f  %s" % (k5, rows[-1]), flush=True)
    rows = np.array([r for r in rows if r is not None])
    X, L = trace(X0, D0, K0, 4e-4, 400., 400., 10)
    np.savez("fig1_data.npz", k5=rows[:, 0], lam=rows[:, 1], Delta=rows[:, 2], B=rows[:, 3],
             sps=rows[:, 4], amp=rows[:, 5], spread=rows[:, 6], trace_x=X, trace_lamH=L / TU)
    print("saved fig1_data.npz: %d scan points, %d trace samples, %.0f s" % (len(rows), len(L), time.time() - t0))
