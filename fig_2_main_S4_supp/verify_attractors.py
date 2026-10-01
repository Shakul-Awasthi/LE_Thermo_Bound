"""Independent re-check of the limit cycles and chaos in the Figure 2 CSV.

Uses only numpy/scipy (stiff solvers, no code from the numba engine except the model
tables), on all or a random subsample of the moving attractors:

limit cycle   start on the stored orbit point x*, integrate one stored period T with Radau
              (rtol 1e-12): closure |x(T) - x*|; monodromy matrix from the variational
              equations: one Floquet multiplier = 1, all others inside the unit circle;
              then 20 further periods: the orbit must stay closed (stable, not transient).
chaos         start at the stored end point, integrate state + tangent with LSODA
              (rtol 1e-10) for 4 consecutive windows of `--chaos-window` time units:
              lambda_1 > 0 in every window (no late collapse to a cycle or fixed point),
              the oscillation persists, and the long-time lambda_1 agrees with the CSV.

    python3 verify_attractors.py --input data_fig2/fig_2_inputs.csv --n-cycle 200 --n-chaos 200 --workers 16

Writes <output>/verify_attractors.csv (one row per checked input) and verify_summary.json.
"""
import os
for _v in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
import argparse, csv, json, math, time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from scipy.integrate import solve_ivp

MODEL_TABLES = None


def model(p):
    """Stoichiometry from crn_cost.MODELS; rates and Gamma from the stored input."""
    global MODEL_TABLES
    if MODEL_TABLES is None:
        from crn_cost import MODELS
        MODEL_TABLES = MODELS
    a, b, *_ = MODEL_TABLES[p['model']]
    A = np.array(a, float); Bm = np.array(b, float)
    k = np.array(p['k'], float) / p['time_unit']
    return A, Bm, np.log(k[::2]), np.log(k[1::2]), np.array(p['Gamma'], float)


def field(z, A, Bm, lkp, lkm, G):
    x = np.exp(z); la = z + G @ x
    jp = np.exp(lkp + A.T @ la); jm = np.exp(lkm + Bm.T @ la)
    S = Bm - A; f = S @ (jp - jm); H = np.diag(1 / x) + G
    J = S @ (np.diag(jp) @ A.T - np.diag(jm) @ Bm.T) @ H          # d f / d x
    return x, f, J


def check_cycle(p, x0, T):
    K = model(p); n = len(x0); z0 = np.log(x0)

    def var(t, y):
        z = y[:n]; M = y[n:].reshape(n, n); x, f, J = field(z, *K)
        Jz = (J * x[None, :] - np.diag(f)) / x[:, None]         # d(f/x)/dz
        return np.r_[f / x, (Jz @ M).ravel()]
    def jz(z):
        x, f, J = field(z, *K)
        return (J * x[None, :] - np.diag(f)) / x[:, None]

    def var_jac(t, y):                          # Newton matrix for Radau; second derivatives omitted
        Jz = jz(y[:n]); out = np.zeros((n + n * n, n + n * n)); out[:n, :n] = Jz
        out[n:, n:] = np.kron(Jz, np.eye(n))
        return out
    sol = solve_ivp(var, (0, T), np.r_[z0, np.eye(n).ravel()], method='Radau', rtol=1e-12, atol=1e-14, jac=var_jac)
    zT = sol.y[:n, -1]; M = sol.y[n:, -1].reshape(n, n)
    closure = float(np.max(np.abs(np.exp(zT) - x0) / x0))
    mu = np.linalg.eigvals(M); i0 = int(np.argmin(np.abs(mu - 1)))
    trivial = float(abs(mu[i0] - 1)); others = float(np.max(np.abs(np.delete(mu, i0)))) if n > 1 else 0.
    # 20 more periods: sample the orbit at multiples of T
    long = solve_ivp(lambda t, z: field(z, *K)[1] / np.exp(z), (0, 20 * T), z0, method='Radau', rtol=1e-11, atol=1e-13,
                     t_eval=T * np.arange(1, 21), jac=lambda t, z: jz(z))
    drift = float(np.max(np.abs(np.exp(long.y) - x0[:, None]) / x0[:, None])) if long.success else np.inf
    ok = closure < 1e-5 and trivial < 1e-3 and others < 1 - 1e-4 and drift < 1e-4
    return dict(closure=closure, floquet_trivial_error=trivial, floquet_max_nontrivial=others,
                drift_after_20_periods=drift, verified=bool(ok))


def check_chaos(p, x0, lam_csv, window, windows):
    K = model(p); n = len(x0); rng = np.random.default_rng(0)
    v = rng.normal(size=n); v /= np.linalg.norm(v); z = np.log(x0); tau = 10.0; lams = []; amps = []

    def rhs(t, y):
        x, f, J = field(y[:n], *K)
        return np.r_[f / x, J @ y[n:]]
    for w in range(windows):
        acc = 0.; xs = []
        for s in range(int(round(window / tau))):
            sol = solve_ivp(rhs, (0, tau), np.r_[z, v], method='LSODA', rtol=1e-10, atol=1e-12)
            if not sol.success:
                return dict(verified=False, error='integration failed')
            z = sol.y[:n, -1]; v = sol.y[n:, -1]; g = np.linalg.norm(v); acc += math.log(g); v = v / g
            xs.append(np.exp(z))
        lams.append(acc / window); xs = np.array(xs)
        amps.append(float(np.max(np.ptp(xs, axis=0) / (1 + xs.mean(axis=0)))))
    lams = np.array(lams); lam = float(lams.mean())
    ok = bool(np.all(lams > 0) and min(amps) > 1e-3 and abs(lam - lam_csv) < max(0.25 * lam_csv, 4 * lams.std(ddof=1) / math.sqrt(windows)))
    return dict(lambda1_windows=lams.tolist(), lambda1_long=lam, lambda1_csv=lam_csv,
                relative_difference=float((lam - lam_csv) / lam_csv), min_amplitude=min(amps), verified=ok)


def work(task):
    kind, row = task; t0 = time.monotonic(); p = json.loads(row['parameters_json'])
    try:
        if kind == 'cycle':
            x0 = np.array([float(row['xcycle_%d' % i]) for i in range(1, len(p['x0']) + 1)])
            res = check_cycle(p, x0, float(row['period']))
        else:
            x0 = np.array([float(row['endpoint_%d' % i]) for i in range(1, len(p['x0']) + 1)])
            res = check_chaos(p, x0, float(row['lambda1']), task_window, 4)
    except Exception as err:
        res = dict(verified=False, error='%s: %s' % (type(err).__name__, err))
    res.update(kind=kind, point_id=row['point_id'], model=row['model'], nonideal=row['nonideal'],
               seconds=time.monotonic() - t0)
    return res


task_window = 2500.


def init(window):
    global task_window
    task_window = window


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--input', default='data_fig2/fig_2_inputs.csv')
    ap.add_argument('--output', default='data_fig2/verify')
    ap.add_argument('--n-cycle', type=int, default=200, help='random cycles to check (0 = none, -1 = all)')
    ap.add_argument('--n-chaos', type=int, default=200, help='random chaos candidates to check (-1 = all)')
    ap.add_argument('--chaos-window', type=float, default=2500., help='length of each of the 4 chaos windows')
    ap.add_argument('--workers', type=int, default=1); ap.add_argument('--seed', type=int, default=7)
    a = ap.parse_args()
    # pass 1: classes only (cheap); pass 2: full rows of the chosen inputs
    with open(a.input, newline='', encoding='utf-8') as f:
        rd = csv.reader(f); head = next(rd); ic = head.index('cls')
        cls_of = [line[ic] for line in rd]
    rng = np.random.default_rng(a.seed); pick = {}
    for kind, cls, n in (('cycle', 'cycle', a.n_cycle), ('chaos', 'chaos_candidate', a.n_chaos)):
        g = [i for i, c in enumerate(cls_of) if c == cls]
        for j in rng.permutation(len(g))[: (len(g) if n < 0 else n)]:
            pick[g[j]] = kind
    tasks = []
    with open(a.input, newline='', encoding='utf-8') as f:
        for i, r in enumerate(csv.DictReader(f)):
            if i in pick:
                tasks.append((pick[i], r))
    init(a.chaos_window)
    if a.workers > 1:
        with ProcessPoolExecutor(a.workers, initializer=init, initargs=(a.chaos_window,)) as pool:
            results = list(pool.map(work, tasks, chunksize=1))
    else:
        results = [work(t) for t in tasks]
    out = Path(a.output); out.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for r in results for k in r))
    with open(out / 'verify_attractors.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, keys); w.writeheader(); w.writerows(results)
    summ = {}
    for kind in ('cycle', 'chaos'):
        g = [r for r in results if r['kind'] == kind]
        if not g:
            continue
        s = dict(checked=len(g), verified=sum(r['verified'] for r in g),
                 failed=[dict(point_id=r['point_id'], model=r['model'], **{k: r.get(k) for k in r if k not in ('point_id', 'model', 'kind')})
                         for r in g if not r['verified']])
        if kind == 'cycle':
            s.update(max_closure=max(r.get('closure', np.inf) for r in g),
                     max_nontrivial_floquet=max(r.get('floquet_max_nontrivial', np.inf) for r in g),
                     max_drift_20_periods=max(r.get('drift_after_20_periods', np.inf) for r in g))
        else:
            d = [abs(r['relative_difference']) for r in g if 'relative_difference' in r]
            s.update(median_abs_rel_diff_lambda1=float(np.median(d)) if d else None, max_abs_rel_diff_lambda1=max(d) if d else None,
                     min_window_lambda1=min(min(r['lambda1_windows']) for r in g if 'lambda1_windows' in r))
        summ[kind] = s
    (out / 'verify_summary.json').write_text(json.dumps(summ, indent=2, default=float))
    short = {k: dict({kk: vv for kk, vv in v.items() if kk != 'failed'}, n_failed=len(v['failed'])) for k, v in summ.items()}
    print(json.dumps(short, indent=2, default=float))


if __name__ == '__main__':
    main()
