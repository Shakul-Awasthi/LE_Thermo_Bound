"""Checks of the numba engine against independent numpy/scipy code. Must print "status": "PASS".

    python3 validate_numerics.py                 # algebra + integrator (about a minute incl. compilation)
    python3 validate_numerics.py --dynamics      # adds WR k5 = 13 (fixed), 15.5 (cycle), 16.5 (chaos)

algebra     Jacobian vs finite differences of f(x); split J = C - D H; every bound term
            (sigma, sigma_ps, Delta, B, Hessian stretching rate) vs a plain numpy formula;
            stretching identity; reservoir completion; lambda = -Delta at detailed balance.
integrator  Dormand-Prince state vs scipy solve_ivp (Radau, tight tolerances); Liouville
            identity sum(spectrum) = <tr J>; integrated stretching rate = lambda_H.
dynamics    classification and audits of three reference WR inputs; motion-cost bound.
"""
import argparse, json
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import null_space
from scipy.optimize import root
import crn_cost as cc


def plain(p, x, v):
    """Independent numpy implementation of the model and of the bound terms."""
    A, Bm, lkp, lkm, G = cc.arrays(p); S = Bm - A; n = x.size
    la = np.log(x) + G @ x
    jp = np.exp(lkp + A.T @ la); jm = np.exp(lkm + Bm.T @ la); j = jp - jm; t = jp + jm
    f = S @ j; H = np.diag(1 / x) + G
    J = S @ (np.diag(jp) @ A.T - np.diag(jm) @ Bm.T) @ H
    u = v / np.sqrt(v @ H @ v); a = S.T @ H @ u; c = .5 * (A + Bm).T @ H @ u
    b = a * c - .5 * S.T @ (u ** 2 / x ** 2)
    terms = dict(sigma=j @ np.log(jp / jm),
                 sigma_ps=2 * np.sum(j ** 2 / t), Delta1=.5 * np.sum(t * a ** 2), B1=.5 * np.sum(t * b ** 2),
                 stretch_H=u @ H @ J @ u - .5 * np.sum(f * u ** 2 / x ** 2))
    return f, J, H, S, j, t, terms


def algebra():
    rng = np.random.default_rng(1409); worst = dict(jacobian=0., split=0., terms=0., identity=0., detailed_balance=0.)
    for model in cc.MODELS:
        for non in (False, True):
            p = cc.sample(model, 1, nonideal=non) if non else cc.reference(model)
            K = cc.arrays(p); A, Bm, *_ = K; n = A.shape[0]
            for _ in range(5):
                x = np.exp(rng.uniform(-1, 1, n)); v = rng.normal(size=n)
                ok, xk, fk, Jk, sc = cc.state(np.log(x), *K); assert ok
                f, J, H, S, j, t, terms = plain(p, x, v)
                h = 1e-6 * x
                fd = np.column_stack([(plain(p, x + h[i] * np.eye(n)[i], v)[0] - plain(p, x - h[i] * np.eye(n)[i], v)[0]) / (2 * h[i]) for i in range(n)])
                e = np.max(abs(fd - Jk)) / (1 + np.max(abs(Jk))); worst['jacobian'] = max(worst['jacobian'], e); assert e < 1e-6, e
                assert np.allclose(fk, f, rtol=1e-12, atol=1e-14) and np.allclose(Jk, J, rtol=1e-11, atol=1e-14)
                nu = .5 * (A + Bm); D = .5 * (S * t) @ S.T; C = (S * j) @ (H @ nu).T
                e = np.max(abs(J - (C - D @ H))) / (1 + np.max(abs(J))); worst['split'] = max(worst['split'], e); assert e < 1e-12, e
                out = np.empty(cc.NOB); res = cc.observe_at(np.log(x), v, *K, out)
                for q, name in enumerate(cc.OBS[:5]):
                    e = abs(out[q] - terms[name]) / (1 + abs(terms[name])); worst['terms'] = max(worst['terms'], e); assert e < 1e-11, (name, e)
                assert abs(out[5] - np.trace(J)) < 1e-10 * (1 + abs(np.trace(J)))
                worst['identity'] = max(worst['identity'], res); assert res < 1e-12
                assert out[0] + 1e-12 >= out[1]                                   # sigma >= sigma_ps
                assert (out[4] + out[2]) ** 2 <= out[1] * out[3] * (1 + 1e-10) + 1e-14   # pointwise bound
            m = A.shape[1]
            assert null_space(np.vstack((S, -np.eye(m), np.eye(m)))).shape[1] == 0     # reservoir completion
            q = cc.project_drive(p, 0.); K = cc.arrays(q)
            target = np.linalg.solve(S @ S.T, S @ np.log(np.array(q['k'][::2]) / np.array(q['k'][1::2])))
            G = np.array(q['Gamma']); sol = root(lambda z: z + G @ np.exp(z) - target, target, jac=lambda z: np.eye(n) + G * np.exp(z), tol=1e-13)
            x = np.exp(sol.x); f, J, H, S, j, t, _ = plain(q, x, np.ones(n)); assert np.max(abs(j)) < 1e-9
            vals, vecs = np.linalg.eig(J); assert np.max(abs(vals.imag)) < 1e-9
            for lam, v in zip(vals.real, vecs.T.real):
                out = np.empty(cc.NOB); cc.observe_at(sol.x, np.ascontiguousarray(v), *K, out)
                e = abs(lam + out[2]); worst['detailed_balance'] = max(worst['detailed_balance'], e); assert e < 1e-7, e
    return worst


def integrator():
    rep = {}
    for model in cc.MODELS:
        p = cc.sample(model, 0); K = cc.arrays(p); n = len(p['x0']); z0 = np.log(p['x0']); T = 20.
        r = cc.window(z0, cc.basis(p['d0']), T, 2, True, *K, 1e-10, 1e-13, 10 ** 7, 0)
        assert r[0] == 0
        ref = solve_ivp(lambda s, z: plain(p, np.exp(z), np.ones(n))[0] / np.exp(z), (0, T), z0, method='Radau', rtol=1e-12, atol=1e-14)
        e_state = np.max(abs(np.exp(r[1]) - np.exp(ref.y[:, -1])) / (1e-8 + np.exp(ref.y[:, -1])))
        spec, lamH, obs = r[3].mean(0), r[4].mean(0), r[5].mean(0)
        e_liouville = abs(spec.sum() - obs[5]); e_stretch = abs(lamH - obs[4])
        rep[model] = dict(state_rel_error=e_state, liouville_error=e_liouville, stretch_error=e_stretch)
        assert e_state < 1e-6 and e_liouville < 1e-7 and e_stretch < 1e-7, rep[model]
    return rep


def dynamics():
    rep = []
    for k5, want in ((13, 'fixed'), (15.5, 'cycle'), (16.5, 'chaos_candidate')):
        p = cc.reference('wr'); p['k'][8] = k5; r = cc.run_point(p)
        rep.append({k: r.get(k) for k in ('cls', 'status', 'lambda_H', 'Delta1', 'B1', 'sigma_ps', 'motion_cost', 'bound_gap',
                                          'period', 'identity_residual', 'spectrum_trace_error', 'stretch_integral_error', 'seconds')})
        rep[-1]['k5'] = k5
        assert r['status'] == 'ok' and r['cls'] == want, (k5, r['cls'], r['status'])
        assert r['bound_gap'] >= 0 and (want == 'fixed' or r['sigma_ps'] >= r['motion_cost'])
    return rep


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--output', default='data_validation'); ap.add_argument('--dynamics', action='store_true')
    a = ap.parse_args()
    cc.warmup()
    report = dict(engine=cc.VERSION, algebra=algebra(), integrator=integrator())
    if a.dynamics:
        report['dynamics'] = dynamics()
    report['status'] = 'PASS'
    Path(a.output).mkdir(parents=True, exist_ok=True)
    (Path(a.output) / 'validation_report.json').write_text(json.dumps(cc.clean(report), indent=2))
    print(json.dumps(cc.clean(report), indent=2))


if __name__ == '__main__':
    main()
