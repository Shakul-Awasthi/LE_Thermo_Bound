"""Numbers for the manuscript from the Figure 2 run: counts, tightness, selection, numerical quality.

Reads the CSV of fig_2_input_selected.py, the sampling database(s), the audit of
fig_2_main_plot.py and the summary of fig_2_supp_plt.py, and writes

    <output>/fig_2_stats.json     every number, machine readable
    <output>/fig_2_stats.md       the same as readable tables
    <output>/fig_2_numbers.tex    LaTeX macros (\\newcommand) for quoting in the text / SM

    python3 fig_2_stats.py --input data_fig2/fig_2_inputs.csv --sampling data_fig2 \\
        --main-audit data_fig2/main_selection --supp-summary Figures/fig2_supp_summary.json \\
        --output data_fig2/stats

Definitions (all time averages along the leading tangent direction, Hessian metric):
  saturation s = Delta1^2 / (B1 sigma_ps)                 motion cost, s <= 1 for cycles and chaos
  bound ratio r = (lambda1 + Delta1)^2 / (sigma_ps B1)    full bound, r <= 1 for every input
  budget use  f = lambda1 / (sqrt(sigma_ps B1) - Delta1)  chaos only: share of the allowed growth rate
  sigma_ps / sigma                                        pseudo-entropy production vs entropy production
"""
import argparse, csv, json, math, re, sqlite3
from pathlib import Path
import numpy as np
from scipy import stats

MODELS = ('wr', 'brusselator', 'oregonator', 'cs14')
LABELS = {'wr': 'WR', 'brusselator': 'Brusselator', 'oregonator': 'Oregonator', 'cs14': 'CS14'}
CLASSES = ('fixed', 'cycle', 'chaos_candidate')
MOVING = ('cycle', 'chaos_candidate')
MIX = {False: 'ideal', True: 'nonideal'}


def num(r, k):
    try:
        v = float(r.get(k, ''))
        return v if math.isfinite(v) else np.nan
    except (TypeError, ValueError):
        return np.nan


def flag(r, k):
    return str(r.get(k, '')).lower() in ('true', '1')


def describe(v):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    if not len(v):
        return dict(n=0)
    return dict(n=int(len(v)), min=float(v.min()), p10=float(np.percentile(v, 10)), median=float(np.median(v)),
                mean=float(v.mean()), p90=float(np.percentile(v, 90)), max=float(v.max()))


KEEP = re.compile(r'^(point_id|model|nonideal|cls|lambda1|Delta1|B1|sigma_ps|sigma|test_dataset|index|drive|'
                  r'identity_residual|spectrum_trace_error|stretch_integral_error|cycle_closure|root_residual|'
                  r'confirmation_passed|steps|seconds|spectrum_blocks_json|xmin_\d+|xmax_\d+|floquet_moduli_\d+)$')


def load(path):
    """Only the columns the statistics need (parameters, rates etc. are skipped to save memory)."""
    rows = []
    with open(path, newline='', encoding='utf-8') as f:
        rd = csv.reader(f); head = next(rd); ix = [(k, i) for i, k in enumerate(head) if KEEP.match(k)]
        for line in rd:
            rows.append({k: line[i] for k, i in ix})
    for r in rows:
        r['_non'] = flag(r, 'nonideal')
        lam, D, B, sp, sg = (num(r, k) for k in ('lambda1', 'Delta1', 'B1', 'sigma_ps', 'sigma'))
        r['_s'] = D * D / (B * sp)
        r['_r'] = (lam + D) ** 2 / (sp * B)
        r['_ratio_ps'] = sp / sg if sg > 0 else np.nan
        r['_f'] = lam / (math.sqrt(sp * B) - D) if r['cls'] == 'chaos_candidate' and math.sqrt(sp * B) > D else np.nan
        xmin = [num(r, 'xmin_%d' % i) for i in range(1, 4)]; xmax = [num(r, 'xmax_%d' % i) for i in range(1, 4)]
        amp = [math.log10(b / a) for a, b in zip(xmin, xmax) if a > 0 and b > 0]
        r['_amp'] = max(amp) if amp else np.nan                # decades of the widest species oscillation
        mods = sorted([num(r, 'floquet_moduli_%d' % i) for i in range(1, 4) if math.isfinite(num(r, 'floquet_moduli_%d' % i))], reverse=True)
        r['_floq2'] = mods[1] if len(mods) > 1 else np.nan      # largest nontrivial Floquet modulus
        try:
            blocks = np.array(json.loads(r.get('spectrum_blocks_json') or 'null'))[:, 0]
            r['_lam_se'] = float(blocks.std(ddof=1) / math.sqrt(len(blocks)))
        except (TypeError, ValueError, IndexError):
            r['_lam_se'] = np.nan
    return rows


# ----------------------------------------------------------------- sections --
def dataset(rows):
    out = dict(total=len(rows), cases={})
    for m in MODELS:
        for non in (False, True):
            g = [r for r in rows if r['model'] == m and r['_non'] == non]
            out['cases']['%s/%s' % (m, MIX[non])] = dict(n=len(g), **{c: sum(r['cls'] == c for r in g) for c in CLASSES})
    out['classes'] = {c: sum(r['cls'] == c for r in rows) for c in CLASSES}
    return out


def sampling(rows, directory):
    """Attempts needed to fill each case (up to the last selected proposal index) and why the others were rejected."""
    d = Path(directory)
    files = sorted((d / 'workers').glob('*.sqlite')) or sorted((d / 'shards').glob('shard_*.sqlite')) or \
        ([d / 'fig_2_sampling.sqlite'] if (d / 'fig_2_sampling.sqlite').exists() else [])
    if not files:
        return None
    last = {}
    for r in rows:
        key = (r['model'], r['_non']); last[key] = max(last.get(key, -1), int(float(r['index'])))
    out = {}
    for m in MODELS:
        for non in (False, True):
            reasons = {}; n = 0; acc = 0; seen = set()
            for fpath in files:
                db = sqlite3.connect('file:%s?mode=ro' % fpath, uri=True)
                q = 'SELECT sample_index,accepted,result FROM attempts WHERE model=? AND nonideal=? AND sample_index<=?'
                for i, a, raw in db.execute(q, (m, int(non), last.get((m, non), -1))):
                    if i in seen:
                        continue
                    seen.add(i); n += 1
                    if a:
                        acc += 1; continue
                    res = json.loads(raw); st = res.get('status'); cl = res.get('cls')
                    why = 'stiff (step budget)' if st == 'step_budget' else 'unresolved' if st == 'ok' and cl == 'unresolved' \
                        else 'audit failed' if st == 'audit_failed' else 'integration failed' if st == 'failed' else 'other'
                    reasons[why] = reasons.get(why, 0) + 1
                db.close()
            out['%s/%s' % (m, MIX[non])] = dict(attempts=n, accepted=acc, acceptance=acc / n if n else None, rejected=reasons)
    return out


def main_figure(audit_dir, rows):
    d = Path(audit_dir); js = d / 'fig_2_main_selection.json'
    if not js.exists():
        return None
    rep = json.loads(js.read_text())
    with open(d / 'fig_2_main_selected.csv', newline='', encoding='utf-8') as f:
        sel = list(csv.DictReader(f))
    moving = [r for r in rows if r['cls'] in MOVING]
    shown = dict(total=len(sel), cycle=sum(r['cls'] == 'cycle' for r in sel), chaos=sum(r['cls'] == 'chaos_candidate' for r in sel))
    per_net = {m: dict(cycle=sum(r['model'] == m and r['cls'] == 'cycle' for r in sel),
                       chaos=sum(r['model'] == m and r['cls'] == 'chaos_candidate' for r in sel),
                       ideal=sum(r['model'] == m and not flag(r, 'nonideal') for r in sel),
                       nonideal=sum(r['model'] == m and flag(r, 'nonideal') for r in sel)) for m in MODELS}
    fixed_all = rep.get('fixed_points'); fixed_in = rep.get('fixed_drawn', fixed_all)
    return dict(fixed_points_total=fixed_all, fixed_points_plotted=fixed_in,
                fixed_points_outside_axes=(fixed_all - fixed_in) if fixed_all is not None else None,
                fixed_under_inset=rep.get('fixed_under_inset'), moving_available=len(moving),
                moving_outside_axes=rep.get('moving_outside_window'),
                moving_available_cycle=sum(r['cls'] == 'cycle' for r in moving),
                moving_available_chaos=sum(r['cls'] == 'chaos_candidate' for r in moving),
                moving_plotted=shown, moving_not_plotted=len(moving) - len(sel), per_network=per_net,
                cell_decades=rep.get('cell_decades'), cap_per_cell=rep.get('cap'), occupied_cells=rep.get('occupied_cells'),
                per_cell_min=rep.get('per_cell_min'), per_cell_max=rep.get('per_cell_max'), axis_limits=rep.get('limits'),
                points_outside_axes=rep.get('all_outside_window'), points_under_inset=rep.get('points_under_inset'),
                inset_histogram_n=rep.get('moving_total'), seed=rep.get('seed'))


def tightness(rows):
    out = dict(saturation={}, bound_ratio={}, sigma_ps_over_sigma={}, chaos_budget_use={})
    for m in MODELS + ('all',):
        for non in (False, True, None):
            g = [r for r in rows if (m == 'all' or r['model'] == m) and (non is None or r['_non'] == non)]
            key = '%s/%s' % (m, 'both' if non is None else MIX[non])
            mv = [r for r in g if r['cls'] in MOVING]
            out['saturation'][key] = dict(moving=describe([r['_s'] for r in mv]),
                                          cycle=describe([r['_s'] for r in mv if r['cls'] == 'cycle']),
                                          chaos=describe([r['_s'] for r in mv if r['cls'] == 'chaos_candidate']))
            out['bound_ratio'][key] = {c: describe([r['_r'] for r in g if r['cls'] == c]) for c in CLASSES}
            out['sigma_ps_over_sigma'][key] = {c: describe([r['_ratio_ps'] for r in g if r['cls'] == c]) for c in CLASSES}
            out['chaos_budget_use'][key] = describe([r['_f'] for r in g])
    mv = [r for r in rows if r['cls'] in MOVING]
    out['sigma_ps_over_sigma_moving'] = describe([r['_ratio_ps'] for r in mv])
    s = np.array([r['_s'] for r in mv]); rr = np.array([r['_r'] for r in rows])
    out['violations'] = dict(motion_cost=int(np.sum(s > 1)), full_bound=int(np.sum(rr > 1)),
                             largest_saturation=float(np.nanmax(s)) if len(s) else None,
                             largest_bound_ratio=float(np.nanmax(rr)),
                             smallest_relative_margin_motion_cost=float(1 - np.nanmax(s)) if len(s) else None,
                             smallest_relative_margin_full_bound=float(1 - np.nanmax(rr)))
    return out


def drivers(rows):
    """Spearman correlations of the saturation (cycles and chaos) with candidate controls."""
    out = {}
    for m in MODELS + ('all',):
        g = [r for r in rows if r['cls'] in MOVING and (m == 'all' or r['model'] == m)]
        res = {}
        for name, key, subset in (('drive', 'drive', MOVING), ('oscillation_decades', '_amp', MOVING),
                                  ('nontrivial_floquet_modulus', '_floq2', ('cycle',)), ('Delta1', 'Delta1', MOVING),
                                  ('sigma_ps_over_sigma', '_ratio_ps', MOVING)):
            h = [r for r in g if r['cls'] in subset]
            x = np.array([r[key] if key.startswith('_') else num(r, key) for r in h], float)
            y = np.array([r['_s'] for r in h]); ok = np.isfinite(x) & np.isfinite(y)
            if ok.sum() >= 10 and np.ptp(x[ok]) > 0:
                rho, p = stats.spearmanr(x[ok], y[ok]); res[name] = dict(n=int(ok.sum()), rho=float(rho), p=float(p))
        out[m] = res
    return out


def mixing(rows):
    """Does nonideal mixing change the saturation? Two-sample Kolmogorov-Smirnov test per network."""
    out = {}
    for m in MODELS:
        a = [r['_s'] for r in rows if r['model'] == m and r['cls'] in MOVING and not r['_non']]
        b = [r['_s'] for r in rows if r['model'] == m and r['cls'] in MOVING and r['_non']]
        if len(a) >= 5 and len(b) >= 5:
            t = stats.ks_2samp(a, b)
            out[m] = dict(n_ideal=len(a), n_nonideal=len(b), median_ideal=float(np.median(a)),
                          median_nonideal=float(np.median(b)), ks=float(t.statistic), p=float(t.pvalue))
    return out


def numerics(rows):
    cyc = [r for r in rows if r['cls'] == 'cycle']; ch = [r for r in rows if r['cls'] == 'chaos_candidate']
    mx = lambda g, k: float(np.nanmax([num(r, k) for r in g])) if g else None
    f2 = [r['_floq2'] for r in cyc if math.isfinite(r['_floq2'])]
    rel = [r['_lam_se'] / num(r, 'lambda1') for r in ch if math.isfinite(r['_lam_se']) and num(r, 'lambda1') > 0]
    return dict(max_identity_residual=mx(rows, 'identity_residual'), max_liouville_error=mx(rows, 'spectrum_trace_error'),
                max_stretch_integral_error=mx(rows, 'stretch_integral_error'), max_cycle_closure=mx(cyc, 'cycle_closure'),
                max_nontrivial_floquet_modulus=float(max(f2)) if f2 else None,
                max_fixed_point_root_residual=mx([r for r in rows if r['cls'] == 'fixed'], 'root_residual'),
                chaos_confirmed_in_two_windows=sum(flag(r, 'confirmation_passed') for r in ch), chaos_total=len(ch),
                chaos_lambda1_relative_standard_error=describe(rel),
                max_integration_steps=mx(rows, 'steps'), mean_seconds_per_accepted_input=float(np.nanmean([num(r, 'seconds') for r in rows])))


# ------------------------------------------------------------------ output --
def fmt(v, digits=2):
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return '--'
    if isinstance(v, (int, np.integer)) or (isinstance(v, float) and v.is_integer() and abs(v) >= 1000):
        return '{:,}'.format(int(v))
    if v != 0 and (abs(v) < 1e-3 or abs(v) >= 1e5):
        return '%.1e' % v
    return ('%%.%dg' % (digits + 1)) % v


def tex_num(v, digits=2):
    s = fmt(v, digits)
    if 'e' in s:
        a, e = s.split('e'); return r'\ensuremath{%s\times10^{%d}}' % (a, int(e))
    return s.replace(',', '{,}')


def macros(S):
    """LaTeX macro names use letters only. \\FigTwo... prefix."""
    M = {}
    ds, mf, tt, nu = S['dataset'], S.get('main_figure') or {}, S['tightness'], S['numerics']
    M['FigTwoInputs'] = ds['total']
    M['FigTwoPerCase'] = next(iter(ds['cases'].values()))['n']
    M['FigTwoFixed'] = ds['classes']['fixed']; M['FigTwoCycles'] = ds['classes']['cycle']
    M['FigTwoChaos'] = ds['classes']['chaos_candidate']
    if mf:
        M['FigTwoMainFixedPlotted'] = mf['fixed_points_plotted']
        M['FigTwoMainFixedOutside'] = mf['fixed_points_outside_axes']
        M['FigTwoMainMovingOutside'] = mf['moving_outside_axes']
        M['FigTwoMainMovingPlotted'] = mf['moving_plotted']['total']
        M['FigTwoMainCyclesPlotted'] = mf['moving_plotted']['cycle']
        M['FigTwoMainChaosPlotted'] = mf['moving_plotted']['chaos']
        M['FigTwoMainMovingAvailable'] = mf['moving_available']
        M['FigTwoMainCellDecades'] = mf['cell_decades']; M['FigTwoMainCap'] = mf['cap_per_cell']
        M['FigTwoMainCells'] = mf['occupied_cells']
    sat = tt['saturation']['all/both']['moving']
    M['FigTwoSatMedian'] = sat.get('median'); M['FigTwoSatNinety'] = sat.get('p90'); M['FigTwoSatMax'] = sat.get('max')
    M['FigTwoSatChaosMax'] = tt['saturation']['all/both']['chaos'].get('max')
    M['FigTwoSatCycleMax'] = tt['saturation']['all/both']['cycle'].get('max')
    M['FigTwoViolations'] = tt['violations']['motion_cost'] + tt['violations']['full_bound']
    M['FigTwoBoundRatioMax'] = tt['violations']['largest_bound_ratio']
    bu = tt['chaos_budget_use']['all/both']
    M['FigTwoBudgetMedian'] = bu.get('median'); M['FigTwoBudgetMax'] = bu.get('max')
    M['FigTwoPsRatioMovingMedian'] = tt['sigma_ps_over_sigma_moving'].get('median')
    M['FigTwoIdentityResidual'] = nu['max_identity_residual']; M['FigTwoLiouvilleError'] = nu['max_liouville_error']
    M['FigTwoCycleClosure'] = nu['max_cycle_closure']; M['FigTwoFloquetMax'] = nu['max_nontrivial_floquet_modulus']
    M['FigTwoChaosLambdaRelErr'] = nu['chaos_lambda1_relative_standard_error'].get('median')
    if S.get('sampling'):
        att = sum(v['attempts'] for v in S['sampling'].values()); acc = sum(v['accepted'] for v in S['sampling'].values())
        M['FigTwoAttempts'] = att; M['FigTwoAcceptance'] = 100 * acc / att if att else None
    V = S.get('verification') or {}
    if 'cycle' in V:
        M['FigTwoVerifyCyclesChecked'] = V['cycle']['checked']; M['FigTwoVerifyCyclesPassed'] = V['cycle']['verified']
        M['FigTwoVerifyCycleClosure'] = V['cycle'].get('max_closure')
    if 'chaos' in V:
        M['FigTwoVerifyChaosChecked'] = V['chaos']['checked']; M['FigTwoVerifyChaosPassed'] = V['chaos']['verified']
        M['FigTwoVerifyChaosLambdaDiff'] = V['chaos'].get('median_abs_rel_diff_lambda1')
    lines = ['% Generated by fig_2_stats.py from ' + S['input'] + ('  [TEST DATA]' if S['test_data'] else ''), '']
    for k, v in M.items():
        lines.append(r'\newcommand{\%s}{%s}' % (k, tex_num(v)))
    return '\n'.join(lines) + '\n', M


def markdown(S, M):
    L = ['# Figure 2: numbers for the manuscript', '', 'Input: `%s`%s' % (S['input'], '  **TEST DATA**' if S['test_data'] else ''), '']
    L += ['## Quotable numbers (LaTeX macros in `fig_2_numbers.tex`)', '', '| macro | value |', '|---|---|']
    L += ['| `\\%s` | %s |' % (k, fmt(v)) for k, v in M.items()]
    L += ['', '## Inputs per case', '', '| case | n | fixed | cycle | chaos |', '|---|---|---|---|---|']
    L += ['| %s | %s | %s | %s | %s |' % (k, v['n'], v['fixed'], v['cycle'], v['chaos_candidate']) for k, v in S['dataset']['cases'].items()]
    if S.get('sampling'):
        L += ['', '## Sampling (attempts up to the last selected proposal index)', '',
              '| case | attempts | accepted | acceptance | rejected |', '|---|---|---|---|---|']
        L += ['| %s | %d | %d | %.1f%% | %s |' % (k, v['attempts'], v['accepted'], 100 * (v['acceptance'] or 0),
                                                 ', '.join('%s %d' % kv for kv in v['rejected'].items()) or '--')
              for k, v in S['sampling'].items()]
    mf = S.get('main_figure')
    if mf:
        L += ['', '## Main figure', '',
              '- fixed points (gray density): %s inside the axes, %s outside (not drawn), %s in total' % (
                  fmt(mf['fixed_points_plotted']), fmt(mf['fixed_points_outside_axes']), fmt(mf['fixed_points_total'])),
              '- cycles and chaos plotted: %s of %s available (%s cycles, %s chaos); %s not shown because of the per-cell cap' % (
                  fmt(mf['moving_plotted']['total']), fmt(mf['moving_available']), fmt(mf['moving_plotted']['cycle']),
                  fmt(mf['moving_plotted']['chaos']), fmt(mf['moving_not_plotted'])),
              '- density cells: %s decade wide, %s occupied, %s to %s points per cell (cap %s)' % (
                  mf['cell_decades'], mf['occupied_cells'], mf['per_cell_min'], mf['per_cell_max'], mf['cap_per_cell']),
              '- axes %s; cycles/chaos outside the axes %s; cycles/chaos under the inset %s (fixed points under it %s); '
              'inset histogram built from %s inputs' % (mf['axis_limits'], fmt(mf['moving_outside_axes']), mf['points_under_inset'],
                                                         mf['fixed_under_inset'], fmt(mf['inset_histogram_n'])),
              '', '| network | cycles | chaos | ideal | nonideal |', '|---|---|---|---|---|']
        L += ['| %s | %d | %d | %d | %d |' % (LABELS[m], v['cycle'], v['chaos'], v['ideal'], v['nonideal']) for m, v in mf['per_network'].items()]
    if S.get('supp_figure'):
        L += ['', '## Supplementary figure (every input plotted)', '', '| panel | network | mixing | plotted | fixed | cycle | chaos |',
              '|---|---|---|---|---|---|---|']
        L += ['| (%s) | %s | %s | %s | %s | %s | %s |' % (p['panel'], LABELS[p['model']], MIX[p['nonideal']], fmt(p['plotted']),
                                                        fmt(p['fixed']), fmt(p['cycle']), fmt(p['chaos'])) for p in S['supp_figure']]
    T = S['tightness']
    L += ['', '## Tightness', '', 'Saturation s = Δ̄₁²/(B̄₁σ̄_ps) of cycles and chaos:', '',
          '| case | n | median | 90% | max |', '|---|---|---|---|---|']
    for k, v in T['saturation'].items():
        d = v['moving']
        if d['n']:
            L.append('| %s | %d | %s | %s | %s |' % (k, d['n'], fmt(d['median']), fmt(d['p90']), fmt(d['max'])))
    L += ['', 'Full bound ratio r = (λ₁+Δ̄₁)²/(σ̄_ps B̄₁), all networks:', '', '| class | n | median | 90% | max |', '|---|---|---|---|---|']
    for c in CLASSES:
        d = T['bound_ratio']['all/both'][c]
        if d['n']:
            L.append('| %s | %d | %s | %s | %s |' % (c, d['n'], fmt(d['median']), fmt(d['p90']), fmt(d['max'])))
    L += ['', 'Violations: motion cost %d, full bound %d; smallest relative margins %s and %s.' % (
        T['violations']['motion_cost'], T['violations']['full_bound'],
        fmt(T['violations']['smallest_relative_margin_motion_cost']), fmt(T['violations']['smallest_relative_margin_full_bound']))]
    L += ['', 'Chaos, share of the allowed growth rate used, f = λ₁/(√(σ̄_ps B̄₁) − Δ̄₁):', '', '| case | n | median | 90% | max |', '|---|---|---|---|---|']
    for k, d in T['chaos_budget_use'].items():
        if d['n']:
            L.append('| %s | %d | %s | %s | %s |' % (k, d['n'], fmt(d['median']), fmt(d['p90']), fmt(d['max'])))
    L += ['', 'σ̄_ps/σ̄ (all networks):', '', '| class | n | median | min | max |', '|---|---|---|---|---|']
    for c in CLASSES:
        d = T['sigma_ps_over_sigma']['all/both'][c]
        if d['n']:
            L.append('| %s | %d | %s | %s | %s |' % (c, d['n'], fmt(d['median']), fmt(d['min']), fmt(d['max'])))
    L += ['', '## What controls the saturation (Spearman ρ, cycles and chaos)', '', '| network | variable | n | ρ | p |', '|---|---|---|---|---|']
    for m, res in S['drivers'].items():
        for name, v in res.items():
            L.append('| %s | %s | %d | %.2f | %.1e |' % (m, name, v['n'], v['rho'], v['p']))
    L += ['', '## Ideal vs nonideal (two-sample KS test on the saturation)', '', '| network | n ideal | n nonideal | median ideal | median nonideal | KS | p |',
          '|---|---|---|---|---|---|---|']
    L += ['| %s | %d | %d | %s | %s | %.3f | %.1e |' % (LABELS[m], v['n_ideal'], v['n_nonideal'], fmt(v['median_ideal']),
                                                        fmt(v['median_nonideal']), v['ks'], v['p']) for m, v in S['mixing'].items()]
    V = S.get('verification')
    if V:
        L += ['', '## Independent re-check of cycles and chaos (verify_attractors.py, scipy stiff solvers)', '']
        for kind, v in V.items():
            L.append('- %s: %d of %d verified; %s' % (kind, v['verified'], v['checked'], ', '.join(
                '%s %s' % (k, fmt(x)) for k, x in v.items() if k not in ('checked', 'verified', 'failed'))))
    L += ['', '## Numerical quality', '']
    L += ['- %s: %s' % (k, fmt(v) if not isinstance(v, dict) else 'median %s, max %s' % (fmt(v.get('median')), fmt(v.get('max'))))
          for k, v in S['numerics'].items()]
    return '\n'.join(L) + '\n'


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--input', default='data_fig2/fig_2_inputs.csv')
    p.add_argument('--sampling', default='data_fig2', help='directory with fig_2_sampling.sqlite or shards/')
    p.add_argument('--main-audit', default='data_fig2/main_selection')
    p.add_argument('--supp-summary', default='Figures/fig2_supp_summary.json')
    p.add_argument('--verify', default='data_fig2/verify/verify_summary.json', help='summary of verify_attractors.py (optional)')
    p.add_argument('--output', default='data_fig2/stats')
    a = p.parse_args()
    rows = load(a.input)
    S = dict(input=a.input, test_data=any(flag(r, 'test_dataset') for r in rows), dataset=dataset(rows),
             sampling=sampling(rows, a.sampling), main_figure=main_figure(a.main_audit, rows),
             supp_figure=json.loads(Path(a.supp_summary).read_text()) if Path(a.supp_summary).exists() else None,
             verification=json.loads(Path(a.verify).read_text()) if Path(a.verify).exists() else None,
             tightness=tightness(rows), drivers=drivers(rows), mixing=mixing(rows), numerics=numerics(rows))
    tex, M = macros(S)
    out = Path(a.output); out.mkdir(parents=True, exist_ok=True)
    (out / 'fig_2_stats.json').write_text(json.dumps(S, indent=2, default=lambda o: None))
    (out / 'fig_2_numbers.tex').write_text(tex)
    (out / 'fig_2_stats.md').write_text(markdown(S, M))
    print(markdown(S, M))


if __name__ == '__main__':
    main()
