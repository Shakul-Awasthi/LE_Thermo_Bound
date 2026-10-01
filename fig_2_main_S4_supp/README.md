# Figure 2: cost of sustained motion

Code for Fig. 2 and its supplementary figure in *Thermodynamic cost of dynamical
instability in deterministic chemical reaction networks*.

- **What it computes.** The code samples the WR, Brusselator, Oregonator and
  CS14 networks, half with ideal and half with nonideal mixing: 100,000
  accepted inputs per network and mixing case, 800,000 in total. For every
  input it computes the terms of the thermodynamic-metric bound,

      |λ₁ + Δ̄₁| ≤ √(σ̄_ps B̄₁),   H = diag(1/x) + Γ,

  with λ₁ measured in the Hessian norm. There are no Euclidean terms.
- **What it plots.** σ̄_ps against Δ̄₁²/B̄₁. Limit cycles and chaos must lie
  above the diagonal.

There are three ways to use it:

- **Procedure 0:** start from the published data. Join and unzip the CSV, then
  make the figures and numbers in minutes, with no sampling.
- **Procedure 1:** compute everything on a desktop computer (one machine,
  days).
- **Procedure 2:** compute everything on the BIRD cluster with PBS: one main
  job, plus optional helper jobs on further nodes. With all nodes it takes about
  2 hours.

All three produce the same files:

| output | content |
|---|---|
| `Figures/fig2_motion.pdf`, `.png` | main-text figure |
| `Figures/fig2_supp.pdf`, `.png` | supplementary figure (PNG at 300 dpi; in the PDF the points of each panel are one embedded image, text and axes vector), 2 × 4 panels (a) WR … (d) CS14, ideal on top and nonideal below |
| `data_fig2/fig_2_inputs.csv` | every plotted input: rates, reservoir potentials, Γ, class, all bound terms, spectrum, … |
| `data_fig2/stats/fig_2_stats.md` | numbers for the manuscript as tables |
| `data_fig2/stats/fig_2_numbers.tex` | the same numbers as LaTeX macros (`\input` it) |
| `data_fig2/verify/` | independent re-check of 200 cycles and 200 chaos candidates |

---

## Procedure 0 — use the published data (no sampling)

The CSV `data_fig2/fig_2_inputs.csv` holds all 800,000 inputs with their
rates, Γ, class and all bound terms. It is about 2 GB, so it is published as a
gzip file cut into pieces of 99 MB (GitHub accepts at most 100 MB per file):

```
data_fig2/fig_2_inputs.csv.gz.part00
data_fig2/fig_2_inputs.csv.gz.part01
…
data_fig2/fig_2_inputs.csv.gz.sha256      checksum of the joined .gz file
```

**1. Join the pieces and unzip** (in the `fig2_code` folder).

Linux or macOS:

```bash
cd data_fig2
cat fig_2_inputs.csv.gz.part* > fig_2_inputs.csv.gz    # join (the pieces sort in the right order)
sha256sum -c fig_2_inputs.csv.gz.sha256                 # must print "OK"  (macOS: shasum -a 256 -c …)
gunzip fig_2_inputs.csv.gz                              # gives fig_2_inputs.csv (about 2 GB)
cd ..
```

Any system, including Windows, with Python only:

```bash
python3 -c "import glob,gzip,shutil; parts=sorted(glob.glob('data_fig2/fig_2_inputs.csv.gz.part*')); open('data_fig2/fig_2_inputs.csv.gz','wb').writelines(open(p,'rb').read() for p in parts); shutil.copyfileobj(gzip.open('data_fig2/fig_2_inputs.csv.gz'), open('data_fig2/fig_2_inputs.csv','wb'))"
```

Both ways join the pieces in order. Joining raw byte pieces is correct here:
the cut was made in the compressed file, and joining restores it byte for
byte. Do not open or edit the pieces individually; on their own they are not
valid files.

**2. Install** as in step 1 of Procedure 1, then **make the figures and numbers**:

```bash
python3 fig_2_main_plot.py --input data_fig2/fig_2_inputs.csv --output Figures
python3 fig_2_supp_plt.py  --input data_fig2/fig_2_inputs.csv --output Figures
python3 verify_attractors.py --input data_fig2/fig_2_inputs.csv --output data_fig2/verify --n-cycle 200 --n-chaos 200 --workers 8
python3 fig_2_stats.py --input data_fig2/fig_2_inputs.csv --sampling data_fig2 --output data_fig2/stats
```

Without LaTeX, add `--no-tex` to the two plot commands. The statistics then
contain everything except the table of sampling attempts and rejections,
because that table needs the raw worker databases, which are not published.

These steps were tested: a CSV cut into pieces, joined, checked and unzipped as
above, and then plotted and analysed.

**For the authors — how the pieces are made.** On BIRD, `fig2_run.pbs` makes
them automatically as its last step:

- it writes `data_fig2/publish/fig_2_inputs.csv.gz.part00, …` and
  `fig_2_inputs.csv.gz.sha256`;
- it checks that the joined pieces reproduce the checksum, and records this in
  `run.log`;
- the CSV itself is kept.

Upload the files in `data_fig2/publish/` into the folder `data_fig2/` of the
repository.

After a desktop run, make the pieces by hand:

```bash
mkdir -p data_fig2/publish && cd data_fig2/publish
gzip -9 -c ../fig_2_inputs.csv > fig_2_inputs.csv.gz
sha256sum fig_2_inputs.csv.gz > fig_2_inputs.csv.gz.sha256
split -b 99000000 -d -a 2 fig_2_inputs.csv.gz fig_2_inputs.csv.gz.part   # .part00, .part01, … (99 MB each)
rm fig_2_inputs.csv.gz
```

gzip reduces the CSV about 3-fold, from about 2 GB to about 650 MB, so about 7
pieces. It takes a few minutes. The same `.csv.gz` can also be deposited in one piece on Zenodo, which
gives it a DOI.

---

## Procedure 1 — desktop computer

**Needs:**

- Python ≥ 3.8 with numpy, scipy ≥ 1.9, numba ≥ 0.57 and matplotlib;
- for the figures, preferably LaTeX with `amsmath`, `bm`, cm-super and
  `dvipng`. Otherwise add `--no-tex` to the two plot commands.

1. **Install** (once), in the unpacked `fig2_code` folder:
   ```bash
   cd fig2_code
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
   Later sessions only need `source .venv/bin/activate`.

2. **Check the engine.** This takes about 1 minute, including the one-time
   numba compilation, and must end with `"status": "PASS"`:
   ```bash
   python3 validate_numerics.py --dynamics
   ```

3. **Optional test run.** This takes a few minutes. It computes at least 32
   inputs per case and makes figures marked TEST DATA:
   ```bash
   python3 fig_2_input_selected.py --test-target 30 --workers 4 --output data_test
   python3 fig_2_main_plot.py --input data_test/fig_2_inputs_test.csv --output Figures_test --audit-dir data_test/main_selection --test-input
   python3 fig_2_supp_plt.py  --input data_test/fig_2_inputs_test.csv --output Figures_test --test-input
   ```

4. **Full sampling.** Set `--workers` to the number of CPU cores (`nproc` on
   Linux, `sysctl -n hw.ncpu` on macOS):
   ```bash
   python3 fig_2_input_selected.py --workers 8 --output data_fig2
   ```
   - It needs about 500 CPU-hours: about 2.5–3 days on 8 cores, about 1.3 days
     on 16.
   - Every result is saved as it is computed. You can stop it at any time with
     Ctrl-C or by switching the computer off; the same command continues where
     it stopped.
   - `python3 fig_2_input_selected.py --status --output data_fig2` shows the
     progress.
   - It ends with `COMPLETE: 800000 rows` and writes `data_fig2/fig_2_inputs.csv`.

5. **Figures:**
   ```bash
   python3 fig_2_main_plot.py --input data_fig2/fig_2_inputs.csv --output Figures
   python3 fig_2_supp_plt.py  --input data_fig2/fig_2_inputs.csv --output Figures
   ```

6. **Re-check and numbers for the manuscript:**
   ```bash
   python3 verify_attractors.py --input data_fig2/fig_2_inputs.csv --output data_fig2/verify --n-cycle 200 --n-chaos 200 --workers 8
   python3 fig_2_stats.py --input data_fig2/fig_2_inputs.csv --sampling data_fig2 --output data_fig2/stats
   ```

Steps 5 and 6 read only the CSV and the audit files. You can also run the
sampling on BIRD (Procedure 2), copy `data_fig2/` to the desktop, and redo the
figures there, for example with other styling.

---

## Procedure 2 — BIRD cluster (PBS)

Everything runs through two PBS files, written like any other BIRD job script
(`module load python/3.8.8`, a virtual environment, `qsub`):

| file | requests | what it does |
|---|---|---|
| `fig2_run.pbs` | 1 node, 80 cores, 32 GB, 8 h | **main job** (always needed): see below |
| `fig2_helper.pbs` | 1 node, 20 cores (change with `-l`), 7 h | **helper** (optional): samples with every core of its node for the main job |

The main job does the following, in order:

1. checks the engine;
2. samples with all its cores;
3. waits until the helpers are done;
4. writes `data_fig2/fig_2_inputs.csv`;
5. makes both figures;
6. runs the independent re-check of cycles and chaos;
7. writes the numbers for the manuscript;
8. cuts the gzipped CSV into 99 MB pieces for GitHub, in `data_fig2/publish/`.

Helpers only sample. They share the work with the main job through the folder
`data_fig2/`:

- Every job takes the next free block of 32 inputs whenever it has idle cores,
  so a large node simply does more than a small one.
- Helpers may start before, together with or after the main job.
- If a helper stops (walltime, node failure), the main job redoes its
  unfinished blocks after 15 minutes.
- The result does not depend on how many helpers ran or where.

### 1. One-time setup (on the login node)

```bash
module load python/3.8.8
python3 -m venv ~/venvs/crn
source ~/venvs/crn/bin/activate
pip install -r requirements.txt
python3 -c "import numpy, scipy, numba, matplotlib; print('ok', numba.__version__)"
```

Under Python 3.8, pip installs numpy 1.24, scipy 1.10, numba 0.58 and
matplotlib 3.7; the code was tested with exactly these versions. Both PBS files
activate `~/venvs/crn` themselves. If you use another environment, change the
`source …/activate` line in both files.

Copy the `fig2_code` folder to your home directory, then `cd ~/fig2_code`. All
`qsub` commands below are run from this folder.

### 2. Optional test run (about 15 min)

```bash
qsub -v TEST=30,OUT=data_test fig2_run.pbs
```

This uses one 80-core node. Results go to `data_test/` and
`Figures_test/`, and the figures are marked TEST DATA. Check
`data_test/logs/run.log`, which should end with `done:`.

### 3. Full run

Submit the main job, and as many helpers as there are free nodes. Give each
helper the size of its node with `-l select=…`:

```bash
qsub fig2_run.pbs                                        # main job, 80 cores (fits bird03)

qsub -l select=1:ncpus=48:mem=32gb fig2_helper.pbs       # a 48-core node (bird01)
qsub -l select=1:ncpus=40:mem=32gb fig2_helper.pbs       # a 40-core node (bird02)
qsub fig2_helper.pbs                                     # a 20-core node (bird04 … bird09):
qsub fig2_helper.pbs                                     #   submit it six times
qsub fig2_helper.pbs
qsub fig2_helper.pbs
qsub fig2_helper.pbs
qsub fig2_helper.pbs
qsub -l select=1:ncpus=12:mem=12gb fig2_helper.pbs       # a 12-core node (bird11 … bird13):
qsub -l select=1:ncpus=12:mem=12gb fig2_helper.pbs       #   submit it three times
qsub -l select=1:ncpus=12:mem=12gb fig2_helper.pbs
```

With all of these, 324 cores work on the run. You can also pin a job to a node
by adding `:host=bird01` to its `select`.

| cores | sampling | whole run |
|---|---|---|
| main job only (80) | about 6.5 h, possibly more | may need a second submission (step 5) |
| main + all helpers (324) | about 1.5 h, at most about 3 h | about 2 h, at most about 4 h |

### 4. Follow the run

```bash
qstat -u $USER
tail data_fig2/logs/run.log                              # main job
ls data_fig2/logs/                                       # one log per helper
module load python/3.8.8; source ~/venvs/crn/bin/activate
python3 fig_2_input_selected.py --status --output data_fig2    # accepted inputs per case so far
```

When the main job has finished, `run.log` ends with `done:`. You will then have:

- `Figures/fig2_motion.pdf`
- `Figures/fig2_supp.pdf` and `.png`
- `data_fig2/fig_2_inputs.csv`
- `data_fig2/stats/`
- `data_fig2/verify/`

### 5. If `run.log` says "sampling not complete"

This happens when the walltime ran out before 100,000 inputs per case were
reached, most likely with few or no helpers. Submit again with the same
commands (`qsub fig2_run.pbs`, plus helpers). Everything computed so far is
kept, and the run continues where it stopped.

### 6. If the cluster crashes

You do not start over. Everything computed so far is kept in `data_fig2/`.

**What is lost.**

- Every job saves its results to disk every few seconds (at most every 10 s).
- A crash therefore loses only the inputs that were being computed at that
  moment, plus the last few seconds of results.
- The result files are crash-safe. If a save was half-written when the cluster
  went down, it is undone automatically the next time the file is opened, so
  the data cannot be corrupted.

**How to restart, once the cluster is back.**

1. Check that no old jobs are left:
   ```bash
   qstat -u $USER
   ```
   If any `crn_fig2` or `crn_fig2_help` jobs are still listed (for example
   stuck), remove them with `qdel <job id>`.
2. Optional: see how far the run got:
   ```bash
   cd ~/fig2_code
   module load python/3.8.8; source ~/venvs/crn/bin/activate
   python3 fig_2_input_selected.py --status --output data_fig2
   ```
   This shows the accepted inputs per case so far.
3. Resubmit exactly the same jobs, from the same folder: `qsub fig2_run.pbs`
   plus the helpers of step 3.

**What happens after the restart.**

- **Nothing is redone.** Inputs already saved are skipped, and the jobs
  continue with the missing ones.
- **Unfinished batches are handed out again.** The batches that the crashed
  jobs were working on are released and taken over by the new jobs once the
  old jobs have been silent for 15 minutes.
  - If you resubmit sooner, this still happens, just a little later.
  - If you wait 15 minutes before resubmitting, it happens right at the start.
- **A crash after sampling costs only minutes.** If the crash happened after
  the sampling was finished (during the figures or the re-check), the main job
  finds the sampling complete, goes straight to the figures and numbers, and
  finishes in about 20 minutes.
- **The result is unchanged.** Every input is fixed in advance, so the final
  CSV is the same as it would have been without the crash.

**What not to do.**

- Do not delete or rename `data_fig2/`: it holds the finished work.
- Do not change settings between submissions (`OUT`, `TEST`, the target). The
  code refuses to mix runs with different settings and stops with an error.
- Do not start a second set of jobs while the first is still running. Nothing
  breaks, but cores are wasted on duplicate work.

---

## Reference

### Files

| file | role |
|---|---|
| `crn_cost.py` | numba engine: models, sampling, integration, classification |
| `validate_numerics.py` | engine checks against independent numpy/scipy code; must say `PASS` |
| `fig_2_input_selected.py` | sampling (one process per machine; several machines cooperate through the output folder), merge to CSV |
| `fig_2_main_plot.py` | main figure, from the CSV only |
| `fig_2_supp_plt.py` | supplementary figure, from the CSV only |
| `verify_attractors.py` | independent re-check of random cycles and chaos candidates (scipy stiff solvers) |
| `fig_2_stats.py` | numbers for the manuscript (`.md`, `.json`, LaTeX macros) |
| `fig2_run.pbs` | BIRD main job: checks, sampling, CSV, figures, re-check, numbers, 99 MB gzip pieces for GitHub |
| `fig2_helper.pbs` | BIRD helper job: extra sampling on another node |
| `requirements.txt` | Python packages |

### Engine (`crn_cost.py`)

- **Networks.** WR, Brusselator, Oregonator and CS14. Every reaction is
  completed with its own fuel/waste reservoir pair, so ln(k⁺/k⁻) = μ_F − μ_W
  holds reaction by reaction. Any positive rate vector is therefore
  thermodynamically consistent: the completed network has no internal cycles
  and hence no Wegscheider conditions.
- **Mixing.** Ideal (Γ = 0) or nonideal with Γ symmetric positive definite, so
  μ = ln x + Γx is the gradient of a convex free energy.
- **Integration.**
  - Log concentrations and the full tangent matrix are integrated with an
    adaptive Dormand–Prince 5(4) scheme, compiled with numba.
  - The tangent is re-orthonormalised after every step.
  - The bound terms are integrated along the leading tangent by 5th-order
    quadrature.
- **Audits on every input.**
  - Stretching identity: pointwise error < 10⁻¹⁰.
  - Liouville: Σλᵢ = ⟨tr J⟩.
  - Integrated stretching rate = λ_H.
- **Classes.**

  | class | test | bound terms |
  |---|---|---|
  | stable fixed point | Newton + spectrum | from the leading eigenmode (phase-averaged for a complex pair) |
  | limit cycle | Poincaré recurrence → Newton shooting on (x₀, T) with the monodromy matrix → Floquet multipliers | along the flow over one exact period, λ₁ = 0 |
  | chaos candidate | two windows (1600 and 3200 time units, after a 600-unit transient) with agreeing λ₁ > 0, λ₂ ≈ 0, λ₃ < 0, Σλ < 0 | trajectory averages |
  | unresolved | none of the above | rejected |

- **Stiff inputs.** An input that would need more than 3·10⁶ integration steps
  is rejected and recorded as `step_budget`.

### Sampling and selection (`fig_2_input_selected.py`)

- **Target.** 100,000 accepted inputs per network and mixing case (`TARGET`).
  The target is recorded in the CSV (`target_per_case`), and the plotters check
  against it.
- **Selection.** The first 100,000 accepted inputs per case in
  proposal-index order. The bound is never used to select.
- **Input generation.** Inputs are generated deterministically from (seed,
  network, mixing, index), so the CSV is the same however many machines
  computed it. A 3-machine test gave exactly the inputs and values of a
  one-process run.
- **Sharing work.** Work is handed out in blocks of 32 indices through
  `data_fig2/claims/`. `mkdir` is atomic, also on NFS, so each block has
  exactly one owner.
  - Workers stop when the accepted inputs plus the expected yield of the
    claimed blocks reach 100,000 + 3√100,000 per case, about 1% extra.
  - Each worker writes `data_fig2/workers/<host>-<job>.sqlite`, and a
    heartbeat in `data_fig2/progress/`.
  - A claimed block records its owner. If the owner has had no heartbeat for
    15 min and the block is unfinished, the block is freed and redone by
    another worker. Duplicated work is harmless, because inputs are
    deterministic.
  - Every attempt is kept, including rejected ones and why.

### Main figure (`fig_2_main_plot.py`)

- **Fixed points.** The stable fixed points of all four networks are drawn
  first, as a gray density: 200 × 200 logarithmic bins, darker where there
  are more fixed points, empty bins white (`--fixed-bins`).
- **Axes.** Both axes run from 10⁻² to 10³ (`--lim`; `--auto-lim` instead fits
  them to all cycles and chaos plus `--margin-decades`).
  - Points outside the window are not drawn.
  - Their numbers are recorded in the audit and in the statistics
    (`\FigTwoMainFixedOutside`, `\FigTwoMainMovingOutside`), so the caption can
    state them.
- **Cycles and chaos.** Limit cycles are blue open squares and chaos is
  vermillion filled triangles. They are distinguished by class, not by network.
- **Density control.** The cycles and chaos are chosen at random on square
  cells 0.2 decade wide (`--cell-decades`). The rules:
  - exactly **1000 limit cycles and 1000 chaos points** (`--per-class`);
  - each network shows as many ideal as nonideal points;
  - every occupied cell shows at least one point;
  - the points are spread as evenly as the data allow. The script first finds
    the smallest per-cell maximum that still fits 1000 + 1000 points. It then
    raises the per-cell minimum as far as possible (a cell with fewer points
    available shows all of them);
  - the points are drawn in one random order.

  The script prints the maximum and minimum it used, for example
  `at most 22 per cell …, at least 9 per cell`, and the audit records them as
  `cap` and `even_level`. Chaos is concentrated in fewer cells than cycles, so
  the chaos cells are the fullest ones.

  If fewer than 1000 of either class lie inside the axes, the script says how
  many there are; use a smaller `--per-class`. `--per-class 0` restores the
  older rule: as many points as `--max-per-cell` allows, with equal numbers of
  cycles and chaos.
- **Inset.** A histogram of the saturation Δ̄₁²/(B̄₁σ̄_ps) of all cycles and
  chaos. It is placed automatically where it hides no cycle or chaos point and
  as little fixed-point density as possible.
- **Re-plotting.** The main figure reads only the CSV. To change its look, rerun
  `fig_2_main_plot.py` alone, which takes about 2–3 minutes for 800,000 rows,
  most of it for the random balanced selection.
- **Audit.** `data_fig2/main_selection/` records:
  - the drawn points in drawing order;
  - the per-cell counts;
  - a summary.

### Supplementary figure (`fig_2_supp_plt.py`)

- **Layout.** One 2 × 4 figure, 7.0 × 3.9 in:
  - columns are (a) WR, (b) Brusselator, (c) Oregonator, (d) CS14;
  - rows are ideal (top) and nonideal (bottom);
  - the two panels of a column share axis limits.
- **Files.** `fig2_supp.png` (300 dpi) and `fig2_supp.pdf`.
  - In the PDF, all points of a panel are one embedded 300-dpi image, while the
    axes, labels and diagonal stay vector. The PDF is therefore small (about
    0.4 MB for 800,000 points) and opens in any viewer.
  - It is written under a temporary name and renamed only when complete, so an
    interrupted run never leaves a broken PDF behind.
  - `--pdf-raster` instead writes the whole figure as one 600-dpi image.
- **Points.** Every input, with no density control:
  - fixed points in light gray first;
  - then all limit cycles, in random order;
  - then all chaos points on top, in random order;
  - cycles and chaos are drawn in the network's colour.

### Verification of cycles and chaos (`verify_attractors.py`)

This is independent of the numba engine: it shares only the stoichiometry
tables and uses scipy's stiff solvers.

- **Cycles:**
  - integrate one stored period with Radau at rtol 10⁻¹²;
  - check the closure;
  - Floquet multipliers from the variational equations: one equal to 1, the
    others inside the unit circle;
  - no drift over 20 more periods.
- **Chaos:**
  - continue the stored end point for 4 × 2500 time units with LSODA;
  - λ₁ > 0 in every window;
  - the oscillation persists;
  - λ₁ agrees with the CSV.

On the 30-per-case test, all 128 cycles and all 26 chaos candidates passed.
The engine's chaos test is standard numerical evidence, not a proof. A chaotic
transient longer than the simulated time would be mislabelled, and the
10,000-unit continuation makes this unlikely.

### Numbers for the manuscript (`fig_2_stats.py`)

- **Counts.**
  - Inputs per network, mixing case and class.
  - Attempts and acceptance, with rejection reasons.
  - Points plotted in each figure. For the main figure, also the fixed points
    outside the axes, how many cycles and chaos the per-cell cap left out,
    occupied cells, axis range and points under the inset.
- **Tightness.**
  - Saturation: median, 90th percentile and maximum per case.
  - Full-bound ratio (λ₁+Δ̄₁)²/(σ̄_ps B̄₁) per class.
  - Violations and the smallest margin.
  - For chaos, the share of the allowed growth rate that is used,
    λ₁/(√(σ̄_ps B̄₁) − Δ̄₁).
  - σ̄_ps/σ̄.
- **What controls the saturation.** Spearman correlation with the drive, the
  oscillation amplitude, the Floquet modulus and Δ̄₁.
- **Ideal vs nonideal.** A KS test of the saturation per network.
- **Numerical quality.** Largest residuals, Floquet margins, and the λ₁
  standard error.
- **Re-check.** The results of `verify_attractors.py`.

### Run time and resources

Time per accepted input on one core, measured with the production settings,
including rejected attempts:

| network | ideal | nonideal |
|---|---|---|
| WR | 0.2 s | 0.3 s |
| Brusselator | 0.9 s | 1.0 s |
| Oregonator | 4.3 s | 3.9 s |
| CS14 | 1.6 s | 5.3 s |

The mean is about 2.2 s, so 800,000 inputs need about 500 CPU-hours. Three
things can change this:

- the per-core speed of the machine;
- hyperthreads counted as cores, which can make it about 1.5× slower;
- the ±20% uncertainty of the timing sample.

The steps after sampling were timed on a synthetic 800,000-row run:

| step | time | peak memory |
|---|---|---|
| merge → 2 GB CSV | 5 min | 0.3 GB |
| main figure | 2–3 min | 1.8 GB |
| supplementary figure | 50 s | 1.3 GB |
| statistics | 1.5 min | 2.9 GB |
| re-check, 200 + 200 attractors on 40 cores | 10–15 min | small |

Disk: about 3.5 GB of worker databases plus the 2 GB CSV.

numba compiles the engine on first use in about 15 s. The compiled code is
cached, so later runs start at once.
