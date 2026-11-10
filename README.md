# MECC: Mutual-Exclusion Control Correction

MECC improves locally private empirical-frequency estimation by using the
within-report structure of complete optimized unary encoding (OUE) reports.
A public split separates coefficient training from mean-zero control moments;
the base estimate uses the column sums of all reports. The client report and
privacy budget are unchanged.

This repository provides the final estimator, same-report baselines, reproducible
input preparation, compact final results, and figure-generation scripts.
All category indices are zero-based and the complete domain size is public.

## Installation

Python 3.10 or newer is required. NumPy 2.0 or newer supplies packed-bit population
counts. A CPU is sufficient.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install ".[figures]"
```

The tested environment is Python 3.13.16, NumPy 2.5.4, SciPy 1.18.1, and
Matplotlib 3.11.2. To use these dependency versions, install `requirements.txt`
before installing the package. All runtime imports are contained in `src/mecc`
or the declared dependencies.

## Quick start

Run three independent OUE randomizations of one fixed Zipf sample and compare
MECC, Calibrate-public, Calibrate-MLE, raw OUE, simplex-projected OUE, and
full-report MLE on exactly the same reports:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 mecc run \
  --distribution zipf --n 256 --d 32 --input-seed 7 \
  --epsilon 3 --report-seed 0 --repetitions 3 --full-mle \
  --output outputs/demo
python -m unittest discover -s tests -v
```

The quick example checks installation and exercises the full estimation path.
Its small input is separate from the paper conditions.

`outputs/demo` contains:

- `metrics.csv`: one row per budget, report seed, and estimator, with `mse`,
  `l1`, and `linf` against the fixed sample's empirical frequency;
- `summary.csv`: repetition means and sample standard deviations (`ddof=1`);
  a standard deviation is blank for a single repetition;
- `settings.json`: public parameters, mechanism seeds, numerical validity, and
  the full-domain MLE gap when that baseline is enabled;
- optional `estimates-eps*-seed*.npz` files when `--save-estimates` is supplied.

MSE is the squared Euclidean error divided by the full public domain size.
L1 is the sum of absolute errors; Linf is the maximum absolute error.
Pre-projection outputs are also reported as `mecc_pre` and
`calibrate_pl0_public_pre`.

## Use the decoder with existing reports

The server accepts complete OUE reports as a `uint8` matrix of shape
`(n, ceil(d/8))`, with category bits packed in little-endian order within each
byte. Padding bits are ignored. Raw private values and evaluation truth are
not arguments to the decoder.

```python
import numpy as np
from mecc import decode

packed = np.load("reports.npy", allow_pickle=False)
n, d, epsilon = len(packed), 128, 3.0
estimates, diagnostics, state = decode(packed, n, d, epsilon)
frequency = estimates["mecc"]
base_frequency = estimates["calibrate_public"]
```

`estimates` also contains `mecc_pre` and `calibrate_pre`. `state` provides the
training/control indices, selected controls `H`, coefficient matrix `beta`,
control moments `T`, and the correction. `diagnostics` reports public parameters
and numerical checks. To generate experimental reports, use
`mecc.privatize(values, d, epsilon, seed=...)`. Its PCG64 seed makes the
simulation reproducible.

## Public parameters

`mecc run --help` lists all options. The default scientific settings are:

| Option | Default | Meaning |
|---|---:|---|
| `--train-fraction` | 0.2 | Training size `ceil(n*tau)`; remaining reports form the control moments |
| `--max-controls` | 32 | At most `min(K,d)` training-selected control categories; diagonal terms excluded |
| `--split-seed` | 20261109 | Public PCG64 permutation of reports |
| `--tie-seed` | 20261110 | Independent public category ranking to break selection ties |
| `--tail-tolerance` | 1e-14 | Bound used for finite binomial-PMF support truncation; 0 uses full support |
| `--input-seed` | 7 | Fixed synthetic sample generation, separate from report randomization |
| `--report-seed` | 0 | First report seed; repetitions use consecutive seeds |
| `--repetitions` | 3 | Number of independent report randomizations |
| `--mle-max-iter` | 200 | Full-report MLE outer active-set iteration limit |
| `--mle-tol` | 1e-7 | Full-domain simplex gradient gap threshold |
| `--mle-max-coordinate-steps` | 5000 | Maximum feasible coordinate updates if the active-set solve needs refinement |

MECC and Calibrate-public use the zero-inclusive prior proportional to
`(k+1)^(-alpha)`, `k=0,...,n`, with alpha fixed by prior mean `n/d`.
Calibrate-MLE fits alpha from privatized marginal counts using the public grid
and local scalar searches in `baselines.py`. Full-report MLE maximizes the
per-report mixture likelihood using active-set L-BFGS-B (up to 300 inner steps)
and feasible simplex coordinate updates. A solve is successful only when its
full-domain gap meets the requested tolerance. The CLI raises an error if it
does not converge.

## Reproduce the paper inputs and conditions

`configs/final.json` fixes the six inputs, six privacy budgets, 100 report seeds,
and decoder parameters. Four synthetic inputs are generated locally by inverse
CDF sampling, with the exact fixed seeds in that file. Input fingerprints are
checked before running the paper settings.

For one complete synthetic condition:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/run_paper.py \
  --scenario zipf-d128-n10000 --epsilon 3 --full-mle \
  --output outputs/paper
```

Use `--repetitions 1` for a short execution check, or omit `--epsilon` to run
all six budgets. Large inputs and full-report MLE take substantially longer
than the quick example. MECC's exact public shrinkage table uses quadratic
work in the sample size when first constructed and is cached within a process.

The real datasets are supplied by the user and are not bundled:

1. Download and extract [MovieLens 1M](https://grouplens.org/datasets/movielens/1m/).
   The reader chooses each user's earliest rating, breaks timestamp ties by
   smallest movie ID, orders users by ID, and maps movie IDs 1–3952 to 0–3951.
   The resulting input has 6040 reports and retains the full public movie range.
2. Download [Kosarak from the FIMI repository](https://www.cs.rpi.edu/~zaki/Workshops/FIMI/data/), file `kosarak.dat.gz`.
   The reader expands transaction items in file order into 8,019,015 clicks,
   then samples 1024 clicks without replacement using PCG64 seed 1069001 and
   sorts the selected positions to retain their original file order.
   IDs 1–41270 map to 0–41269. The privacy unit is one sampled click, treated
   as one simulated reporter.

```bash
mecc prepare --dataset movielens --source ml-1m/ratings.dat --output data/movielens.npy
mecc prepare --dataset kosarak --source kosarak.dat.gz --output data/kosarak.npy
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/run_paper.py \
  --data-dir data --full-mle --output outputs/paper
```

For other inputs, save a one-dimensional integer `.npy` array with values in
`0,...,d-1`. Keep absent categories in the public domain:

```bash
mecc run --values data/custom.npy --d 128 --epsilon 1 3 5 \
  --repetitions 100 --output outputs/custom
```

The bundled code runs the direct OUE comparisons. The cross-mechanism native
baseline envelope in the paper is included as a result column, rather than as
a single executable estimator or a reimplementation of the other mechanisms.

## Bundled final results and figures

The main comparison CSVs in `results/` cover all six inputs and all six budgets,
including conditions with negative or negligible gains. Each condition uses 100
independent OUE randomizations of its fixed empirical sample. Additional coordinate
and row-alignment diagnostics cover all six inputs at epsilon 5. The files contain
aggregate or per-repetition error contributions, not private samples or report matrices.

| File | Contents |
|---|---|
| `summary.csv` | Means and sample standard deviations for each included estimator; `best_native_method` and `best_native_mse` identify the minimum native baseline mean for that condition; `mse_ratio_to_best_native` divides the estimator's mean MSE by this envelope |
| `paired.csv` | MECC-minus-control mean differences, unadjusted paired 95% t intervals, and relative mean improvements for MSE/L1/Linf |
| `mechanism-summary.csv` | Oracle gain, coefficient regret, predicted and observed pre-projection MSE changes, and their repetition-mean intervals |
| `coordinate-gains.csv` | Contributions from four empirical-frequency groups, before and after projection, with paired 95% t intervals over 100 repetitions |
| `coordinate-gains-per-seed.csv` | Per-repetition group contributions underlying `coordinate-gains.csv`, retaining all nonempty groups |
| `row-alignment.csv` | Ten same-report aligned/shuffled pairs per input, with per-seed excess errors and their paired shifts |
| `row-alignment-statistics.csv` | Mean aligned-to-shuffled shifts, paired 95% t intervals, and counts of repetitions with increased error |

In `paired.csv`, `relative_mean_improvement` is
`(control_mean - mecc_mean) / control_mean`; positive values favor MECC.
The intervals concern paired *differences*, not ratios, and are not adjusted
for multiple comparisons. Mechanism oracle quantities are evaluation diagnostics;
the estimator learns coefficients only from its training reports. The native
envelope is the minimum of available baseline means per condition, not one
deployable method.

The coordinate groups rank categories with positive true empirical frequency in
descending order, breaking ties by category ID: ranks 1–32, 33–128, 129 onward,
and a separate absent-category group. Both projection stages use the same
mean post-projection Calibrate-public full-domain MSE as the denominator.
Contributions are in percentage points: 100 times the group's paired squared-error
reduction divided by the full domain size and this common denominator. Empty
groups have zero coordinates and blank estimates, displayed as `n/a`.
`stage` is `pre` or `post`; `coordinates` records the group size,
`contribution_pp` is its per-seed contribution, and `mean_pp` with
`ci95_low_pp`/`ci95_high_pp` summarizes those contributions. `common_base_mse`
records the shared denominator.

The row-alignment intervention permutes control-report columns independently while
retaining the training reports and column sums. Each of the ten lines connects
the aligned and shuffled versions of one report realization. Excess errors use
the same-seed Calibrate-public error as their origin and the mean Calibrate-public
error over the ten paired repetitions as their common denominator. Positive values
indicate extra error; the paired shift is shuffled minus aligned. These intervals
also use unadjusted paired t calculations.
The per-seed columns are `aligned_change_pp`, `shuffled_change_pp`, and
`paired_shift_pp`; `reports_worse` counts positive paired shifts.

```bash
python scripts/plot_results.py
```

This recreates six final figures in `outputs/figures/`: the four-panel mechanism
construction, six-panel error curves, complete gain heatmap, risk decomposition
with intervals, coordinate-group contributions, and paired row-alignment effects.
The mechanism construction is computed from public formulas and chosen constants;
it illustrates the statistical mechanism rather than reporting an observed dataset.
PNG files use 300 DPI and PDFs retain vector text and lines.
The figures use Times New Roman when installed and otherwise DejaVu Serif;
the selected font family is recorded in the generated manifest. Fonts are not bundled.

## Repository layout

```text
src/mecc/       Estimator, OUE reports, numerical methods, evaluation, and CLI
configs/       Final paper settings and fixed-input fingerprints
scripts/       Paper-condition runner and result-to-figure script
tests/         Report, privacy, control-moment, numerical, and data-reader checks
results/       Compact final aggregate metrics
NOTICE.md      Method and dataset attribution; licensing status
```

The release contains no author identities, institutional contacts, repository history,
raw datasets, or machine-specific configuration. Scientific references and public
dataset attribution remain in `NOTICE.md`.
