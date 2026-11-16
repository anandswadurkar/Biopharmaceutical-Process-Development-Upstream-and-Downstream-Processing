# Biopharmaceutical process development: upstream and downstream

An interactive notebook workbench for mammalian-cell bioprocess development. It takes a
fed-batch culture through purification to release, with the data-driven layer the field
actually uses on top: MVDA and ML on batch records, a Raman soft sensor, a hybrid
mechanistic/ML model, global sensitivity and uncertainty propagation.

**The notebook asks a question before every step.** Answer it, or press Enter to take
the default from the system config. Run it headless and the same defaults are used
without prompting, so the committed outputs are reproducible. Answers are recorded,
shown at the end, and can be saved and replayed.

---

## Scope

**In scope**

| Area | What is modelled |
|---|---|
| Upstream | Fed-batch CHO culture: unstructured Monod kinetics, lactate/ammonium inhibition, the metabolic shift to lactate consumption, temperature shift, feedback and fixed-bolus feeding |
| Downstream | Protein A capture by a lumped kinetic model (tanks in series + Langmuir), CEX/AEX polishing, viral filtration, UF/DF |
| Data-driven | PLS, ridge, random forest and Gaussian process on batch KPIs; PLS soft sensor on simulated Raman spectra; serial hybrid correction of specific productivity |
| Rigour | Verification against analytic solutions and mass balances; Sobol sensitivity; Monte Carlo UQ; batch-grouped cross-validation with a baseline beside every score |
| QbD | Release tables against CQA criteria, I-MR control charts, Cp/Cpk capability, comparability with a stated detection limit |

**Out of scope.** Glycosylation and other post-translational modification networks,
viral clearance validation, media design, perfusion and continuous operation, GMP
documentation, regulatory submission. Antibody discovery and developability (sequence
or structure based) are a different problem and are not here.

**What the models are.** Screening tools. Unstructured kinetics upstream, a lumped
model for capture, correlations for polishing. They compare options; they do not
support a filing.

---

## Install

```powershell
conda env create -f environment.yml
conda activate bioproc
pip install -e . --no-deps
python tools/install_kernel.py
```

Everything comes from conda-forge. The only pip step is the editable install of this
repo's own pure-Python package, with `--no-deps` so pip cannot pull a compiled wheel
over a conda build.

`tools/install_kernel.py` is **not optional on Windows**. A conda environment's Python
launched without activation leaves `Library\bin` off PATH, NumPy's delay-loaded BLAS
fails to resolve, and the process aborts with no traceback — the front end reports only
"the kernel died". The script installs a kernelspec that activates the environment, and
a `.pth` that fixes PATH for front ends that bypass the kernelspec. Re-run it after
recreating the environment.

### Run it

```powershell
python tools/make_notebook.py              # write the notebooks
python tools/make_notebook.py --execute    # run them headless, keeping outputs
jupyter lab notebooks/bioprocess_workbench.ipynb
```

In Jupyter, pick the **Python (bioproc)** kernel and run the cells in order. Each
section prompts; Enter takes the config default.

### Replay a run

Every run writes `answers.json` next to its figures. To reproduce someone else's run
exactly:

```powershell
$env:BIOPROC_ANSWERS = "$HOME\bioproc-data\demo_mab\answers.json"
python tools/make_notebook.py --execute
```

---

## Data sources

The notebook offers three, and states on every figure which one produced it.

| Source | What it is | Caveat it carries |
|---|---|---|
| `simulate` (default) | Batches generated from the mechanistic model with a stated ground truth | **Synthetic.** A positive control showing the analysis recovers a known answer. Not evidence about any real process |
| `file` | Your own CSV/Excel export | You map your column names explicitly and confirm units; nothing is inferred from magnitude |
| `open_dataset` | 106 CHO fed-batch cultures, 5 L–500 L, from Gangadharan et al. (2021), [doi:10.1016/j.compchemeng.2021.107226](https://doi.org/10.1016/j.compchemeng.2021.107226) | Anonymised and **0–1 normalised by its authors**, so it carries no units. The notebook refuses to attach any and reports normalised errors only |

Downloads go through `pooch` with a checksum and the access date recorded. Data lands
in `BIOPROC_DATA_DIR` (default `~/bioproc-data`), outside OneDrive and the repo.

Open bioprocess data is scarce — the review this project follows says so plainly, and
it is why the default is a positive control rather than a real dataset.

---

## Optional simulators

None is required. The notebook has a built-in model for every step, detects what is
installed, and records which engine produced each number.

| Tool | What it adds | Install |
|---|---|---|
| [CADET-Process](https://github.com/fau-advanced-separations/CADET-Process) | Rigorous chromatography (general rate model, SMB) — a reference implementation to check the lumped model against | `pip install CADET-Process` + `conda install -c conda-forge cadet` |
| [BioSTEAM](https://biosteam.readthedocs.io/) | Techno-economic analysis and LCA: cost of goods, utilities, equipment sizing | `pip install biosteam` |
| [IDAES](https://github.com/IDAES/idaes-pse) | Equation-oriented flowsheet modelling and optimisation on Pyomo | `conda install -c IDAES-PSE idaes-pse` |
| [DWSIM](https://github.com/DanWBR/dwsim10) | Steady-state and dynamic flowsheeting (.NET desktop) | Installer; drive via its Python interface |

**Install these into a separate environment**, not on top of `bioproc`. A pip build of a
compiled dependency over a conda environment is how the silent crashes start on this
platform.

---

## Studying your own process

Copy a config and edit it — the analysis code is identical for every system:

```powershell
copy configs\systems\demo_mab.toml configs\systems\my_process.toml
$env:BIOPROC_SYSTEM = "my_process"
```

A config holds the cell line's kinetics, initial state, feed, column and resin,
acceptance criteria, and the default answer to every question the notebook asks. Two
demo systems ship with it (`demo_mab`, `demo_intensified`) so the code is exercised on
more than one process.

---

## Pitfalls this repo protects against

Each of these costs a wrong number if ignored.

- **A random train/test split leaks.** Time points from one batch are nearly identical
  neighbours; split by batch or every model looks good. All cross-validation here is
  grouped by `batch_id`.
- **Accuracy flatters imbalanced data.** Batch failure runs around 10%, where "always
  pass" scores 90%. MCC and balanced accuracy are reported with a confusion matrix;
  raw accuracy is never the headline.
- **A score without a baseline is not a result.** Every model is reported beside the
  mean predictor, and `skill_vs_mean ≤ 0` is called out in words.
- **"No significant difference" is meaningless without a detection limit.** Every
  comparison reports the smallest difference the design could have resolved at 80%
  power.
- **Tanks-in-series is a model parameter, not a mesh.** The capture model's dispersion
  comes from the tank count, so refining it is not convergence. The notebook shows DBC₁₀
  moving across tank counts and labels that spread as model uncertainty.
- **Clearance must account for product loss.** Losing product *concentrates* the
  impurity that remains, so every log-reduction is divided by the step yield.
- **Normalised data has no units.** The open dataset is 0–1 scaled; absolute values
  from it are meaningless, and the notebook refuses to pretend otherwise.
- **A process failing spec is not a code failure.** Missed criteria appear as
  `OUT OF SPEC` findings; only a broken calculation raises.

---

## Status

Built 2026-11-15. First working version, executing end to end.

| | |
|---|---|
| Package | `src/bioproc`, 11 modules |
| Notebooks | 2 × 47 cells, generated from `tools/make_notebook.py` |
| Sections | 14, each gated on a question; 16 figures per notebook |
| Tests | 141 passing: known answers, hypothesis invariants, invalid inputs, conservation, rate plausibility |
| Verification | 6 checks pass per run: growth vs. analytic, Langmuir closed form, glucose and product mass balances, diafiltration `1 − e^(−N)`, step convergence |
| Runtime | ~50 s per notebook (Windows laptop, CPU only) |

Demo results on `demo_mab` (synthetic, seed 20261115, 500 L, glucose-setpoint feeding):

| Quantity | Value |
|---|---|
| Peak VCD | 20.6 ×10⁶ cells/mL |
| Day-14 titer | 4,662 mg/L, 2,552 g harvested |
| Viability at harvest | 69% (95% at inoculation) |
| Specific productivity | 27.9 pg/cell/day, IVCD 167 ×10⁶ cell·d/mL |
| DBC₁₀ at 4 min residence | ~40 g/L resin, 10 cycles on a 31.8 L column |
| Overall downstream yield | 71.1% → 1,815 g drug substance |
| Dominant sensitivity | `mu_max` (ST = 0.72 of titer variance) |
| Harvest with uncertainty | 2.3 [1.53, 3.31] ×10³ g at 10% input CV |

These are a positive control for the analysis, not a claim about any process.

**Known limitations, stated rather than hidden.**

- **No validation.** Everything here is verification plus screening. No bench or plant
  data has been through it. The kinetic parameters are illustrative CHO-like values
  chosen so the specific rates land in reported ranges — not a fit — and the resin
  parameters are not from a measured isotherm.
- **Capability is weak on the demo campaign** (best group Cpk ≈ 1.08 against the
  2,000 mg/L criterion). That is a property of the synthetic campaign's 8% lot
  variation, not a tuned result.
- **The cost figure is direct materials only** (~$8/g), roughly 3–16% of published mAb
  cost of goods. Facility, labour and QC dominate and are not modelled.
- **Baseline correction does not help the simulated spectra.** PLS absorbs a smooth
  fluorescence background as a latent variable, so ModPoly correction makes the
  calibration worse here. The notebook measures this rather than assuming.

---

## Reading

- Khuat, Bassett, Otte, Grevis-James, Gabrys (2023). *Applications of Machine Learning
  in Biopharmaceutical Process Development and Manufacturing.*
  [arXiv:2310.09991](https://doi.org/10.48550/arXiv.2310.09991) — the review this
  project's structure follows.
- Narayanan et al. (2019). *A new generation of predictive models: the added value of
  hybrid models.* [doi:10.1002/bit.27097](https://doi.org/10.1002/bit.27097)
- Narayanan et al. (2021). *Hybrid models for the simulation and prediction of
  chromatographic processes for protein capture.*
  [doi:10.1016/j.chroma.2021.462248](https://doi.org/10.1016/j.chroma.2021.462248)
- Kyriakopoulos et al. (2017). *Kinetic modeling of mammalian cell culture
  bioprocessing.* [doi:10.1002/biot.201700229](https://doi.org/10.1002/biot.201700229)
- Gagnon et al. (2011). *High-end pH-controlled delivery of glucose.*
  [doi:10.1002/bit.23072](https://doi.org/10.1002/bit.23072)
- Mante et al. (2019). *A heuristic approach to handling missing data in biologics
  manufacturing databases.*
  [doi:10.1007/s00449-018-02059-5](https://doi.org/10.1007/s00449-018-02059-5)

## Licence and citation

MIT (`LICENSE`). If you use it, cite via `CITATION.cff`.
