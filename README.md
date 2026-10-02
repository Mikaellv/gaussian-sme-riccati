# gaussian-sme-riccati

Solver for the **stochastic master equation of a continuously monitored Gaussian mode network**,
as a Jupyter notebook.

For linear dynamics with general-dyne (e.g. heterodyne) monitoring, the SME reduces to

* a deterministic **Riccati equation** for the conditional covariance matrix σ, whose steady state
  is found with `scipy.linalg.solve_continuous_are`, and
* a linear **Itô SDE** for the conditional mean, integrated with Euler–Maruyama.

| file | content |
|---|---|
| [`gaussian_sme.ipynb`](gaussian_sme.ipynb) | solver functions + worked example (8-mode squeezing network under heterodyne) |
| [`SOLUTION.md`](SOLUTION.md) | derivation, conventions, numerical method and validation, results |
| `original/riccatiequation.py` | the original script this was refactored from |
| `figures/` | figures produced by the notebook |

## Quick start

```bash
pip install -r requirements.txt
jupyter lab gaussian_sme.ipynb
```

Change the parameter cell in Sec. 4 (squeezing `r`, phases `theta`, `beam_splitter`, `kappa`,
`bath_n`, measurement `s_meas, phi_meas`) and re-run.

## Main functions

```python
A, D, B, E = sme_matrices(H, C, sigma_in, sigma_m)   # drift, diffusion, monitoring
sigma_c   = steady_conditional_cm(A, D, B, E)         # CARE (stabilising solution)
sigma_u   = steady_unconditional_cm(A, D)             # Lyapunov
t, sig_t  = riccati_trajectory(sigma0, A, D, B, E, t_max)
t, r, sig = mean_trajectory(r0, sigma0, A, D, B, E, t_max, dt, rng)
```

## Fixes relative to the original script

* `solve_continuous_are` solves `aᵀX + Xa − …`, so the drift must be passed **transposed**.
  The original passed `Ã` instead of `Ãᵀ`. The resulting matrix did not satisfy the model's
  Riccati equation (residual 1.35) and its log-negativity map is off by up to 0.52.
* `takeSub` had an index typo (`CM[j, j+1]` where `CM[j, i+1]` belongs). It is replaced by
  `two_mode_cm`.
