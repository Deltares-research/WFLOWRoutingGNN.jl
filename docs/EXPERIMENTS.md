This document serves as an archive of experiments, listing evaluation metrics and some basic diagnoses and hypotheses of each experiment.

Experiment configs live under `experiments/`. Newest at the top.

---

# OLD STATUS AS OF 27-08-2026

*Everything below this header predates the introduction of these structured
status logs; it back-fills experiments run before 27-08-2026.*

---

## Mass-balance on/off A/B sweep — `sava_small_v081_mb_sweep`

Identical setup (8 layers, hidden 64, mlp 2, batch 8, 250 epochs, curriculum
`steps = [1,2,5,8,10]`, `lr_start ≈ 1.35e-4`, `grad_clip = 1.0`), varying **only**
`enforce_mass_balance`.

| Metric | MB on | MB off |
|---|---|---|
| final train / val loss | 0.0285 / 0.0242 | 0.0441 / 0.0438 |
| date-range rollout Q RMSE | **27.5** | ≈ 7.5×10⁵ |
| date-range rollout H RMSE | **6.55** | ≈ 1.1×10⁴ |
| full val rollout (1906 steps) | finite | Float32 overflow → Inf from step ≈1571 |

**Diagnosis:** teacher-forced training only ~1.8× worse without MB, but the
free-running rollout diverges exponentially. Without an algebraic anchor a small
consistent bias compounds multiplicatively. **The hard MB stabilises at
inference time.** Basis for the hard-constraint decision.

## θ-method sweep — `sava_small_v081_theta_sweep`

Explores the implicit/explicit/trapezoidal (θ) discretisation; `θ* = σ_h/(dt·σ_q)`
cancels the q↔h gradient stiffness. Default shipped: `mb_theta = 1.0` (fully
implicit). See [notes/mass_balance_stability_notes.md](notes/mass_balance_stability_notes.md).

## h_loss_scale variants — `sava_small_v081_increment`, `*_mb_prior_increment`, `*_mb_no_prior_absolute`, `*_mb_exp_prior`

Compares how the h loss term is scaled/weighted (increment vs absolute, with/without
Q rating prior). Informed the `h_loss_weight = σ_h/(dt·σ_q)` choice.

## Hyperparameter search — `sava_small_v081_hpar_search_hps09`, `sava_v081_autotune`

LR-range test / autotune and hidden-dim / layer-count / mlp-depth search on the
small basin.

## Full-basin scale-up — `sava_full_v081` (slurm-246431)

Large run: 8 SparseConv layers, hidden 64, mlp 2, `enforce_mass_balance=true`,
`mb_theta=1.0`, 75,137 params; graph 8235 nodes / 11322 timesteps; 7918 train /
2262 val / 1132 test windows.

**Outcome — training divergence.** Single-step loss blew up around epoch 2
(~170 LR-backoff events fired). **Root cause:** heavy-tailed normalised targets
(extreme cells at hundreds of σ) amplified by the squared loss; larger basins
have heavier tails. *Not* the σ-ratio (identical across models), *not* LR step
size (peak LR only ~2e-7), *not* N-scaling (loss is a mean over nodes).
**Remedies** → variance-stabilising input transform + peak-weighted Huber loss;
tracked in [TODO.md](TODO.md) and [notes/peak_accuracy_todo.md](notes/peak_accuracy_todo.md).

## Smoke / regression runs — `test_run`, `test_run_full`, `test_sava_small(_v081)`, `test_sava_v081`

Small fixtures used for quick end-to-end validation of preprocessing, training,
rollout and postprocessing.