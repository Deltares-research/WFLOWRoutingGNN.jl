# Loss Functions and Evaluation Metrics for Peak-Discharge Forecasting

## Purpose and scope

This document provides a practical framework for training and evaluating machine-learning models that forecast discharge hydrographs while giving appropriate attention to rare peak-discharge events. It covers:

- deterministic, asymmetric, quantile, multi-task, probabilistic and extreme-value losses;
- construction and normalisation of peak-aware weights;
- hyperparameter-tuning strategies;
- metrics to monitor during training;
- event-level and time-series metrics for inference evaluation;
- experimental design and model-selection recommendations.

The central recommendation is to start with a **smooth, basin-relative, peak-weighted Huber loss**, compare it against **unweighted Huber** and **multi-quantile loss**, and add an exceedance-classification head only if peak detection remains a distinct failure mode. Model selection should use peak-specific validation metrics subject to constraints on false alarms and whole-hydrograph performance, rather than weighted validation loss alone.

---

## 1. Notation

| Symbol | Meaning |
|---|---|
| $b$ | Basin index |
| $t$ | Time index |
| $e$ | Independent hydrological event index |
| $y_{b,t}$ | Observed discharge |
| $\hat y_{b,t}$ | Deterministic predicted discharge |
| $q_{\tau,b,t}$ | Predicted conditional quantile at level $\tau$ |
| $u_b$ | Basin-specific high-flow threshold |
| $s_b$ | Basin-specific discharge scale |
| $r(y_{b,t})$ | Peak relevance function |
| $w_{b,t}$ | Training weight |
| $z_{b,t}$ | Indicator that discharge exceeds $u_b$ |
| $p_{b,t}$ | Predicted exceedance probability |
| $Q_{\max,e}$ | Observed maximum discharge for event $e$ |
| $\hat Q_{\max,e}$ | Predicted maximum discharge for event $e$ |

All target-derived quantities, including $u_b$, $s_b$ and class frequencies, should be estimated using **training data only** within each split or cross-validation fold.

---

## 2. Design principles

1. **Optimise the operational objective, not class balance itself.** Peak accuracy, missed events, false alarms, lead time and probabilistic reliability are distinct objectives.
2. **Use basin-relative thresholds and scales.** A single absolute-discharge threshold can let large basins dominate regional training.
3. **Treat independent events as the unit of evidence.** Adjacent high-flow timesteps from one hydrograph are not independent rare events.
4. **Normalise weighted losses.** Dividing by the sum of weights prevents the gradient scale from changing mechanically with the weighting strength.
5. **Validate on the natural data distribution.** Do not balance validation or test sets.
6. **Do not select on one aggregate hydrological score.** NSE, KGE, peak bias, timing and event-detection metrics reveal different behaviours. Mizukami et al. found that calibration against NSE could still reproduce annual peaks poorly, while KGE provided a better compromise in their multi-basin study [1].
7. **Report basin-level distributions.** Pooled metrics can be dominated by a few large rivers, long records or extreme events.

---

# Part I: Loss functions

## 3. Baseline deterministic losses

### 3.1 Mean squared error

$$
L_{\mathrm{MSE}}=\frac{1}{N}\sum_{b,t}(y_{b,t}-\hat y_{b,t})^2
$$

**Pros**

- Simple, smooth and widely supported.
- Strongly penalises large magnitude errors.
- Often provides stable gradients near the optimum.

**Cons**

- Can be dominated by isolated extremes, sensor errors or a few large basins.
- Does not distinguish peak underprediction from overprediction.
- Does not guarantee good event timing or event detection.
- Squared-error emphasis alone does not ensure accurate annual peaks [1].

**Relevant tuning**

- Usually none within the loss itself.
- Learning rate, target transformation, gradient clipping and target scaling remain important.

### 3.2 Mean absolute error

$$
L_{\mathrm{MAE}}=\frac{1}{N}\sum_{b,t}|y_{b,t}-\hat y_{b,t}|
$$

**Pros**

- More robust than MSE to extreme residuals and erroneous observations.
- Easy to interpret in discharge units.

**Cons**

- Does not strongly distinguish moderate from very large errors.
- Constant-magnitude gradients away from zero can make fine optimisation less smooth.
- May still prioritise the densely represented ordinary-flow regime.

### 3.3 Huber loss

Let $e=y-\hat y$. Then

$$
\ell_\delta(e)=
\begin{cases}
\frac{1}{2}e^2, & |e|\leq\delta,\\
\delta\left(|e|-\frac{1}{2}\delta\right), & |e|>\delta.
\end{cases}
$$

and

$$
L_{\mathrm{Huber}}=\frac{1}{N}\sum_{b,t}\ell_\delta(y_{b,t}-\hat y_{b,t}).
$$

**Pros**

- MSE-like near zero and MAE-like for large residuals.
- Reduces sensitivity to isolated, very large errors.
- A strong base loss for peak weighting.

**Cons**

- Introduces the scale parameter $\delta$.
- A poor $\delta$ can make the loss behave almost entirely like MAE or MSE.
- Does not by itself prioritise peaks.

**Tuning $\delta$**

Use the same units as the model target. Candidate approaches are:

- set $\delta$ from residual quantiles of an unweighted baseline;
- express $\delta$ in normalised target units;
- tune a compact logarithmic grid;
- if basin scales vary strongly, compute the loss on basin-normalised targets or use a basin-relative $\delta_b$.

A practical small grid in standardised target units is $\delta\in\{0.5,1,2\}$, but the final range should be informed by baseline residuals.

---

## 4. Smooth peak-weighted regression

This is the recommended first peak-aware loss.

### 4.1 Relevance function

A smooth threshold-exceedance relevance function is

$$
r_{b,t}=
\left[
\max\left(0,\frac{y_{b,t}-u_b}{s_b}\right)
\right]^\gamma.
$$

The associated weight is

$$
w_{b,t}=1+\lambda r_{b,t}.
$$

The normalised weighted loss is

$$
L_{\mathrm{weighted}}=
\frac{
\sum_{b,t}w_{b,t}\,\ell(y_{b,t},\hat y_{b,t})
}{
\sum_{b,t}w_{b,t}
}.
$$

Suitable base errors $\ell$ are MSE, MAE or Huber. **Peak-weighted Huber** is a conservative starting point.

### 4.2 Constructing the threshold $u_b$

Possible definitions are:

- an operational warning threshold;
- a basin-specific empirical percentile;
- a peaks-over-threshold threshold;
- a flow associated with a specified return period;
- multiple threshold levels associated with operational warning categories.

**Recommendation:** use the operational threshold if one exists and is consistent with the application. Otherwise, compare a small set of training-period percentiles such as the 95th, 98th and 99th percentiles. These are tuning candidates, not universal standards.

**Caution:** a percentile gives comparable sample proportions across basins but not necessarily comparable hydrological impacts.

### 4.3 Constructing the scale $s_b$

Candidates include:

$$
s_b=u_b,
$$

$$
s_b=\operatorname{IQR}(y_b),
$$

$$
s_b=\operatorname{SD}(y_b),
$$

or a hydrologically meaningful reference such as mean discharge or mean annual flood. For regional models, basin-area-normalised runoff can also reduce scale imbalance.

**Recommendation:** choose a robust basin-specific scale such as IQR, or optimise on basin-normalised runoff and transform predictions back to discharge.

### 4.4 Alternative continuous relevance functions

#### Logistic transition

$$
r(y)=\frac{1}{1+\exp[-k(y-u_b)]}.
$$

This gives a smooth transition around the threshold. The slope $k$ controls transition sharpness.

#### Percentile-rank relevance

If $F_b$ is the empirical training-period discharge CDF:

$$
r(y)=\left[F_b(y)\right]^\gamma.
$$

This increases weights throughout the upper distribution rather than only above a hard threshold.

#### Impact-based relevance

If operational impacts can be mapped to discharge:

$$
r(y)=\frac{C(y)}{C_{\mathrm{ref}}},
$$

where $C(y)$ is a monotonic impact or consequence function. This is the most decision-relevant construction but requires defensible impact information.

### 4.5 Weight capping

To prevent one observation from dominating:

$$
w'_{b,t}=\min(w_{b,t},w_{\max}).
$$

Tune $w_{\max}$ by inspecting the weight distribution and gradient contributions. The cap should not be chosen solely to improve validation loss.

### 4.6 Event-normalised weighting

A long flood can contain many high-weight timesteps. For event $e$, define

$$
\tilde w_{e,t}=\frac{w_{e,t}}{\sum_{t\in e}w_{e,t}}.
$$

Optionally multiply the event by a severity weight $v_e$:

$$
L_{\mathrm{event\text{-}normalised}}=
\frac{1}{|E|}\sum_{e\in E}v_e\sum_{t\in e}\tilde w_{e,t}\ell(y_{e,t},\hat y_{e,t}).
$$

This prevents event duration from determining event importance.

### 4.7 Pros and cons

**Pros**

- Small implementation change.
- Works with most differentiable regression architectures.
- Weight construction is transparent and auditable.
- Smooth relevance avoids a discontinuity immediately around $u_b$.
- Can combine basin normalisation, event normalisation and operational thresholds.

**Cons**

- Large weights may cause systematic overprediction or unstable gradients.
- Weighting observed targets can alter the effective training distribution.
- Results depend on the threshold, scaling and event definition.
- It still does not explicitly model event occurrence, timing or probability.

### 4.8 Tuning strategy

Begin with

$$
\lambda\in\{0,0.5,1,2,4,8\},\qquad
\gamma\in\{0.5,1,2\}.
$$

Interpretation:

- $\gamma<1$: gives substantial weight to moderate threshold exceedances;
- $\gamma=1$: linear increase above the threshold;
- $\gamma>1$: concentrates weight on the most extreme observations.

Tune $u_b$, $\lambda$, $\gamma$, $\delta$ and any cap $w_{\max}$ using a validation scorecard. Do not optimise all five simultaneously with a large unconstrained search. A staged strategy is preferable:

1. Fit unweighted Huber and choose a reasonable $\delta$.
2. Fix $\gamma=1$ and tune $u_b$ and $\lambda$.
3. Inspect peak errors, false alarms, weight distributions and gradient concentration.
4. Tune $\gamma$ only if the severity response is inadequate.
5. Introduce $w_{\max}$ or event normalisation if a few events dominate.

### 4.9 Diagnostic loss contribution

Monitor the fraction of weighted loss from peaks:

$$
C_{\mathrm{peak}}=
\frac{
\sum_{y_{b,t}>u_b}w_{b,t}\ell_{b,t}
}{
\sum_{b,t}w_{b,t}\ell_{b,t}
}.
$$

Also monitor the fraction of gradient norm attributable to peak observations or event windows. A very small fraction of data accounting for nearly all gradient magnitude indicates excessive weighting, label problems or both.

---

## 5. Asymmetric peak-weighted regression

Let $e=y-\hat y$, so $e>0$ means underprediction. Define

$$
L_{\mathrm{asym}}=
\frac{
\sum_{b,t}w_{b,t}
\begin{cases}
a\,\ell(e_{b,t}), & e_{b,t}>0,\\
\ell(e_{b,t}), & e_{b,t}\leq0
\end{cases}
}{
\sum_{b,t}w_{b,t}
},
$$

where $a>1$ penalises underprediction more strongly.

### High-flow-only asymmetry

A safer variation activates the asymmetry gradually:

$$
a_{b,t}=1+(a_{\max}-1)\,\bar r(y_{b,t}),
$$

where $\bar r\in[0,1]$ is a bounded relevance score. Then replace $a$ with $a_{b,t}$ in the loss.

### Pros

- Directly represents a larger penalty for missed or underestimated peaks.
- Easy to add to a weighted Huber or MSE objective.
- Produces an interpretable trade-off parameter.

### Cons

- Can make systematic overprediction the easiest strategy.
- An arbitrary asymmetry ratio can be difficult to defend.
- May degrade moderate-flow bias and false-alarm performance.
- Does not automatically give calibrated exceedance probabilities.

### Tuning

Try

$$
a\in\{1,1.5,2,4\}.
$$

Select $a$ using the trade-off between:

- missed-event frequency;
- signed and absolute peak error;
- false-alarm ratio;
- moderate-flow bias;
- whole-hydrograph performance.

If operational costs are available, translate them into a range of plausible asymmetry ratios. Otherwise, present a Pareto curve rather than claiming one ratio is universally optimal.

---

## 6. Quantile loss

For quantile level $\tau\in(0,1)$, the pinball loss is

$$
L_\tau(y,q_\tau)=
\begin{cases}
\tau(y-q_\tau), & y\geq q_\tau,\\
(1-\tau)(q_\tau-y), & y<q_\tau.
\end{cases}
$$

For multiple quantiles $\mathcal T$:

$$
L_{\mathrm{MQ}}=
\sum_{\tau\in\mathcal T}v_\tau
\frac{1}{N}\sum_{b,t}L_\tau(y_{b,t},q_{\tau,b,t}).
$$

### Interpretation

For equal-magnitude errors at $\tau=0.9$, underprediction costs nine times as much as overprediction. Unlike an ad hoc asymmetric point loss, pinball loss has a clear conditional-quantile interpretation when used without target-dependent weighting.

### Recommended initial quantiles

$$
\mathcal T=\{0.1,0.5,0.9,0.95\}.
$$

Add $0.99$ only if there are enough independent high-flow events to estimate it reliably.

### Pros

- Produces decision-relevant upper-tail forecasts.
- Represents uncertainty without specifying a complete parametric distribution.
- Directly supports empirical calibration checks.
- Multiple quantiles can approximate CRPS [8].

### Cons

- Extreme quantiles are data hungry.
- Separate outputs can cross.
- A finite set of quantiles is not a complete distribution.
- Target-dependent peak weighting changes the quantile interpretation.

### Quantile crossing

A predicted set should satisfy

$$
q_{\tau_1}\leq q_{\tau_2}\quad\text{for}\quad\tau_1<\tau_2.
$$

Preferred remedies are:

- monotonic output parameterisation;
- non-negative increments between successive quantiles;
- a crossing penalty;
- post-processing, if architectural enforcement is unavailable.

### Tuning

- Start with equal $v_\tau$.
- Check empirical coverage before increasing upper-quantile weights.
- Add or remove quantile levels based on operational decisions and event support.
- Select models using quantile score, coverage and interval width together.

### Caution on weighted quantile loss

A loss such as

$$
\sum_{b,t,\tau}w(y_{b,t})v_\tau L_\tau(y_{b,t},q_{\tau,b,t})
$$

targets quantiles under a reweighted distribution. If probability calibration is important, first fit ordinary quantile loss and evaluate whether additional tail weighting is necessary.

---

## 7. Multi-task flow, exceedance and event-peak loss

Let

$$
z_{b,t}=\mathbb{1}(y_{b,t}>u_b).
$$

A composite objective is

$$
L=
\lambda_Q L_{\mathrm{flow}}
+\lambda_E L_{\mathrm{exceedance}}
+\lambda_P L_{\mathrm{event\ peak}}
+\lambda_T L_{\mathrm{timing}}.
$$

Do not introduce all terms immediately. A practical initial version uses $L_{\mathrm{flow}}+\lambda_E L_{\mathrm{exceedance}}$.

### 7.1 Weighted binary cross-entropy

$$
L_{\mathrm{WBCE}}=
-\frac{1}{N}\sum_{b,t}
\left[
\alpha z_{b,t}\log p_{b,t}
+(1-z_{b,t})\log(1-p_{b,t})
\right].
$$

A frequency-based initial positive weight is

$$
\alpha_0=\frac{N_-}{N_+}.
$$

Where possible, estimate prevalence from independent event opportunities rather than counting every above-threshold timestep as an independent positive.

Tune with

$$
\alpha\in\{0.25,0.5,1,2\}\times\alpha_0.
$$

### 7.2 Focal loss for exceedance

For true-class probability $p_t$:

$$
L_{\mathrm{focal}}=-\alpha_t(1-p_t)^\gamma\log(p_t).
$$

Focal loss downweights well-classified examples and was introduced to address extreme foreground-background imbalance [4].

Tune

$$
\gamma\in\{0,1,2,3\},
$$

where $\gamma=0$ corresponds to cross-entropy, plus a small grid for $\alpha_t$.

**Use focal loss when:** easy non-events dominate optimisation even after reasonable class weighting.

**Caution:** evaluate probability calibration separately.

### 7.3 Event peak-magnitude loss

For event windows $e$:

$$
L_{\mathrm{peak}}=
\frac{1}{|E|}\sum_{e\in E}
\left(Q_{\max,e}-\hat Q_{\max,e}\right)^2.
$$

Possible definitions are

$$
Q_{\max,e}=\max_{t\in e}y_t,
\qquad
\hat Q_{\max,e}=\max_{t\in e}\hat y_t.
$$

A hard maximum can give sparse gradients. Alternatives include a dedicated peak head or a smooth maximum approximation:

$$
\operatorname{smax}_\kappa(x_1,\ldots,x_T)
=
\frac{1}{\kappa}\log\left(\sum_t e^{\kappa x_t}\right).
$$

### 7.4 Peak timing loss

If differentiable timing supervision is needed, predict time-to-peak with a separate head:

$$
L_{\mathrm{timing}}=|T_{\mathrm{peak},e}-\hat T_{\mathrm{peak},e}|.
$$

Using the `argmax` of the hydrograph directly is generally not differentiable. A soft-argmax or separate timing output is more suitable for training.

### Pros

- Separates occurrence, magnitude and timing.
- Exceedance probabilities directly support warning thresholds.
- Shared representations can improve learning when tasks are compatible.

### Cons

- More outputs, labels and hyperparameters.
- Task gradients can conflict.
- Task weights can be difficult to tune.
- Classification probabilities may need calibration.

### Task-weight tuning

Use a staged manual search:

1. train the flow head alone;
2. add exceedance loss with $\lambda_E$ on a logarithmic grid;
3. compare event detection, false alarms and hydrograph degradation;
4. add a peak head only if magnitude remains deficient;
5. add timing only if it is operationally important and not captured adequately.

Possible grids are

$$
\lambda_E,\lambda_P,\lambda_T\in\{0.01,0.1,1,10\},
$$

but normalise component losses or inspect their gradient norms before interpreting these numerical values.

---

## 8. Hydrograph-structured losses

Pointwise errors may penalise a slightly shifted but otherwise realistic hydrograph heavily while giving limited insight into event shape. A composite loss can include

$$
L=
\lambda_1L_{\mathrm{point}}
+\lambda_2L_{\mathrm{peak}}
+\lambda_3L_{\mathrm{volume}}
+\lambda_4L_{\mathrm{shape}}.
$$

### Event-volume loss

$$
L_{\mathrm{volume}}=
\frac{1}{|E|}\sum_e
\left|
\sum_{t\in e}\hat y_t-
\sum_{t\in e}y_t
\right|.
$$

### Derivative or rising-limb loss

$$
L_{\Delta Q}=
\frac{1}{N-1}\sum_t
\left|
(y_t-y_{t-1})-(\hat y_t-\hat y_{t-1})
\right|.
$$

### Pros

- Encodes peak magnitude, volume and temporal dynamics explicitly.
- Can target known hydrograph failure modes.

### Cons

- Composite objectives become harder to diagnose.
- Shape terms can conflict with pointwise accuracy.
- Event boundaries and masking require careful implementation.
- Dynamic time warping and related alignment losses may reward timing shifts that are operationally unacceptable.

**Recommendation:** add one structured term at a time and perform an ablation.

---

## 9. Distributional likelihood losses

A probabilistic model predicts distribution parameters $\theta(x)$ and minimises negative log-likelihood:

$$
L_{\mathrm{NLL}}=-\frac{1}{N}\sum_{b,t}\log f(y_{b,t}\mid\theta(x_{b,t})).
$$

Possible non-negative discharge distributions include Gamma, log-normal, Tweedie and finite mixtures. The family should be selected using residual and predictive diagnostics, not convenience alone.

### Pros

- Produces a complete predictive distribution.
- Supports probabilities, intervals and decision analysis.
- NLL is a proper score when the predictive family is correctly handled.

### Cons

- Misspecified distributions can produce misleading tails.
- Scale and mixture parameters can be unstable.
- A high likelihood does not guarantee useful extreme-event reliability.

### Tuning

- distribution family;
- number of mixture components;
- lower bound on scale;
- regularisation of very large scale or shape parameters;
- architecture of parameter heads;
- optional body-tail mixture or zero-flow component.

### CRPS training

For predictive CDF $F$ and observation $y$:

$$
\operatorname{CRPS}(F,y)
=
\int_{-\infty}^{\infty}
\left[F(z)-\mathbb{1}(y\leq z)\right]^2\,dz.
$$

CRPS evaluates the full predictive distribution and reduces to MAE for a deterministic point forecast [7]. It is a proper scoring rule [9].

---

## 10. Extreme-value-theory tail loss

For a high threshold $u_b$, excesses $x=y-u_b>0$ can be modelled with a Generalised Pareto Distribution (GPD):

$$
G(x;\sigma,\xi)
=
1-
\left(1+\xi\frac{x}{\sigma}\right)^{-1/\xi},
$$

for values satisfying $1+\xi x/\sigma>0$, with the exponential limit used when $\xi\to0$.

A tail negative log-likelihood is

$$
L_{\mathrm{GPD}}=
-\sum_{y_{b,t}>u_b}
\log g(y_{b,t}-u_b;\sigma_{b,t},\xi_{b,t}).
$$

A body-tail model can combine:

- a central discharge model below $u_b$;
- an exceedance-probability model;
- a conditional GPD for the excess.

Neural extreme-quantile regression has been used for flood-risk forecasting by combining neural networks, recurrent structure and extreme-value theory [5].

### Pros

- Provides a principled tail model.
- Supports high quantiles, exceedance probabilities and return-level estimation.
- More defensible than generic extrapolation when data are scarce in the far tail.

### Cons

- Threshold selection is difficult.
- Tail samples must represent sufficiently independent events.
- Tail-parameter estimates can be unstable.
- Distributional assumptions require careful diagnostics.
- Implementation and validation are substantially more complex.

### Tuning and diagnostics

- compare several high thresholds;
- inspect parameter stability as the threshold changes;
- decluster exceedances where necessary;
- require a minimum number of independent exceedance events;
- regularise or pool tail parameters across basins;
- validate high-quantile coverage and exceedance calibration;
- report uncertainty in return levels.

EVT is appropriate as a second-stage extension when return levels or extrapolation are explicit objectives, not merely as a way to improve ordinary hydrograph peaks.

---

# Part II: Hyperparameter tuning and model selection

## 11. Data splitting

Use splits that match the inference setting:

- forward temporal validation for future forecasting;
- grouped splits when observations share storms, stations, upstream systems or event identity;
- spatial hold-out basins for prediction in ungauged basins;
- combined spatial-temporal tests if both forms of transfer matter.

Ensure that overlapping input or target windows from the same event do not cross training and validation boundaries. Thresholds, scales, relevance functions and calibration transforms must be fitted inside the training split.

## 12. Staged tuning plan

### Stage A: establish baselines

1. Unweighted MAE, MSE and Huber.
2. Fixed architecture and data split.
3. Compare overall flow, peak magnitude, event timing and detection.

### Stage B: add peak weighting

1. Fix Huber $\delta$.
2. Compare plausible $u_b$ values.
3. Tune $\lambda$ with $\gamma=1$.
4. Inspect weight concentration and false alarms.
5. Tune $\gamma$ or introduce a cap only if needed.

### Stage C: compare uncertainty-aware loss

1. Fit multi-quantile loss.
2. Evaluate coverage, pinball loss and interval width.
3. Compare upper quantiles with deterministic peak forecasts.

### Stage D: add task specialisation

1. Add an exceedance head.
2. Compare weighted BCE and focal loss.
3. Tune the decision threshold after training on validation data.
4. Add event-peak or timing heads only when justified by an observed deficiency.

## 13. Suggested ablation matrix

| Experiment | Flow loss | Peak weighting | Additional output |
|---|---|---:|---|
| A | Huber | None | None |
| B | Huber | Smooth | None |
| C | Huber | Smooth + asymmetric | None |
| D | Multi-quantile | None | Quantiles |
| E | Huber | Smooth | Exceedance probability |
| F | Huber | Smooth | Exceedance + peak magnitude |
| G | Distributional or EVT | Model-specific | Probabilistic tail |

Use the same splits, architecture budget, random-seed policy and evaluation code wherever possible.

## 14. Early stopping and final selection

Do not early-stop on training loss or overall RMSE alone. Suitable strategies are:

### Constrained selection

Select the model with the best peak metric among models satisfying predeclared constraints, for example:

- false-alarm ratio below an operational maximum;
- no more than an accepted degradation in KGE;
- acceptable volume bias;
- acceptable quantile coverage.

### Normalised composite score

For validation errors scaled against a baseline:

$$
S=
\omega_1E_{\mathrm{peak}}
+\omega_2E_{\mathrm{timing}}
+\omega_3E_{\mathrm{false\ alarm}}
+\omega_4E_{\mathrm{overall}},
$$

where $\sum_i\omega_i=1$. Fix weights before the final comparison and report every component separately.

### Pareto selection

Present the frontier between peak error, missed-event rate, false alarms and whole-hydrograph skill. This is preferable when no single operational cost function is agreed.

---

# Part III: Metrics to track during training

## 15. Optimisation diagnostics per epoch

Track for both training and validation where applicable:

- total loss;
- every loss component separately;
- unweighted base regression loss;
- weighted and unweighted peak-region loss;
- learning rate;
- global gradient norm;
- proportion of gradients clipped;
- parameter or activation instability indicators;
- mean, median, upper quantiles and maximum of $w_{b,t}$;
- fraction of observations above $u_b$;
- fraction of total loss from peak observations, $C_{\mathrm{peak}}$;
- fraction of gradient norm from peak observations or event windows;
- number of independent peak events represented in the epoch;
- missing-value or invalid-output counts;
- distribution of predicted discharge, including negative predictions if the architecture permits them.

These indicators diagnose whether improvement is due to real peak learning or domination by a few observations.

## 16. Compact validation dashboard per epoch

### 16.1 Whole-hydrograph metrics

#### MAE

$$
\operatorname{MAE}=\frac{1}{N}\sum_i|y_i-\hat y_i|.
$$

#### RMSE

$$
\operatorname{RMSE}=
\sqrt{\frac{1}{N}\sum_i(y_i-\hat y_i)^2}.
$$

#### Percent bias

$$
\operatorname{PBIAS}=100
\frac{\sum_i(\hat y_i-y_i)}{\sum_i y_i}.
$$

State the sign convention because it differs across implementations.

#### Nash-Sutcliffe efficiency

$$
\operatorname{NSE}=
1-
\frac{\sum_i(y_i-\hat y_i)^2}
{\sum_i(y_i-\bar y)^2}.
$$

NSE is useful but should not be the only high-flow metric. High-flow calibration research has shown that an NSE objective can still perform poorly on annual peaks [1].

#### Kling-Gupta efficiency

$$
\operatorname{KGE}=
1-
\sqrt{(r-1)^2+(\alpha-1)^2+(\beta-1)^2},
$$

where

$$
r=\operatorname{corr}(y,\hat y),\qquad
\alpha=\frac{\sigma_{\hat y}}{\sigma_y},\qquad
\beta=\frac{\mu_{\hat y}}{\mu_y}.
$$

Track KGE and all three components. KGE decomposes correlation, variability and bias [2,3]. A mean-flow benchmark corresponds to approximately $-0.41$ under the original KGE formulation, not zero; NSE and KGE should not be interpreted on the same scale [3].

### 16.2 Peak-region metrics

Evaluate on observations above $u_b$ or within identified event windows:

$$
\operatorname{MAE}_{\mathrm{high}}=
\frac{1}{N_{\mathrm{high}}}
\sum_{i:y_i>u_b}|y_i-\hat y_i|,
$$

$$
\operatorname{RMSE}_{\mathrm{high}}=
\sqrt{
\frac{1}{N_{\mathrm{high}}}
\sum_{i:y_i>u_b}(y_i-\hat y_i)^2
}.
$$

Also track:

- event peak MAE;
- event peak RMSE;
- signed peak bias;
- median absolute relative peak error;
- annual peak-flow bias;
- high-flow volume bias or FHV;
- peak timing error.

### 16.3 Event-detection metrics

After matching observed and predicted events using a predeclared temporal tolerance, define TP, FP and FN at event level.

#### Probability of detection / recall

$$
\operatorname{POD}=\operatorname{Recall}=\frac{TP}{TP+FN}.
$$

#### Precision

$$
\operatorname{Precision}=\frac{TP}{TP+FP}.
$$

#### False-alarm ratio

$$
\operatorname{FAR}=\frac{FP}{TP+FP}.
$$

#### Critical success index

$$
\operatorname{CSI}=\frac{TP}{TP+FP+FN}.
$$

#### F-score

$$
F_\beta=(1+\beta^2)
\frac{\operatorname{Precision}\cdot\operatorname{Recall}}
{\beta^2\operatorname{Precision}+\operatorname{Recall}}.
$$

Use $\beta>1$ only when recall is explicitly more important than precision.

Track one predeclared operational threshold each epoch. Perform multi-threshold analysis less frequently or after training to avoid making noisy threshold choices.

### 16.4 Probabilistic metrics

#### Brier score for threshold exceedance

$$
\operatorname{BS}=\frac{1}{N}\sum_i(p_i-z_i)^2.
$$

#### Pinball loss

Track separately for every quantile $\tau$.

#### Empirical quantile coverage

$$
\widehat C_\tau=\frac{1}{N}\sum_i\mathbb{1}(y_i\leq q_{\tau,i}).
$$

Compare $\widehat C_\tau$ with $\tau$ overall and during high-flow periods.

#### Prediction-interval coverage probability

For interval $[q_{\ell,i},q_{u,i}]$:

$$
\operatorname{PICP}=
\frac{1}{N}\sum_i
\mathbb{1}(q_{\ell,i}\leq y_i\leq q_{u,i}).
$$

#### Mean prediction-interval width

$$
\operatorname{MPIW}=
\frac{1}{N}\sum_i(q_{u,i}-q_{\ell,i}).
$$

Coverage and width must be interpreted together. Narrow intervals are not desirable if they are under-covering.

#### CRPS

Use exact CRPS where available or approximate it from multiple quantiles [8].

#### Quantile crossing rate

$$
\operatorname{QCR}=
\frac{
\#\{(i,\tau_j,\tau_k):\tau_j<\tau_k,\,q_{\tau_j,i}>q_{\tau_k,i}\}
}{
\#\{(i,\tau_j,\tau_k):\tau_j<\tau_k\}
}.
$$

### 16.5 Training dashboard recommendation

For routine epoch-level monitoring, use a compact set:

1. validation total loss and components;
2. whole-flow MAE;
3. high-flow MAE;
4. event peak MAE;
5. signed peak bias;
6. event POD, precision and FAR at one threshold;
7. KGE plus $\alpha$ and $\beta$;
8. $C_{\mathrm{peak}}$ and maximum weight;
9. pinball loss, coverage and CRPS if probabilistic;
10. gradient norm and clipping fraction.

---

# Part IV: Inference and final model evaluation

## 17. Evaluation protocol

1. Use untouched temporal and/or spatial hold-outs representative of inference.
2. Preserve natural event prevalence.
3. Identify observed events independently of model predictions.
4. Match predicted and observed events using a documented temporal tolerance.
5. Report metrics per forecast lead time.
6. Report basin-level distributions, not only pooled values.
7. Use event- or hydrological-year block bootstrap intervals rather than resampling individual timesteps.
8. Compare against operationally relevant baselines, such as persistence, climatology, an existing hydrological model or the current production system.
9. Fix model, probability-calibration transform and decision threshold before final test evaluation.

## 18. Whole-hydrograph performance

Report:

- MAE and RMSE;
- NSE;
- KGE and its $r$, $\alpha$ and $\beta$ components;
- PBIAS;
- water-volume bias;
- metrics across seasons, lead times and basins;
- benchmark-relative skill where appropriate.

No single aggregate metric should be treated as sufficient. KGE and NSE have non-equivalent interpretations [3], and high-flow-focused calibration may improve peaks while degrading other hydrograph properties [1].

## 19. Event peak magnitude

For matched event $e$:

### Signed peak error

$$
E_{\mathrm{peak},e}=
\hat Q_{\max,e}-Q_{\max,e}.
$$

### Absolute peak error

$$
AE_{\mathrm{peak},e}=
|\hat Q_{\max,e}-Q_{\max,e}|.
$$

### Relative peak error

$$
RPE_e=
\frac{\hat Q_{\max,e}-Q_{\max,e}}
{Q_{\max,e}}.
$$

### Absolute relative peak error

$$
ARPE_e=|RPE_e|.
$$

Report:

- median signed peak error;
- median absolute peak error;
- RMSE of event peak magnitude;
- median signed and absolute relative peak error;
- error quantiles, not only a mean;
- results by peak-severity or return-period bin;
- annual peak-flow bias;
- FHV or another explicitly defined upper-flow-volume metric.

Annual peak-flow bias was developed specifically to identify high-flow differences, and peak-specific calibration may trade off against other metrics [1,10].

## 20. Event timing and hydrograph shape

### Peak timing error

$$
E_{T,e}=\hat T_{\mathrm{peak},e}-T_{\mathrm{peak},e}.
$$

Report:

- median signed timing error;
- median absolute timing error;
- 90th or 95th percentile absolute timing error;
- fraction of peaks within operational tolerances such as one model timestep;
- results by forecast lead time.

### Event volume bias

$$
VB_e=
\frac{
\sum_{t\in e}\hat y_t-
\sum_{t\in e}y_t
}{
\sum_{t\in e}y_t
}.
$$

Also consider:

- rising-limb error;
- recession error;
- duration-above-threshold error;
- maximum rate-of-rise error;
- onset-timing error.

Define all event boundaries and matching tolerances before final evaluation.

## 21. Event occurrence and warning performance

At each operational threshold, report:

- POD/recall;
- precision;
- FAR;
- CSI;
- F-score if an agreed precision-recall preference exists;
- number of missed events;
- false alerts per basin-year;
- lead time at first correct alert;
- event-level confusion matrix.

Across probability thresholds, report:

- precision-recall curve;
- average precision or PR-AUC;
- ROC-AUC only as a supplementary ranking metric;
- threshold-performance trade-off curves.

When events are rare, precision-recall measures are generally more informative than accuracy or ROC plots alone [6].

## 22. Probabilistic inference evaluation

For quantile, ensemble or parametric forecasts, report:

- CRPS;
- pinball loss at every quantile;
- empirical quantile coverage;
- central interval coverage and width;
- Brier score for every operational threshold;
- Brier skill score against a reference forecast;
- reliability diagrams;
- sharpness conditional on acceptable calibration;
- probability integral transform or rank histogram diagnostics;
- calibration by event severity, lead time and basin;
- upper-tail exceedance reliability;
- quantile crossing rate.

Proper scoring rules are designed to reward honest probabilistic forecasts in expectation [9]. CRPS evaluates the entire predictive CDF and generalises MAE to probabilistic forecasts [7].

## 23. Required stratification

Report important metrics by:

- basin;
- forecast lead time;
- high-flow threshold or warning level;
- peak-severity or return-period bin;
- season or hydrological regime;
- event type, if a defensible classification exists;
- gauged versus ungauged or spatial-transfer setting;
- hydrological year or held-out period;
- event duration;
- data-quality category, if quality flags are available.

For regional models, provide median, interquartile range and lower-tail basin performance. A high median can hide complete failures in a subset of basins.

## 24. Confidence intervals and uncertainty in metrics

Because discharge timesteps are autocorrelated, do not form uncertainty intervals by independently resampling rows. Suitable resampling units include:

- independent events;
- hydrological years;
- storms or meteorological episodes;
- basins, when estimating across-basin generalisation.

Report bootstrap intervals for core metrics and the paired difference between candidate and baseline models. Paired resampling is important because models are evaluated on the same events.

---

# Part V: Recommended first implementation

## 25. Initial objective

Use

$$
L =
\frac{
\sum_{b,t}
\left[
1+\lambda
\left(
\max\left(0,\frac{y_{b,t}-u_b}{s_b}\right)
\right)^\gamma
\right]
\ell_\delta(y_{b,t}-\hat y_{b,t})
}{
\sum_{b,t}
\left[
1+\lambda
\left(
\max\left(0,\frac{y_{b,t}-u_b}{s_b}\right)
\right)^\gamma
\right]
},
$$

where $\ell_\delta$ is Huber loss.

Initial settings:

- $u_b$: operational warning threshold, or a training-period high-flow percentile;
- $s_b$: robust basin-specific discharge scale;
- $\gamma=1$;
- $\lambda\in\{0,0.5,1,2,4,8\}$;
- $\delta$ selected from baseline residual scale;
- optional $w_{\max}$ only if weight or gradient diagnostics justify it.

## 26. Minimum comparison set

1. Unweighted Huber.
2. Peak-weighted Huber.
3. Asymmetric peak-weighted Huber.
4. Multi-quantile loss at $\tau\in\{0.1,0.5,0.9,0.95\}$.
5. Peak-weighted Huber plus an exceedance head.

## 27. Selection rule

Choose the model with the best event peak-magnitude performance subject to predeclared constraints on:

- false-alarm ratio or false alerts per basin-year;
- event recall;
- whole-hydrograph KGE and bias;
- peak-timing error;
- predictive coverage, if probabilistic.

Do not select on weighted validation loss alone.

## 28. Red flags

Stop or revise an experiment if:

- a handful of events account for most of the gradient norm;
- the model improves peak RMSE by systematically overpredicting moderate flows;
- event recall rises only because false alarms become operationally unacceptable;
- upper quantiles are consistently under-covered;
- quantile outputs cross frequently;
- regional performance is driven by a few large basins;
- overlapping windows from the same event appear in different splits;
- the model is evaluated on a balanced test distribution;
- thresholds or scales were computed using validation or test targets;
- metric gains disappear under event-block or year-block uncertainty intervals.

---

# Bibliography

1. Mizukami, N., Rakovec, O., Newman, A. J., Clark, M. P., Wood, A. W., Gupta, H. V., & Kumar, R. (2019). On the choice of calibration metrics for “high-flow” estimation using hydrologic models. *Hydrology and Earth System Sciences, 23*, 2601–2614. https://doi.org/10.5194/hess-23-2601-2019

2. Gupta, H. V., Kling, H., Yilmaz, K. K., & Martinez, G. F. (2009). Decomposition of the mean squared error and NSE performance criteria: Implications for improving hydrological modelling. *Journal of Hydrology, 377*(1–2), 80–91. https://doi.org/10.1016/j.jhydrol.2009.08.003

3. Knoben, W. J. M., Freer, J. E., & Woods, R. A. (2019). Technical note: Inherent benchmark or not? Comparing Nash–Sutcliffe and Kling–Gupta efficiency scores. *Hydrology and Earth System Sciences, 23*, 4323–4331. https://doi.org/10.5194/hess-23-4323-2019

4. Lin, T.-Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017). Focal loss for dense object detection. *Proceedings of the IEEE International Conference on Computer Vision*, 2980–2988. https://doi.org/10.48550/arXiv.1708.02002

5. Pasche, O. C., & Engelke, S. (2024). Neural networks for extreme quantile regression with an application to forecasting of flood risk. *The Annals of Applied Statistics, 18*(4), 2818–2839. https://doi.org/10.1214/24-AOAS1907

6. Saito, T., & Rehmsmeier, M. (2015). The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets. *PLOS ONE, 10*(3), e0118432. https://doi.org/10.1371/journal.pone.0118432

7. Gneiting, T., & Raftery, A. E. (2007). Strictly proper scoring rules, prediction, and estimation. *Journal of the American Statistical Association, 102*(477), 359–378. https://doi.org/10.1198/016214506000001437

8. Berrisch, J., & Ziel, F. (2023). CRPS learning. *Journal of Econometrics, 237*(2), 105221. https://doi.org/10.1016/j.jeconom.2023.105221

9. Waghmare, K., & Ziegel, J. (2025). Proper scoring rules for estimation and forecast evaluation. arXiv. https://doi.org/10.48550/arXiv.2504.01781

10. Zambrano-Bigiarini, M. (2024). `hydroGOF`: Goodness-of-fit functions for comparison of simulated and observed hydrological time series. R package documentation for annual peak flow bias. https://search.r-project.org/CRAN/refmans/hydroGOF/html/APFB.html

11. NeuralHydrology contributors. (2026). NeuralHydrology evaluation metrics documentation. https://neuralhydrology.readthedocs.io/en/latest/api/neuralhydrology.evaluation.metrics.html

12. Branco, P., Torgo, L., & Ribeiro, R. P. (2017). SMOGN: A pre-processing approach for imbalanced regression. *Proceedings of Machine Learning Research, 74*, 36–50. https://proceedings.mlr.press/v74/branco17a.html

13. Tofighi, S., Gurbuz, F., Mantilla, R., & Xiao, S. (2025). Advancing machine learning-based streamflow prediction through event greedy selection, asymmetric loss function, and rainfall forecasting uncertainty. *Applied Sciences, 15*(21), 11656. https://doi.org/10.3390/app152111656

---

## Source notes

- Reference [1] directly motivates reporting peak-specific metrics alongside aggregate time-series efficiencies.
- References [2] and [3] support decomposition and interpretation of KGE.
- Reference [4] is the original focal-loss paper; its hydrological use should be limited to an exceedance-classification component unless separately validated.
- Reference [5] provides a flood-forecasting application combining neural networks and extreme-value theory.
- Reference [6] supports using precision-recall analysis for imbalanced event detection.
- References [7]–[9] support probabilistic evaluation through proper scoring rules and CRPS.
- References [10] and [11] provide implementation-oriented documentation for hydrological metrics.
- Reference [13] is a recent streamflow study involving event selection and asymmetric peak loss; it should be assessed in the context of its specific data and experimental design rather than treated as a universal prescription.
