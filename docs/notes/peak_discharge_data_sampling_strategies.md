# Data-Sampling Strategies for Imbalanced Peak-Discharge Forecasting

## Purpose and scope

This document provides a practical framework for sampling training data when rare peak-discharge events are underrepresented. It is written for regional and single-basin streamflow forecasting, but most principles apply to imbalanced regression and time-series forecasting more broadly.

It covers:

- event-window oversampling and background undersampling;
- weighted and stratified mini-batch sampling;
- hard-negative mining, active learning and curriculum sampling;
- basin-balanced and hierarchical sampling;
- synthetic regression methods such as SMOTER and SMOGN;
- hydrologically informed augmentation and stochastic storm generation;
- sampling hyperparameters and tuning strategies;
- diagnostics to track during training;
- metrics and experimental design for final inference evaluation;
- leakage prevention and reproducibility;
- pros and cons for each strategy;
- a hydrology-prioritised bibliography.

The primary recommendation is to begin with **event-aware weighted mini-batch sampling of observed data**, using independent rainfall-runoff events rather than individual peak timesteps as the sampling unit. Combine this with moderate removal of redundant background windows and explicit hard-negative sampling. Treat feature-space synthetic oversampling as a later experiment because generic interpolation can violate rainfall-runoff timing, antecedent-state consistency and hydrograph structure.

---

## 1. Notation

| Symbol | Meaning |
|---|---|
| $b$ | Basin index |
| $e$ | Independent event index |
| $t$ | Time index |
| $x_{b,e}$ | Input window for event or background case |
| $y_{b,e}$ | Target hydrograph or target window |
| $Q_{\max,b,e}$ | Observed event peak discharge |
| $u_b$ | Basin-specific high-flow threshold |
| $r_{b,e}$ | Relevance or severity score for event $e$ |
| $p_{b,e}$ | Probability of sampling event $e$ |
| $n_k$ | Number of available cases in sampling stratum $k$ |
| $m_k$ | Number of cases drawn from stratum $k$ per epoch or batch |
| $\pi_k$ | Natural prevalence of stratum $k$ |
| $\tilde\pi_k$ | Training sampling prevalence of stratum $k$ |
| $w_i$ | Optional importance correction for sampled example $i$ |

All thresholds, strata, density estimates, relevance functions and hard-negative selections must be constructed using the training portion of each split only.

---

# Part I: Core principles

## 2. Sampling unit: events, not rows

A rare hydrological event commonly produces many adjacent high-flow timesteps and many overlapping training windows. Sampling each timestep independently inflates the apparent number of rare examples and can allow one long event to dominate training.

The preferred sampling unit is an **independent event window** containing enough context to represent:

- antecedent catchment state;
- forcing history;
- rising limb;
- event peak;
- recession, when included in the target horizon;
- forecast lead time and target horizon.

Define event separation before sampling. Suitable definitions depend on temporal resolution, basin response time and operating context. The event catalogue should be generated consistently for all model variants.

### Effective event count

Let event $e$ contribute $n_e$ overlapping windows. The raw window count is

$$
N_{\mathrm{windows}}=\sum_e n_e,
$$

but this is not the number of independent rare events. Always report both:

- number of independent events;
- number of generated windows.

### Pros

- Better reflects the independent information available.
- Prevents long-duration floods from dominating by window count alone.
- Enables event-level severity, timing and catchment-based sampling.
- Aligns sampling diagnostics with operational evaluation.

### Cons

- Requires a defensible event-identification algorithm.
- Event boundaries can be ambiguous during compound or multi-peak episodes.
- Very long or overlapping events require explicit handling.

---

## 3. Data-split and leakage rules

Sampling must be performed **after** splitting data into training, validation and test sets. Resampling the full dataset before the split leaks information and gives validation or test data an artificial class distribution. The imbalanced-learn documentation explicitly identifies these two problems [9].

For time-series models:

1. Split the continuous record first.
2. Add guard bands where input or target windows could overlap a split boundary.
3. Construct windows separately inside each partition.
4. Fit thresholds, scalers, relevance functions and synthetic generators on training data only.
5. Resample only the training partition.
6. Leave validation and test data at natural prevalence.

### Grouping requirements

Keep the following together where appropriate:

- windows from the same hydrological event;
- windows from the same storm system;
- upstream-downstream observations sharing the same forcing episode;
- repeated synthetic variants derived from one original event;
- near-duplicate windows.

### Leakage audit

For every experiment record:

- event IDs in each split;
- basin IDs in each split;
- earliest and latest timestamps in each split;
- guard-band duration;
- provenance links from synthetic samples to source events;
- nearest-neighbour similarity across splits;
- whether thresholds and scalers were fitted only on training data.

---

# Part II: Observed-data sampling strategies

## 4. Uniform event sampling

Instead of drawing windows uniformly, draw independent events uniformly and then draw one or more windows from each selected event:

$$
p_e=\frac{1}{|E|}.
$$

If multiple windows are needed from the same event, sample their positions conditionally after selecting the event.

### Pros

- Simple baseline that prevents long events from receiving weight proportional to duration.
- Uses only observed data.
- Preserves event diversity better than duplicating all peak timesteps.

### Cons

- Treats moderate and extreme events equally.
- Does not explicitly preserve the natural background distribution.
- Short or poorly measured events can receive the same probability as well-observed events.

### Tuning

- number of event windows per batch;
- number of windows per selected event;
- position distribution within an event;
- proportion of event versus background windows.

Compare uniform-event sampling against natural window sampling before introducing severity weighting.

---

## 5. Random oversampling of observed rare events

Rare events or event windows are sampled with replacement. If $E_R$ is the rare-event set:

$$
p_e=\frac{1}{|E_R|},\qquad e\in E_R.
$$

A batch can mix rare and ordinary cases:

$$
B=B_R\cup B_M,
$$

where $B_R$ contains oversampled rare events and $B_M$ contains background or moderate-flow cases.

### Pros

- Very easy to implement.
- Does not invent synthetic physical states.
- Strong baseline for testing whether rare-event exposure is the bottleneck.
- Compatible with sequence models and complex inputs.

### Cons

- Repeated rare events can be memorised.
- Oversampling overlapping windows from one event creates low-diversity batches.
- The altered training prevalence can affect probabilistic calibration.
- A 50:50 balance can produce too many false alarms.

### Tuning

Use the rare-event batch fraction $\rho_R$ as the main hyperparameter:

$$
\rho_R=\frac{|B_R|}{|B|}.
$$

A practical comparison grid is

$$
\rho_R\in\{\pi_R,0.10,0.20,0.35,0.50\},
$$

where $\pi_R$ is the natural rare-event prevalence. These values are experimental candidates, not hydrological standards.

Also tune:

- maximum repeats per event per epoch;
- minimum number of distinct events per batch;
- maximum fraction of a batch from one basin;
- separation between windows sampled from the same event.

Monitor unique-event coverage and repeat counts, not only sample count.

---

## 6. Severity-weighted event sampling

Assign each event a relevance score $r_e\in[0,1]$ based on basin-relative peak severity, warning level, return-period category or expected impact. Draw events according to

$$
p_e=
\frac{(\epsilon+r_e)^\eta}
{\sum_j(\epsilon+r_j)^\eta}.
$$

Here:

- $\epsilon>0$ ensures non-extreme cases can still be sampled;
- $\eta=0$ gives uniform event sampling;
- larger $\eta$ concentrates on severe events.

### Constructing relevance

#### Threshold-excess relevance

$$
r_e=\min\left[1,
\left(
\frac{\max(0,Q_{\max,e}-u_b)}{s_b}
\right)^\gamma
\right].
$$

#### Percentile relevance

For basin-specific empirical CDF $F_b$:

$$
r_e=F_b(Q_{\max,e})^\gamma.
$$

#### Warning-level relevance

Assign ordered scores to operational levels, for example $r_e\in\{0,0.25,0.5,0.75,1\}$. The specific values should be tuned or based on operational consequences.

#### Impact relevance

$$
r_e=\frac{C(Q_{\max,e})}{C_{\max}},
$$

where $C$ is an agreed impact or cost function.

### Pros

- Smoothly prioritises more consequential events.
- Avoids a strict rare/non-rare boundary.
- Allows basin-relative or impact-based definitions.
- Easy to combine with event-level sampling.

### Cons

- Can over-concentrate on a handful of historical extremes.
- Severity and rarity are not identical.
- Percentile relevance may equate events with different impacts.
- Requires careful correction or recalibration for probabilistic outputs.

### Tuning

Try

$$
\eta\in\{0,0.5,1,2\},
$$

and choose $\epsilon$ to maintain a minimum probability for ordinary cases. Evaluate:

- unique events per epoch;
- entropy of the event sampling distribution;
- maximum event probability;
- peak performance by severity bin;
- moderate-flow bias and false alarms.

Sampling entropy is

$$
H(p)=-\sum_e p_e\log p_e.
$$

A steep decline in $H(p)$ indicates concentration on a small event subset.

---

## 7. Stratified severity sampling

Partition event windows into strata, such as:

1. stable background;
2. hydrologically active non-event;
3. near-threshold;
4. frequent flood peak;
5. rare or extreme peak.

Draw a fixed or adaptive number $m_k$ from stratum $k$.

### Sampling probabilities

For an example $i$ in stratum $k$:

$$
p_i=\frac{m_k}{|B|}\frac{1}{n_k}.
$$

### Tempered stratum allocation

If natural prevalence is $\pi_k$, define

$$
\tilde\pi_k=\frac{\pi_k^\tau}{\sum_j\pi_j^\tau},
$$

where:

- $\tau=1$ preserves natural prevalence;
- $\tau=0$ gives equal stratum prevalence;
- $0<\tau<1$ partially balances strata.

This is often safer than forcing equal class sizes.

### Pros

- Transparent control of batch composition.
- Near-threshold cases can be emphasised separately from extreme peaks.
- Supports operational warning categories.
- Enables severity-specific diagnostics.

### Cons

- Stratum boundaries are hyperparameters.
- Sparse extreme strata may still repeat the same events.
- Abrupt categories can separate hydrologically similar observations.
- Equal-stratum sampling can heavily distort prevalence.

### Tuning

Try

$$
\tau\in\{1,0.75,0.5,0.25,0\}.
$$

Tune boundaries using training data only. Compare basin-relative percentiles, operational thresholds and return-period bands. Prefer the simplest stratification that resolves an observed performance deficiency.

---

## 8. Near-threshold sampling

Peak forecasting systems often need to discriminate events around a warning threshold. Define a near-threshold band

$$
\mathcal N_b=
\{e:c_1u_b\leq Q_{\max,e}<c_2u_b\},
$$

or use percentile bands around $u_b$.

Give these events a dedicated batch fraction $\rho_N$.

### Pros

- Provides information near the operational decision boundary.
- Can improve sensitivity-specificity trade-offs.
- Prevents all sampling effort going to only the largest floods.

### Cons

- Does not by itself improve far-tail magnitude estimation.
- Threshold uncertainty can make band membership unstable.
- Too much near-threshold sampling can reduce ordinary-flow or extreme-event coverage.

### Tuning

Tune:

- lower and upper band widths;
- near-threshold fraction $\rho_N$;
- separate bands for rising-limb precursors and realised peak magnitude.

Track threshold-crossing precision, recall, calibration and false alerts per basin-year.

---

## 9. Random undersampling of background periods

Discard or sample fewer ordinary-flow windows. If the natural background count is $N_M$, retain each with probability $q_M$:

$$
N'_M\sim\operatorname{Binomial}(N_M,q_M).
$$

### Pros

- Reduces training cost and redundant stable-flow observations.
- Increases the fraction of informative hydrologically active windows.
- Easy to combine with event oversampling.

### Cons

- Can remove antecedent-state, seasonal or transition information.
- Random removal may discard hard negatives.
- Changes the training prevalence and potentially calibration.
- Aggressive undersampling can make ordinary dynamics unstable.

### Tuning

Compare retained background fractions such as

$$
q_M\in\{1,0.5,0.25,0.10\}.
$$

Before removal, partition background by:

- season;
- basin;
- flow regime;
- antecedent wetness;
- data quality;
- forcing intensity.

Sample within these groups so that reducing redundancy does not erase hydrological coverage.

---

## 10. Diversity-aware background undersampling

Instead of random removal, retain background cases that cover the predictor-state space. Possible selection methods include:

- clustering and sampling from each cluster;
- farthest-point or k-centre selection;
- reservoir sampling within season-basin-regime strata;
- prototype selection;
- coverage of rainfall, soil moisture, snow state and prior discharge combinations.

### Pros

- Preserves a broader range of non-event states than random undersampling.
- Can reduce dataset size substantially while retaining diversity.
- Helps preserve rare but important non-flood conditions.

### Cons

- Requires a meaningful distance representation.
- High-dimensional distances may be unreliable.
- A representation learned from the full dataset can leak information.
- Clustering costs can be high.

### Tuning

- retained sample budget;
- representation used for similarity;
- number of clusters or prototypes;
- per-basin and per-season minimum quotas;
- maximum cluster imbalance.

Fit representations and clusters on training data only.

---

## 11. Hard-negative mining

Hard negatives are non-event windows that the current model gives high peak predictions or high exceedance probability.

### Procedure

1. Train a baseline model on the current training sample.
2. Generate out-of-fold or strictly training-only predictions.
3. Select non-event cases with high predicted event scores or large positive discharge errors.
4. Increase their sampling probability in the next training round.
5. Retain a random-background component to preserve coverage.

Define a hard-negative score, for example

$$
h_i=p_i\mathbb{1}(z_i=0),
$$

or

$$
h_i=\max(0,\hat Q_{\max,i}-Q_{\max,i})\mathbb{1}(z_i=0).
$$

Then sample

$$
p_i\propto (\epsilon+h_i)^\eta.
$$

### Hydrological interpretation

Potential hard negatives include intense rainfall without a flood response, wet-looking meteorological patterns with dry antecedent states, snow-storage cases, regulated-flow episodes or spatial rainfall mismatches. These categories should be diagnosed from the data rather than assumed.

### Pros

- Directly targets false alarms.
- Uses model errors to identify informative ordinary cases.
- Often more useful than randomly retaining large numbers of easy negatives.

### Cons

- Can focus on mislabeled events, sensor errors or model artefacts.
- Iterative training is more complex.
- In-sample predictions can select memorisation artefacts.
- Excessive mining can narrow the background distribution.

### Tuning

Tune:

- number or fraction of mined negatives;
- score threshold;
- random-to-hard-negative ratio;
- refresh frequency;
- cap per basin, season and event period;
- number of mining rounds.

Use out-of-fold training predictions when feasible. Audit the highest-scoring hard negatives manually or with data-quality flags.

---

## 12. Error- and uncertainty-based sampling

Prioritise cases on which the current model has high error or high uncertainty:

$$
p_i\propto(\epsilon+E_i)^\eta,
$$

where $E_i$ may be absolute error, peak error, predictive variance, ensemble disagreement or interval width.

### Pros

- Targets informative examples rather than rarity alone.
- Can expose difficult transition regimes.
- Compatible with active learning and iterative model refinement.

### Cons

- High uncertainty may reflect noise or out-of-distribution cases.
- Prediction error requires careful out-of-fold estimation.
- Can induce feedback loops in which the model repeatedly sees the same difficult cases.

### Tuning

- error or uncertainty definition;
- exponent $\eta$;
- sampling floor $\epsilon$;
- maximum probability cap;
- mixture proportion with uniform sampling.

Always include a non-zero uniform component:

$$
p_i=(1-\rho)p_i^{\mathrm{uniform}}+\rho p_i^{\mathrm{priority}}.
$$

---

## 13. Curriculum and annealed sampling

Change the sampling distribution during training. A generic schedule is

$$
\tilde\pi_k(s)=
(1-a_s)\tilde\pi_k^{\mathrm{balanced}}
+a_s\pi_k^{\mathrm{natural}},
$$

where $s$ is training stage and $a_s$ increases towards 1.

Alternative curricula can begin near the natural distribution and progressively emphasise difficult rare events.

### Pros

- Separates representation learning from later peak specialisation.
- Can finish training closer to the natural distribution.
- May reduce instability from strong balancing at initialisation.

### Cons

- Adds schedule hyperparameters.
- Benefits are problem dependent.
- Makes comparisons and reproducibility more difficult.
- Late changes can undo earlier rare-event gains.

### Tuning

- starting distribution;
- target distribution;
- transition start and end epochs;
- linear, cosine or step schedule;
- whether peak emphasis increases or decreases.

Treat curriculum sampling as a later ablation, not the first baseline.

---

# Part III: Regional and multi-basin sampling

## 14. Basin-balanced hierarchical sampling

Regional models can be imbalanced both within and between basins. A hierarchical sampler first chooses basin $b$, then severity stratum $k$, then event $e$:

$$
p(b,k,e)=p(b)\,p(k\mid b)\,p(e\mid b,k).
$$

### Basin probability

Uniform basin sampling:

$$
p(b)=\frac{1}{B}.
$$

Record-length sampling:

$$
p(b)=\frac{N_b}{\sum_jN_j}.
$$

Tempered basin sampling:

$$
p(b)=\frac{N_b^\tau}{\sum_jN_j^\tau},\qquad0\leq\tau\leq1.
$$

### Pros

- Prevents long-record or event-rich basins from automatically dominating.
- Makes basin and severity balance independently controllable.
- Suitable for regional deep-learning models.

### Cons

- Uniform basin sampling can overemphasise very short or low-quality records.
- Record-length sampling can reproduce the original imbalance.
- Sparse strata in some basins require fallback rules.

### Tuning

Try $\tau\in\{0,0.25,0.5,0.75,1\}$. Establish minimum data-quality and event-count requirements. Report per-basin update frequency and gradient contribution.

---

## 15. Hydrological similarity-aware sampling

To support poorly represented basins, sample additional examples from hydrologically similar basins according to static attributes or learned representations.

For similarity $s(b,j)\geq0$:

$$
p(j\mid b)=\frac{s(b,j)^\kappa}{\sum_l s(b,l)^\kappa}.
$$

### Pros

- Can share information across comparable catchments.
- Supports regionalisation and sparse-basin learning.
- Allows targeted transfer rather than global pooling.

### Cons

- Similar static attributes do not guarantee similar event response.
- Learned similarity can leak if fitted on held-out targets.
- High $\kappa$ may create narrow neighbour sets.

### Tuning

- attribute set;
- standardisation of attributes;
- distance metric;
- number of neighbours;
- similarity exponent $\kappa$;
- local versus global mixture proportion.

Evaluate on spatially held-out basins if prediction in ungauged basins is an objective.

---

# Part IV: Synthetic and augmented data

## 16. Simple perturbation of observed events

Augment observed sequences using small, physically defensible perturbations, such as changes to input forcing or timing that preserve intended labels.

Generic additive noise is

$$
x'=x+\epsilon,\qquad \epsilon\sim\mathcal N(0,\Sigma).
$$

However, arbitrary independent noise can be hydrologically invalid. Perturbations should respect units, temporal correlation, spatial structure, non-negativity and known measurement uncertainty.

### Pros

- Increases local variation around observed rare events.
- Easy to combine with event oversampling.
- Can encode known observation uncertainty.

### Cons

- May break rainfall-runoff consistency.
- Does not create genuinely new event mechanisms.
- Label-preservation assumptions can be false.
- Large perturbations can produce implausible extremes.

### Tuning

- perturbation magnitude;
- temporal and spatial covariance;
- which features may be perturbed;
- augmentation probability;
- maximum number of variants per source event.

Validate each augmentation family with physical checks and an ablation against simple duplication.

---

## 17. Time warping, cropping and sequence transformations

Possible sequence augmentations include:

- random cropping while retaining required context;
- limited temporal shifts;
- moderate time dilation or compression;
- masking selected observations;
- block bootstrap of residual or forcing components.

### Pros

- Can improve robustness to alignment and missingness.
- Preserves more temporal structure than row-wise interpolation.
- Useful when timestamps or forecast initiation vary.

### Cons

- Time warping can change travel time, peak timing and event duration.
- Cropping can remove antecedent conditions.
- Masking may simulate a different data-availability problem.
- Labels often need to be transformed jointly.

### Tuning

- maximum shift or warp;
- crop length and allowed positions;
- variables transformed jointly;
- probability of each transform;
- physical validity constraints.

Do not use transformations merely because they are common in image or generic time-series modelling.

---

## 18. Mixup-style interpolation

For two examples $(x_i,y_i)$ and $(x_j,y_j)$:

$$
\tilde x=\lambda x_i+(1-\lambda)x_j,
$$

$$
\tilde y=\lambda y_i+(1-\lambda)y_j,
$$

with

$$
\lambda\sim\operatorname{Beta}(\alpha,\alpha).
$$

### Pros

- Simple regularisation.
- Generates intermediate examples without exact duplication.
- Can smooth decision boundaries.

### Cons

- Linear combinations of storms, catchment states and hydrographs may not represent any physically possible system state.
- Mixing different basins can be especially problematic.
- Peak timing can be blurred or made multi-modal.
- Targets can violate mass balance with mixed inputs.

### Tuning

- $\alpha$;
- same-basin versus cross-basin mixing;
- similarity restrictions;
- variable groups that may be mixed;
- event alignment before mixing.

For discharge forecasting, treat mixup as exploratory and compare it against event duplication under strict physical validation.

---

## 19. SMOTER and SMOGN for imbalanced regression

SMOGN combines regression-oriented synthetic oversampling with Gaussian-noise generation and undersampling of common target regions. The original study frames imbalanced regression as the combination of rare user-relevant target values and scarce representation, and reports that its effects differ by learner [2,3].

A general interpolation form is

$$
\tilde x=x_i+\lambda(x_j-x_i),
$$

$$
\tilde y=y_i+\lambda(y_j-y_i),
$$

where $j$ is a neighbour of rare case $i$ and $\lambda\in[0,1]$. SMOGN may instead add Gaussian noise where neighbour distances are unsuitable for interpolation [2,4].

### Pros

- Designed for continuous-target imbalance.
- Combines oversampling and undersampling.
- Supports relevance-defined rare regions.
- Available in open-source implementations [4].

### Cons for hydrological sequences

- Flattened-window interpolation can violate temporal sequencing.
- Interpolated rainfall, antecedent state and discharge may be mutually inconsistent.
- Nearest neighbours may belong to different event mechanisms or basins.
- Gaussian perturbation can produce invalid values or spatial structures.
- Synthetic samples may be close replicas rather than new independent events.

Newer imbalanced-regression work also notes that simple interpolation and Gaussian noise may fail to represent complex nonlinear feature-target relationships, motivating a GAN-based filtering stage [5]. This is general imbalanced-regression evidence rather than hydrology-specific validation.

### Tuning

- relevance threshold;
- rare-to-common sampling ratio;
- number of neighbours $k$;
- distance metric;
- interpolation versus noise threshold;
- noise scale;
- same-basin or similarity restrictions;
- maximum synthetic samples per source event.

### Hydrological safeguards

- apply to event-level representations rather than individual rows;
- restrict neighbours to the same basin or a defensible hydrological similarity group;
- align hydrograph phases before interpolation;
- preserve non-negativity and variable bounds;
- check water balance and temporal consistency;
- retain source-event IDs;
- compare against simple observed-event oversampling;
- never synthesise validation or test examples.

---

## 20. Generative sequence models

Conditional GANs, variational autoencoders or diffusion models can generate rare-event windows conditioned on basin attributes, antecedent state, season or peak severity.

### Pros

- Can model nonlinear joint distributions.
- Potentially generates greater diversity than interpolation.
- Conditioning can target specific event regimes.

### Cons

- Requires substantial rare-event data to train the generator.
- Plausible-looking sequences may be physically inconsistent.
- Evaluation of generative fidelity and coverage is difficult.
- Mode collapse can reproduce a small set of known events.
- Generated labels and inputs may not be jointly valid.

### Tuning and validation

- conditioning variables;
- latent dimension;
- generator-to-discriminator update ratio;
- number of synthetic samples;
- acceptance thresholds;
- similarity-to-training-event limits;
- physical constraint penalties.

Evaluate:

- marginal and joint distributions;
- rainfall-discharge lag;
- event volume and runoff coefficient;
- rising and recession behaviour;
- spatial rainfall coherence;
- nearest-neighbour distance to real events;
- predictive utility in an observed-only test set.

Synthetic quality should be judged primarily by physically valid downstream performance, not visual realism alone.

---

## 21. Hydrological-model-generated synthetic events

A more defensible hydrology-specific strategy is to generate meteorological events and pass them through a calibrated hydrological model to produce consistent runoff targets.

A 2025 Iowa River Basin study used stochastic storm transposition to create realistic rainfall events, supplied them to a hydrological model, and used active learning to select informative events for deterministic and probabilistic LSTM training [1]. The Bureau of Reclamation describes stochastic storm transposition as a physically based method for creating extreme-rainfall scenarios that can be combined with hydrological models for flood-hazard and uncertainty analysis [6,7].

### Generic pipeline

1. Sample or generate a meteorological forcing event.
2. Sample a seasonally consistent antecedent catchment state.
3. Run a hydrological model.
4. Apply physical and quality-control filters.
5. derive event windows and labels;
6. sample synthetic and observed events jointly during ML training.

### Pros

- Preserves a model-based relationship between forcing, state and discharge.
- Can explore events outside the short observed record.
- Supports controlled variation in storm magnitude, placement and antecedent state.
- Allows active selection of informative simulations.

### Cons

- ML models can inherit structural and calibration biases from the hydrological model.
- Simulated extremes may be overconfident or physically incomplete.
- Computational cost can be substantial.
- Real and synthetic distributions may differ.
- Synthetic data are not independent observations of nature.

### Tuning

- observed-to-synthetic batch ratio;
- storm-generation parameters;
- antecedent-state sampling distribution;
- hydrological-model ensemble size;
- acceptance criteria;
- maximum synthetic severity;
- per-basin simulation budget;
- active-learning acquisition function.

### Weighting synthetic examples

Use a synthetic confidence weight $c_i\in[0,1]$:

$$
L=
\frac{
\sum_{i\in\mathrm{obs}}\ell_i+
\lambda_S\sum_{i\in\mathrm{syn}}c_i\ell_i
}{
N_{\mathrm{obs}}+\lambda_S\sum_{i\in\mathrm{syn}}c_i
}.
$$

Tune $\lambda_S$ rather than treating simulated and observed examples as automatically equivalent.

---

## 22. Active learning for simulation selection

When synthetic generation requires expensive hydrological simulations, select scenarios likely to add information.

Possible acquisition scores include:

- predictive uncertainty;
- ensemble disagreement;
- expected peak severity;
- distance from existing training events;
- expected error reduction;
- coverage gaps in forcing-state space.

For score $a(x)$:

$$
x^*=\arg\max_{x\in\mathcal C}a(x),
$$

where $\mathcal C$ is the candidate scenario set.

The hydrological study by Tofighi et al. explicitly used an active-learning approach to identify informative rainfall events and reduce data-generation effort [1].

### Pros

- Focuses simulation budget on informative events.
- Can improve coverage of poorly represented hydrological conditions.
- Avoids indiscriminate generation of many redundant events.

### Cons

- Acquisition scores can favour noisy or unrealistic cases.
- Requires iterative generation and retraining.
- Coverage can narrow without an exploration component.

### Tuning

- acquisition function;
- exploration-exploitation mixture;
- batch size per acquisition round;
- number of rounds;
- diversity penalty;
- physical acceptance criteria.

Use a mixture such as

$$
a'(x)=\rho a_{\mathrm{informative}}(x)+(1-\rho)a_{\mathrm{diversity}}(x).
$$

---

# Part V: Correcting sampling-distribution changes

## 23. Importance weighting

If training samples are drawn from proposal distribution $q(i)$ instead of natural distribution $p(i)$, an unbiased empirical-risk estimate can use

$$
w_i=\frac{p(i)}{q(i)},
$$

and

$$
L_{\mathrm{IW}}=
\frac{\sum_iw_i\ell_i}{\sum_iw_i}.
$$

### Pros

- Corrects the target risk under ideal assumptions.
- Separates the computational sampling distribution from the desired objective distribution.

### Cons

- Large weights produce high-variance gradients.
- Natural probabilities are difficult to estimate for event windows.
- Full correction can undo the rare-event emphasis.
- Dependence among overlapping windows complicates interpretation.

### Stabilised correction

Use tempered weights

$$
w_i=\left(\frac{p(i)}{q(i)}\right)^\beta,
\qquad0\leq\beta\leq1,
$$

plus clipping:

$$
w'_i=\min(w_i,w_{\max}).
$$

Tune $\beta$ and $w_{\max}$ using validation performance and gradient diagnostics.

---

## 24. Probability calibration after resampling

Oversampling changes the prevalence seen during training. For a classification head, predicted exceedance probabilities may therefore require calibration on untouched validation data with natural prevalence.

Evaluate:

- Brier score;
- reliability diagrams;
- expected calibration error with caution about binning;
- event-level reliability;
- reliability by lead time and basin;
- precision and false-alert budgets at chosen thresholds.

Threshold selection and probability calibration must use validation data, not the final test set.

---

# Part VI: Hyperparameter tuning and experiment design

## 25. Main sampling hyperparameters

| Strategy | Main hyperparameters |
|---|---|
| Event oversampling | rare batch fraction, repeats per event, windows per event |
| Severity-weighted sampling | relevance definition, $\epsilon$, $\eta$, probability cap |
| Stratified sampling | stratum boundaries, temperature $\tau$, per-stratum quotas |
| Background undersampling | retained fraction, stratification variables, minimum quotas |
| Hard-negative mining | selection score, mined fraction, refresh rate, random mixture |
| Basin balancing | basin temperature $\tau$, minimum record quality, basin cap |
| SMOGN/SMOTER | relevance threshold, $k$, distance, noise, synthetic ratio |
| Sequence augmentation | transform family, probability, magnitude, constraints |
| Model-generated events | synthetic ratio, scenario distribution, simulator ensemble, confidence weight |
| Active learning | acquisition function, exploration mix, batch size, rounds |
| Importance correction | $\beta$, weight cap, normalisation |

## 26. Staged tuning strategy

### Stage A: establish sampling baselines

1. Natural window sampling.
2. Uniform event sampling.
3. Moderate observed-event oversampling.
4. Moderate background undersampling.

### Stage B: add hydrological structure

1. Stratify by basin-relative severity.
2. Add near-threshold quota.
3. Add hierarchical basin sampling.
4. Compare event-normalised and window-based variants.

### Stage C: target model errors

1. Mine hard negatives using out-of-fold predictions.
2. Test error- or uncertainty-prioritised sampling.
3. Tune the uniform-priority mixture.

### Stage D: synthetic augmentation

1. Test small physically constrained perturbations.
2. Compare simple duplication with SMOGN or interpolation.
3. If a process model is available, evaluate model-generated events.
4. Add active learning only when simulation cost justifies it.

## 27. Minimum ablation matrix

| Experiment | Sampling unit | Rare-event treatment | Background treatment | Synthetic data |
|---|---|---|---|---|
| A | Window | Natural | Natural | None |
| B | Event | Uniform events | Natural quota | None |
| C | Event | Severity-weighted | Natural quota | None |
| D | Event | Stratified | Moderate undersampling | None |
| E | Event | Stratified | Hard negatives + random | None |
| F | Event | Stratified | Hard negatives + random | Simple augmentation |
| G | Event | Stratified | Hard negatives + random | Hydrological simulation |
| H | Event | Stratified | Hard negatives + random | SMOGN or generative method |

Use identical model architecture, loss, temporal/spatial splits, evaluation code, seed policy and tuning budget wherever possible. This isolates the value of sampling.

## 28. Model-selection rule

Do not select the sampler by training loss. Select it using untouched validation data and a predeclared rule, such as:

- minimum event peak error subject to a false-alert constraint;
- maximum event recall subject to minimum precision;
- Pareto frontier of peak error, false alarms and whole-hydrograph KGE;
- probabilistic score subject to adequate upper-tail calibration.

A sampler is useful only if it improves inference under the natural distribution.

---

# Part VII: Diagnostics during training

## 29. Sampling-distribution diagnostics

Track per epoch:

- total windows drawn;
- number of unique windows;
- number of independent events;
- unique-event fraction;
- repeats per event;
- maximum event repeat count;
- number and fraction by severity stratum;
- number and fraction by basin;
- number and fraction by season;
- near-threshold fraction;
- background fraction;
- hard-negative fraction;
- observed versus synthetic fraction;
- sampling entropy;
- effective sample size;
- maximum sampling probability;
- distribution of event peak magnitude;
- distribution of antecedent states;
- distribution of input rainfall totals and intensities;
- provenance of every synthetic sample.

### Unique-event fraction

$$
U_E=\frac{\#\text{ unique events drawn}}{\#\text{ event draws}}.
$$

### Effective sample size from sampling probabilities

$$
N_{\mathrm{eff}}=\frac{1}{\sum_i p_i^2}.
$$

For importance weights:

$$
N_{\mathrm{eff},w}=
\frac{(\sum_iw_i)^2}{\sum_iw_i^2}.
$$

A declining effective sample size indicates increasing concentration.

## 30. Optimisation diagnostics

Track:

- training and validation loss;
- loss by severity stratum;
- loss by observed versus synthetic data;
- gradient norm;
- gradient contribution by severity and basin;
- fraction of clipped gradients;
- variation of batch loss;
- learning rate;
- fraction of invalid or missing samples;
- maximum per-example loss;
- prediction distribution by sampled stratum.

If oversampling boosts rare-event performance only by letting a small number of events dominate the gradient, reduce oversampling strength or enforce event diversity.

## 31. Compact validation dashboard

On natural-prevalence validation data, track:

### Whole hydrograph

- MAE;
- RMSE;
- NSE;
- KGE and its correlation, variability and bias components;
- percent bias or water-volume bias.

### Peak magnitude

For event $e$:

$$
RPE_e=\frac{\hat Q_{\max,e}-Q_{\max,e}}{Q_{\max,e}}.
$$

Track:

- event peak MAE and RMSE;
- median signed RPE;
- median absolute RPE;
- annual peak-flow bias;
- high-flow volume bias;
- error by severity stratum.

### Event detection

$$
\operatorname{POD}=\frac{TP}{TP+FN},
$$

$$
\operatorname{Precision}=\frac{TP}{TP+FP},
$$

$$
\operatorname{FAR}=\frac{FP}{TP+FP},
$$

$$
\operatorname{CSI}=\frac{TP}{TP+FP+FN}.
$$

Also track false alerts per basin-year and missed-event count.

### Timing

$$
E_{T,e}=\hat T_{\mathrm{peak},e}-T_{\mathrm{peak},e}.
$$

Track median signed and absolute peak-timing error.

### Probabilistic forecasts

Track:

- Brier score for threshold exceedance;
- precision-recall curve and average precision;
- CRPS;
- pinball loss by quantile;
- empirical quantile coverage;
- interval coverage and width;
- reliability by basin, lead time and severity.

Precision-recall analysis is particularly relevant for strongly imbalanced binary event detection [10].

---

# Part VIII: Final inference evaluation

## 32. Evaluation protocol

1. Evaluate on untouched natural-prevalence temporal and/or spatial hold-outs.
2. Fix the sampler, model, loss, calibration method and warning threshold before final testing.
3. Match events using a predeclared temporal tolerance.
4. Report by basin, lead time, season and severity.
5. Use block-bootstrap intervals based on events, storms, hydrological years or basins.
6. Compare samplers using paired resampling on identical held-out events.
7. Include training cost and unique rare-event exposure.
8. Audit false positives and missed extremes qualitatively as well as quantitatively.

## 33. Core result table

For each sampler, report:

### Sampling characteristics

- unique independent events seen;
- rare-event repeat distribution;
- effective sample size;
- background reduction;
- synthetic fraction;
- training examples and compute;
- per-basin and per-severity coverage.

### Predictive outcomes

- whole-flow MAE/RMSE;
- KGE and components;
- event peak MAE/RMSE;
- signed and absolute relative peak error;
- peak timing error;
- POD, precision, FAR and CSI;
- false alerts per basin-year;
- error by peak-severity band;
- uncertainty intervals for metric differences;
- CRPS, Brier score and calibration when probabilistic.

### Generalisation diagnostics

- performance on events never used as augmentation sources;
- performance on held-out years;
- performance on held-out basins;
- performance by event mechanism, if defensibly labelled;
- sensitivity to random seeds;
- sensitivity to sampling strength.

## 34. Sampling-efficiency metrics

### Rare-event exposure

$$
X_R=\sum_{s=1}^{S}\#\{\text{rare events drawn at training step }s\}.
$$

Also report unique rare-event exposure $U_R$.

### Gain per unique rare event

For metric improvement $\Delta M$ over baseline:

$$
G_U=\frac{\Delta M}{U_R}.
$$

This is descriptive rather than a standard hydrological metric, but it helps distinguish genuine diversity gains from repeated exposure.

### Compute-normalised gain

$$
G_C=\frac{\Delta M}{\text{training compute or updates}}.
$$

Report the compute measure explicitly.

## 35. Synthetic-data acceptance tests

Before including synthetic data, verify:

- non-negative and bounded variables;
- plausible rainfall totals, intensities and spatial patterns;
- plausible antecedent-state combinations;
- plausible rainfall-runoff lag;
- runoff coefficient or water-balance ranges;
- hydrograph continuity;
- rising-limb and recession behaviour;
- peak magnitude and timing distributions;
- distance from source and validation events;
- no duplicate or near-duplicate leakage;
- observed-only test improvement;
- performance without synthetic data as an ablation.

Synthetic events should be labelled as synthetic throughout the pipeline and never counted as additional independent observed evidence.

---

# Part IX: Recommended first implementation

## 36. Initial sampler

Construct independent event and background catalogues for every basin. Use a hierarchical batch sampler:

1. Sample a basin with tempered probability

$$
p(b)=\frac{N_b^{0.5}}{\sum_jN_j^{0.5}}.
$$

2. Sample a case category with initial fractions such as:

- 25% rare peak events;
- 25% moderate or near-threshold events;
- 25% hard or hydrologically active non-events;
- 25% random background.

These are starting candidates for tuning, not evidence-based universal values.

3. Sample an independent event uniformly within the selected basin-category cell.
4. Sample at most a small, fixed number of windows per selected event in a batch.
5. Require a minimum number of distinct events and basins per batch.
6. Keep validation and test sampling natural.

## 37. First tuning grid

Compare:

### Basin temperature

$$
\tau_b\in\{0,0.5,1\}.
$$

### Rare-event batch fraction

$$
\rho_R\in\{\pi_R,0.10,0.20,0.35\}.
$$

### Background retention

$$
q_M\in\{1,0.5,0.25\}.
$$

### Hard-negative fraction

$$
\rho_H\in\{0,0.10,0.25\}.
$$

Tune in stages, not as a full Cartesian product:

1. basin balance;
2. rare-event fraction;
3. background retention;
4. hard-negative share.

## 38. Decision rule

Retain a sampling strategy only if it improves event-level peak performance on natural-prevalence validation data without unacceptable deterioration in:

- false alerts;
- moderate-flow bias;
- whole-hydrograph KGE;
- timing;
- probability calibration;
- lower-performing basins.

The best first production candidate is likely to be the simplest observed-data sampler on the Pareto frontier, not the most aggressively balanced sampler.

## 39. Recommended order of experimentation

1. Natural window sampling.
2. Uniform event sampling.
3. Event sampling with moderate rare-event oversampling.
4. Stratified event sampling with near-threshold quota.
5. Moderate background undersampling.
6. Hard-negative mining.
7. Basin-balanced hierarchical sampling, if regional.
8. Physically constrained observed-event augmentation.
9. Hydrological-model-generated events.
10. SMOGN, mixup or generative models as exploratory comparisons.

---

# Part X: Pros/cons summary

| Strategy | Main advantage | Main risk | Recommended role |
|---|---|---|---|
| Uniform event sampling | Prevents long events dominating | Ignores severity | Essential baseline |
| Random event oversampling | Simple and physically safe | Memorisation | First rare-event intervention |
| Severity-weighted sampling | Smooth peak emphasis | Concentration on few extremes | Strong candidate |
| Stratified sampling | Transparent batch control | Boundary sensitivity | Strong candidate |
| Near-threshold sampling | Improves operational discrimination | Does not target far tail | Add when warning thresholds matter |
| Background undersampling | Reduces redundancy and cost | Removes state coverage | Use moderately |
| Diversity-aware undersampling | Preserves background variety | Representation dependence | Useful at scale |
| Hard-negative mining | Directly attacks false alarms | Mines noise or label errors | Add after baseline |
| Curriculum sampling | Controls emphasis over training | More hyperparameters | Later ablation |
| Basin-balanced sampling | Prevents basin domination | Overweights short records | Important regionally |
| Perturbation augmentation | Adds local variation | Breaks physical relations | Use only with constraints |
| Mixup | Easy regularisation | Hydrologically implausible interpolation | Exploratory |
| SMOGN/SMOTER | Designed for continuous imbalance | Sequence and physics violations | Lower priority |
| Generative models | Potential nonlinear diversity | Difficult physical validation | Research option |
| Hydrological simulation | Forcing-runoff consistency through model | Simulator bias | Preferred synthetic route |
| Active simulation selection | Efficient simulation budget | Acquisition bias | Useful if simulation is costly |

---

# Part XI: Red flags

Revise or reject a sampler if:

- overlapping windows from one event appear in multiple splits;
- validation or test data are resampled;
- a few events account for most training updates;
- unique-event exposure stays low while window count rises sharply;
- peak recall improves only through unacceptable false alarms;
- ordinary-flow and antecedent-state coverage collapses;
- one or two basins dominate sampled gradients;
- synthetic events are physically inconsistent;
- synthetic cases are counted as independent observed extremes;
- calibration is assessed on a balanced validation set;
- hard-negative mining repeatedly selects data-quality failures;
- gains vanish when evaluated by independent event or hydrological year;
- results depend strongly on one random seed;
- tuning uses the final test set;
- the sampling strategy is changed without holding loss and architecture constant.

---

# Bibliography

1. Tofighi, S., Gurbuz, F., Mantilla, R., & Xiao, S. (2025). Advancing machine learning-based streamflow prediction through event greedy selection, asymmetric loss function, and rainfall forecasting uncertainty. *Applied Sciences, 15*(21), 11656. [Article](https://doi.org/10.3390/app152111656)

2. Branco, P., Torgo, L., & Ribeiro, R. P. (2017). SMOGN: A pre-processing approach for imbalanced regression. *Proceedings of Machine Learning Research, 74*, 36-50. [PMLR article](https://proceedings.mlr.press/v74/branco17a.html)

3. Moniz, N., Branco, P., & Torgo, L. (2016). Resampling strategies for imbalanced time series. *2016 IEEE International Conference on Data Science and Advanced Analytics*, 282-291. [DOI](https://doi.org/10.1109/DSAA.2016.42)

4. Kunz, N. (2020). `smogn`: Synthetic Minority Over-Sampling Technique for Regression with Gaussian Noise. Python package. [PyPI package](https://pypi.org/project/smogn/)

5. Alahyari, S., & Domaratzki, M. (2025). SMOGAN: Synthetic minority oversampling with GAN refinement for imbalanced regression. arXiv. [Preprint](https://doi.org/10.48550/arXiv.2504.21152)

6. Holman, K. D., Wright, D. B., & Yu, G. (2020). *Stochastic Storm Transposition for Physically-Based Rainfall and Flood Hazard Analyses*. U.S. Bureau of Reclamation, Final Report ST-2020-1735-1. [Report](https://www.usbr.gov/research/publications/download_product.cfm?id=2953)

7. U.S. Bureau of Reclamation. (2020). Development of web-based stochastic storm transposition toolkit for physically based rainfall and flood hazard analysis. [Project description](https://www.usbr.gov/research/projects/detail.cfm?id=1735)

8. Karlovits, G. (2023). Stochastic storm transposition in HEC-HMS. U.S. Army Corps of Engineers Hydrologic Engineering Center. [Technical article](https://www.hec.usace.army.mil/confluence/display/HECNews/Stochastic+Storm+Transposition+in+HEC-HMS)

9. imbalanced-learn developers. (2026). Common pitfalls and recommended practices: Data leakage. [Documentation](https://imbalanced-learn.org/stable/common_pitfalls.html)

10. Saito, T., & Rehmsmeier, M. (2015). The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets. *PLOS ONE, 10*(3), e0118432. [Article](https://doi.org/10.1371/journal.pone.0118432)

11. Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). SMOTE: Synthetic Minority Over-sampling Technique. *Journal of Artificial Intelligence Research, 16*, 321-357. [Article](https://doi.org/10.1613/jair.953)

12. Silvestrin, L. P., Pantiskas, L., & Hoogendoorn, M. (2021). A framework for imbalanced time-series forecasting. arXiv. [Preprint](https://doi.org/10.48550/arXiv.2107.10709)

13. Batista, G. E. A. P. A., Prati, R. C., & Monard, M. C. (2004). A study of the behavior of several methods for balancing machine learning training data. *SIGKDD Explorations, 6*(1), 20-29. [DOI](https://doi.org/10.1145/1007730.1007735)

14. He, H., & Garcia, E. A. (2009). Learning from imbalanced data. *IEEE Transactions on Knowledge and Data Engineering, 21*(9), 1263-1284. [DOI](https://doi.org/10.1109/TKDE.2008.239)

15. Branco, P., Torgo, L., & Ribeiro, R. P. (2016). A survey of predictive modeling on imbalanced domains. *ACM Computing Surveys, 49*(2), Article 31. [DOI](https://doi.org/10.1145/2907070)

---

## Source notes and evidence limits

- Reference [1] is the most directly relevant machine-learning hydrology source identified for event selection, stochastic storm generation and active learning. Its design and results apply to its Iowa River Basin study and should not be treated as universal prescriptions.
- References [6]-[8] support stochastic storm transposition and its connection to hydrological modelling, but they address rainfall and flood-hazard simulation rather than proving that synthetic events improve every ML forecast model.
- References [2]-[5], [11] and [13]-[15] are general imbalanced-regression or imbalanced-learning sources. Their methods require additional physical validation before hydrological use.
- Reference [3] explicitly extends resampling to imbalanced time-series forecasting and describes temporal and relevance bias in case selection, but it is not hydrology specific.
- Reference [9] supports the leakage rule that sampling should occur only after splitting and only within the training data.
- Numerical sampling grids in this document are proposed experiment designs, not values established as optimal by the cited literature.
