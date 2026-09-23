# Detailed parameter reference

Defaults are loaded from `config/defaults.json`; an explicit JSON file can override supported settings. The README lists the four quantities recommended for routine sensitivity checks. This document explains their basis and the advanced settings. Save the effective `run_config.json` with results. Operational conventions are not confidence levels or universal physical constants.

## Cross residual scale: h_cross_kms = 5 km/s

The adopted value is motivated by an initial reference evaluation of 353 Group-C systems: the arithmetic mean of their dimensional, geometrically normalised Cross residual statistic was 4.994 km/s, rounded to 5 km/s. This is a system-level residual scale, not the internal stellar RV dispersion or a five-sigma boundary. The statistic in that reference evaluation differs from the uncertainty-adjusted residual definition specified in the method document; the reference establishes the scale's empirical origin, not a new calibration of the present estimator or a uniquely optimal value.

At fixed effective residual, increasing h gives more Cross support and decreasing h gives less. However, h also enters directional uncertainty attenuation through h^2/3, so the effective residual must be recomputed. The fitted affine field and entity-centre uncertainties do not depend on h. Only systems with an active Cross term are affected. Do not remap a saved effective residual as if it were independent of h.

## KPD prominence: outlier_prominence = 3

A component splits only when its longest connection divided by its reference length is strictly greater than this threshold. The default requires a pronounced relative gap and was assessed against a threshold of 2 as a more conservative operational choice. It does not imply three-sigma significance or a calibrated false-positive rate.

For a fixed component, a higher threshold accepts fewer splits and a lower threshold accepts more. Entire recursive spatial-then-velocity partitions need not form a nested sequence: an earlier split changes which later groups are tested. Recompute partitions and core selection after a change. KPD settings do not change the full-system SRI, but can change whether a core is available and which members it contains. Equality never triggers a split.

## Grade boundaries: gold_threshold = 0.70; silver_threshold = 0.50

These empirical recommendations describe intervals on the continuous SRI scale, informed by structural assessment and external pair comparisons. They are not association probabilities or a guarantee of equal purity across evidence groups. Raising either boundary reduces the number above it; lowering it increases that number. Only grades change, not component scores, KPD or core membership. The configuration requires 0 < silver_threshold < gold_threshold < 1.

## Advanced model settings

### Affine ridge strength: affine_ridge = 0.1

The slope penalty is alpha*N_train*||G||_F^2 in RMS-normalised position coordinates; the intercept is unpenalised. The value 0.1 is a stability--bias compromise for sparse or nearly degenerate spatial geometry. It is not a physical velocity-gradient threshold.

Larger values shrink slopes more strongly and can suppress genuine gradients. Smaller values permit stronger gradients but can amplify extrapolation and influential-entity effects. Predictions, training influence weights, geometric amplification and propagated residual errors can all change; total scores are not guaranteed to change monotonically. Use the default unless evaluating a different model with a complete validation set.

### Scoring RV eligibility: min_rv_sources = 3

This is the operational minimum valid RV-source count for qualified Vmem pair handling and for 3D scoring/KPD. It is not proof that three sources determine a precise dispersion. Changing it can alter the evidence group, Cen/Cross activation, RV-free fallback and KPD/core assignment; scores need not move monotonically.

The upstream merger veto has its own fixed three-source requirement on each node and a three-systemic-error conflict threshold. Changing scoring eligibility does not redefine that contraction rule. The published prescription uses 3 for both eligibility requirements.

## Response definitions and fixed method identifiers

All five response functions, including their numerical constants, are specified together in [method.md](method.md). `spatial_mem_scale=2` records the adopted spatial adjacency-response scale; it is part of the defined method, not a recommended routine tuning option. Changing it requires rebuilding spatial member support and every bridge-control gain for both full and core structures. No additional velocity-member or Cen response parameters are introduced for superficial symmetry.

The following identifiers record fixed construction choices rather than selectable alternatives:

| Identifier | Required value | Meaning |
| --- | --- | --- |
| `velocity_center_method` | `component_median` | Independent component medians rotated to entity UVW |
| `spatial_cen_width_model` | `smem_scatter` | Directional extent from the shared robust member scatter |
| `spatial_cen_tested_edge_floor` | `true` | Include the tested endpoints' widths in spatial protection |
| `cross_aggregation` | `mean_residual` | Average adjusted residual magnitudes |
| `cross_cap_single_largest` | `false` | No largest-loss capping |

Other identifier values are rejected. The contraction ratios <=1, upstream RV-conflict threshold, 720 projection directions and 100 bridge rotations are explicit construction or numerical-resolution conventions; they are not all exposed as user switches.

## Runtime uncertainty policy

`use_uncertainty=true` validates required measurements and propagates systemic centre errors. Invalid required values fail the system explicitly, never silently becoming zero uncertainty. Larger errors can attenuate evidence against compatibility; they do not prove association.

Explicit false mode uses nominal scoring and disables the uncertainty-dependent RV merger veto. Entity partitions can therefore change as well as scores. It is not equivalent to zeroing only the final covariance while freezing the rest of the pipeline. `USE_UNCERTAINTY=None` follows the JSON setting; an explicit runtime or command-line flag overrides it.

Input paths, output paths, column names, bridge support, scope, KPD execution and derived-table storage are runtime controls. They should be chosen to match the input data, not to optimise grades. See the README for usage.

## Reporting changes

Change one setting at a time, keep raw inputs and uncertainty policy fixed where appropriate, and report full distributions and partition changes rather than only favourable examples. A change from the defaults defines a different scoring or selection prescription and should be stated with the results.
