# Method and interpretation

## Entities and observables

Source distances are 1000/parallax pc. Tangential velocities are 4.74047*proper_motion/parallax km/s, with pmra including cos(dec). Tangential error propagation includes the supplied parallax--proper-motion correlations; omitted correlation columns imply zero correlation. The full five-parameter astrometric covariance and sky-position errors are not propagated.

Spatial contact for contraction is the minimum inter-cloud source separation divided by the sum of the nodes' median internal XYZ-MST lengths. Tangential compatibility is the largest projected median separation divided by summed projected MAD, evaluated over 720 directions in the common plane. Both ratios must be <=1. For groups, spatial contact uses the minimum cross-group ratio and velocity compatibility the maximum, preventing velocity chaining. A three-source-qualified RV conflict above three combined systemic-error units vetoes a merger. The entire structure contracting to one entity triggers base-node scoring instead.

Spatial entity centres are member-count-weighted means of the constituent nodes' coordinate medians. They need not equal pooled member medians. Intrinsic member clouds contain node members only.

## Member scores

For each cloud, estimate a robust scatter matrix S from pair differences, with median-length clipping, dimension calibration and pilot-whitening refinement. For a cloud pair, H=sqrt(S_A)+sqrt(S_B). Transform both clouds by H^-1 and calculate

```text
d = [median_A(nearest distance to B) + median_B(nearest distance to A)] / (2*sqrt(p))
spatial_response = 1/sqrt(1+(d/a)^2), a=2
velocity_response = exp(-d)
```

Average responses along the respective member-distance MSTs. The spatial a value is a response convention, not a confidence level. Vmem uses modeled perspective transport only for RV-qualified pairs; both ends of other pairs use RV-free tangential clouds. This transport is separate from independent 3D centre estimation.

Eligible bridge sources are assigned to the nearest entity member cloud within the requested scope. The spatial MST is fixed before adding bridge sources. With observed gain g, base score S0, and midrank q among 100 line-of-sight rotations,

```text
Smem = clip(S0 + max(g,0)*max(2*q-1,0), 0, 1).
```

The gain cannot reduce continuity. Rotations preserve configuration geometry and distances, not the full extinction or density field. The rank measures additional spatial support, not association probability. Bridge sources never enter intrinsic widths or velocity centres.

## Spatial centre score

Build the physical XYZ centre MST and select its longest edge L. Let b be the linearly interpolated Q75 of all remaining edge lengths. Project every entity scatter matrix along the same longest-edge unit direction u:

```text
w_i = sqrt(u.T S_i u)
W_test = w_i + w_j
W_ref = median(w_a + w_b for all remaining tree edges)
B = max(b, W_test, W_ref)
Scen = sqrt(min(1, B/L)).
```

Identical centres give unity. Directional projected extent and the radial unit scale of an H ellipsoid are different geometric quantities; sharing S does not require replacing one with the other.

## Independent 3D centres and systemic errors

Take medians of valid source Vra, Vdec and RV separately, and rotate these three components at the mean entity sightline into heliocentric Galactic UVW. No solar-motion correction is applied. Each component uses its own valid observations. Missing stellar RVs are not filled to construct this centre.

For component k,

```text
I_k = sum(1/epsilon_sk^2)
s_obs,k = 1.4826 * MAD(v_sk)
s_int,k^2 = max(0, s_obs,k^2 - median(epsilon_sk^2))
sigma_sys,k^2 = 1/I_k + s_int,k^2/N_valid,k
C_entity = B_rotation @ diag(sigma_sys,k^2) @ B_rotation.T.
```

This independent-component model approximates representative-centre uncertainty. It is neither the full member dispersion nor an exact sampling covariance of component medians. Broad sky coverage and non-random RV availability limit the centre approximation. Strict uncertainty mode requires valid errors; explicit nominal mode supplies zero covariance to scoring and disables the RV conflict veto.

## Velocity centre score

Use the physical UVW MST of RV-qualified entities. L is the longest edge; b is the linearly interpolated Q75 of the remaining positive edge lengths. Keep the nominal tree and quantile ordering fixed. Propagate the gradient of Delta=L-b through independent entity covariances, including shared endpoints and quantile interpolation:

```text
sigma_delta^2 = sum(gradient_e.T @ C_e @ gradient_e)
d = max(0,L-b)/b
q = sigma_delta^2/b^2
d_eff^2 = min(d^2,1) + max(d^2-1,0)/(1+q)
Vcen = 1/sqrt(1+d_eff).
```

Tied reference lengths use their mean gradient. Zero uncertainty gives sqrt(b/L). For d>1, arbitrarily large uncertainty cannot raise the score above 1/sqrt(2). Identical velocity centres give unity; a positive longest edge without a positive reference is invalid. This conditional propagation does not describe uncertainty in MST selection.

Both centre terms are active only with at least three RV-qualified entities. Scen then uses all spatial entities, while Vcen uses only RV-qualified entities.

## Cross relation

For each held-out entity, fit an affine velocity field to the other entities. Positions are centred and RMS-normalised; the intercept is unpenalised. Minimise weighted squared residuals plus alpha*N_train*||G||_F^2, with alpha=0.1. Influence weights use training velocities only. For sufficiently many entities, nested deletion measures influence; lower multiplicity uses ridge/PRESS influence. Weights are moderated to maintain effective support.

With fixed fitted weights, let Q_ij be the residual operator and A_i=1+sum(w_ij^2) the geometric amplification from the unweighted scalar ridge design. Define

```text
r_i = sum_j(Q_ij @ v_j)/sqrt(A_i)
E_i = sum_j(Q_ij @ C_j @ Q_ij.T)/A_i.
```

At the target sightline, separate radial residual r_r and variance sigma_r^2, and diagonalise covariance within the actual 2D tangent plane. Set tau^2=h^2/3, h=5 km/s:

```text
phi(r,sigma2) = min(r^2,tau^2) + tau^2/(tau^2+sigma2)*max(r^2-tau^2,0)
radial_adjusted = min(r_r^2,tau^2)
  + tau^2/(tau^2+sigma_r^2)*sqrt(r_r^2/(r_r^2+sigma_r^2))*max(r_r^2-tau^2,0)
g = 1 - (M-4)/M * (1 - 1/sqrt(1+||r_t||^2/(2*tau^2)))
loss_i = sum_k(phi(r_tk,sigma_tk^2)) + r_r^2 - g*(r_r^2-radial_adjusted)
D = mean(sqrt(loss_i))
Cross = 1/sqrt(1+(D/h)^2).
```

Zero-over-zero residual fractions use a finite limiting loss. No largest-loss capping is performed. Four qualified entities permit regularised computation but do not ensure prediction identifiability; geometry status is reported. Error attenuation does not create independent evidence of association. The score does not guarantee a strict 1/M influence bound for an anomalous entity.

## Aggregation and grades

For M<3, S=Smem and V=Vmem. Otherwise S=sqrt(Smem*Scen) and V=sqrt(Vmem*Vcen). R0=2*S*V/(S+V), with zero-score boundaries evaluated by continuity. For M<4, SRI=R0; otherwise SRI=R0*Cross.

Groups A, B, C correspond to M<3, M=3, M>=4. Gold starts at 0.70 and Silver at 0.50. These empirical recommendations are not calibrated probabilities. There is no catalogue-level null percentile in the final score.

## KPD and core

KPD uses the same entities independently of SRI. It first partitions spatial member-distance trees. Within each spatial block, an RV-free member-distance test may split a side composed entirely of RV-limited entities. RV-qualified centres are then partitioned in physical UVW.

```text
reference = max(second_longest_edge, 1/sqrt(p))  # member-cloud distances
reference = second_longest_edge                 # 3D centre distances
split when longest_edge/reference > 3.
```

Use p=3 for spatial clouds and p=2 for RV-free velocity clouds. The floor is the unit shared-H scale after dimension normalisation, not an uncertainty interval. Each accepted step removes one edge and recurses; fewer than three entities are not split further. Edge ratios receive no direct uncertainty attenuation.

Unseparated RV-limited entities join the nearest fixed qualified component by RV-free cloud distance. Exact ties remain unresolved. Any accepted split sets KPD=1; an assessed system without a split has KPD=0. Fewer than three entities means unassessed.

The core is the largest final component by entity count, with member-source count breaking ties. A unique core with at least two entities is rebuilt from its sources and compatible bridges without contraction. Excluded member sources do not become bridges. The core score and group are distinct from the full score; a group change prevents treating the score difference as a calibrated reliability gain.

## Configuration conventions

`config/defaults.json` controls h, KPD prominence, spatial response scale, ridge strength, scoring RV eligibility, grade boundaries and uncertainty mode. Method identifiers are fixed to the construction above; alternatives are rejected. The upstream merger's three-source RV requirement and three-systemic-error veto, 720 projection directions and 100 background rotations are explicit fixed construction conventions, not all exposed as user switches. Changing scoring RV eligibility does not redefine the upstream merger veto. No threshold is inferred by fitting the input catalogue's quality distribution.
