# SRI

SRI measures the structural compatibility of a stellar complex from Gaia source observations. It returns five component scores, a full-system score and grade, an independent Kinematic Partition Diagnostic (KPD), and a separately recalculated core score when a unique core is identified.

## Quick start

Python 3.10 or later is required. Open a terminal in this directory:

```sh
python -m pip install -r requirements.txt
python run_sri.py --input examples/raw_basic/Snake_Member_Catalogue.csv --parent none --output example
```

For your own data:

```sh
python run_sri.py --input /path/to/members.csv --parent none --output results
```

The distribution does not include research input or output catalogues. The small `examples/` fixture is provided to test the interface, not as a scientific sample. Relative paths are resolved against the working directory. In Spyder, set this directory as the working directory or edit the absolute paths in the USER DEFAULTS block of `run_sri.py`.

## Input

Supply one CSV with one row per source. The default identifier columns are `Snake`, `id_node`, and `source_id`. Override them using `--group-col`, `--node-col`, and `--source-col`. Results always report the original system identifier as `system_id`. Source IDs are read as strings; duplicates within a system are rejected.

| Column | Units and requirements |
| --- | --- |
| `ra`, `dec` | ICRS degrees |
| `parallax` | Positive finite mas; distance is 1000/parallax pc |
| `pmra`, `pmdec` | mas/yr; pmra includes cos(dec) |
| `radial_velocity` | km/s; missing measurements are allowed |
| `parallax_error`, `pmra_error`, `pmdec_error` | Positive finite uncertainties in strict uncertainty mode |
| `radial_velocity_error` | Positive finite km/s uncertainty whenever RV is measured |
| `parallax_pmra_corr`, `parallax_pmdec_corr` | Optional correlations; omitted columns imply zero correlation; supplied values must be finite and within [-1,1] |

Derived positions, velocities, and scores supplied in the input are not used for scoring. Each system requires at least two base nodes with sufficient non-degenerate member clouds. Compatible nodes may form common scoring entities. If contraction produces only one entity, scoring uses the base-node representation.

## Bridge sources

A blank node label identifies a potential bridge source. With `--parent none`, these rows are ignored. To enable spatial bridge support:

```sh
python run_sri.py --input members.csv --parent true --bridge-scope-col id_part --output results
```

Use `--bridge-scope-col none` for unrestricted support within each system. An explicit `--parent none` disables an inherited scope unless another scope is explicitly requested. The packaged USER DEFAULTS block disables bridge support. Enable it explicitly when bridge sources are part of the input.

Bridge sources affect only the spatial member-support correction. They do not enter intrinsic entity widths, member counts used to choose the core, velocity centres, or velocity scores. The correction uses 100 line-of-sight rotations and can only add support. This internal gain rank is not a catalogue-wide score calibration or association probability.

## Method

- **Smem:** mean spatial member-cloud adjacency along its minimum spanning tree (MST), with response `1/sqrt(1+(d/a)^2)` and default `a=2`.
- **Scen:** the longest physical centre-tree edge relative to the other edges' 75th percentile, with directional member-extent protection.
- **Vmem:** mean `exp(-d)` tangential member-cloud support on its own MST. RV-qualified pairs use a common-plane model transport; other pairs use the RV-free representation at both ends. Modeled transport velocities are not individual observed 3D velocities.
- **Vcen:** the longest physical UVW-tree edge relative to the other positive edges' 75th percentile, with finite uncertainty attenuation of excess discontinuity.
- **Cross:** leave-one-entity-out affine velocity prediction from position, using training-only influence weights and direction-dependent residual uncertainty attenuation. Adjusted residual magnitudes are averaged without largest-loss capping.

Vcen and Cross use independently estimated entity UVW centres: valid source Vra, Vdec, and RV medians, rotated at the entity mean sightline. No missing stellar RV is filled to construct these centres. The systemic variance of each component is defined as `I^-1 + s_int^2/N_valid`; each component uses its own valid measurements. Rotating the component variances gives the centre covariance. It is a centre-uncertainty approximation, not the full member velocity dispersion or an exact sampling covariance of medians.

Let M be the number of RV-qualified entities. Both Cen terms are active for M >= 3; Scen then uses all spatial entities, whereas Vcen and Cross use only RV-qualified entities. Cross is active for M >= 4. The aggregation is:

```text
S = Smem                    if M < 3, otherwise sqrt(Smem * Scen)
V = Vmem                    if M < 3, otherwise sqrt(Vmem * Vcen)
R0 = 2*S*V/(S+V)            (zero when either channel is zero)
SRI = R0                    if M < 4, otherwise R0 * Cross
```

Evidence groups are A (M < 3), B (M = 3), and C (M >= 4). Recommended grades are Gold (SRI >= 0.70), Silver (0.50 <= SRI < 0.70), and Bronze (SRI < 0.50). These are operational score thresholds, not calibrated probabilities. Weak constraints can reduce evidence against compatibility; they do not establish a reliable physical association.

See [method details](docs/method.md) and [validation scope](docs/validation.md).

## Configuration and uncertainty

See [parameter rationale and adjustment effects](docs/parameters.md) for the origin, meaning and limitations of each default.

Scientific defaults are in `config/defaults.json`. `CONFIG=None` loads this file. Use `--config settings.json` to supply an explicit configuration; unspecified keys use packaged defaults. Restart Python after editing the defaults file.

The four parameters recommended for routine sensitivity checks are:

| Parameter | Default | Effect of adjustment |
| --- | ---: | --- |
| `h_cross_kms` | 5 km/s | Sets Cross residual tolerance; changing it requires recomputing both error attenuation and the response |
| `outlier_prominence` | 3 | KPD requires a strictly larger gap ratio; a higher threshold generally accepts fewer splits, without changing full SRI |
| `gold_threshold` | 0.70 | Raising it reduces Gold classifications; scores and partitions do not change |
| `silver_threshold` | 0.50 | Raising it increases Bronze classifications; scores and partitions do not change |

Default values define the reported method. Parameter changes are sensitivity analyses, not per-object grade corrections. Ridge strength and RV eligibility are advanced settings, described in [the detailed parameter guide](docs/parameters.md). Response formulas and their constants are defined together in [the method document](docs/method.md), rather than offered as a menu of alternative kernels.

Runtime settings are separate: input/output paths, identifier columns, bridge policy and scope, uncertainty mode, KPD execution and derived-table storage. These are configured in the USER DEFAULTS block or by the documented command-line options.

Method identifiers in the configuration specify the published construction, not alternative research methods. Unsupported estimator, width, or residual-aggregation settings are rejected.

`USE_UNCERTAINTY=None` follows the configuration. An explicit True/False or `--use-uncertainty` / `--no-use-uncertainty` overrides it. In strict mode, invalid required errors or covariances fail the system with a recorded reason; they never silently become zero error. Explicit nominal mode disables error attenuation and the uncertainty-dependent RV merger veto, so the entity partition can also change.

## KPD and the core

KPD does not alter the full-system SRI or remove input sources. It recursively partitions spatial member clouds, applies an RV-free two-dimensional test only to RV-limited sides, and then partitions RV-qualified 3D centres within each part. Each step tests the longest MST edge against the second longest. Spatial and 2D reference lengths have a shared-H floor of `1/sqrt(dimension)`; 3D uses centre distances without an error or cloud-width floor. The ratio must be strictly greater than 3. Components with fewer than three entities are not split further.

RV-limited entities that remain unseparated are assigned by their RV-free cloud distances to fixed RV-qualified components. Exact assignment ties remain unresolved. A candidate flag indicates an accepted partition, not confirmed individual outliers. Unassessed systems are distinct from assessed systems without a split.

For KPD=1, the core is the unique component with the most entities; member-star count breaks ties. A remaining tie or a core smaller than two entities prevents core scoring. The core is rebuilt from its sources and compatible bridge candidates, preserving its entity partition. Excluded node members are not converted to bridge sources. Core scores and evidence groups are reported separately and do not replace full-system grades.

## Outputs

- `SRI.csv`: full scores, evidence group, grade, KPD and core summary.
- `KPD.csv`: partitions, channel flags, tested ratios and core status.
- `Core_SRI.csv`: successfully calculated core scores and components.
- `Derived_Entity_Kinematics.csv`: full/core entity centres and systemic uncertainties.
- `Derived_Node_Catalogue.csv`: derived node summaries; omit with `--no-derived`.
- `failures.csv`: failed systems and reasons; a strict-mode core failure also fails that system.
- `input_mapping.json`, `run_config.json`: input identity mapping and effective configuration.

Use `--no-kpd` to omit KPD/core computation. Exit code 0 means success; 2 indicates system failures. Existing result files in the selected output directory may be overwritten. Input files are not modified.

## Verification

```sh
python -m pytest tests -q
```

`MANIFEST.sha256` contains hashes of distribution files. The archive excludes local `input/`, `output/`, caches, backups, and research experiments.
