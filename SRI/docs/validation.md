# Validation scope

The executable test suite covers Gaia coordinate reconstruction, required-error failure, background isolation, separation of tangential transport from independent 3D centres, configuration validation, centre-score limits, partition recursion, RV-limited assignment and full/core consistency.

Source-level regression covers 126 labelled positive pairs, 407 optical-control pairs, 250 multiple systems and 11 stellar-complex reference cases. Full-score, partition and core tables agree with same-input, same-configuration references within floating-point tolerance (absolute and relative tolerance 1e-12). Grades, entity partitions and core membership agree exactly. JSON diagnostics are compared as parsed fields rather than literal floating-point strings.

The paired-sample check gives AUC 0.949573 and an optical-control selection fraction of 7/407 at the Gold boundary. It is an external consistency diagnostic in a nearby-pair regime, not a universal purity calibration. Sample selection and interpretation are specified in the accompanying paper.

Run `python -m pytest tests -q` for executable checks. The packaged example is an interface fixture, not a benchmark for scientific classification. These checks do not establish physical purity or robustness to every possible membership change. Two pandas future-compatibility warnings can arise from test-fixture concatenation without affecting the comparisons.
