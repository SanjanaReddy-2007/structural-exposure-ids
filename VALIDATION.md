# Implementation validation

Checked on 8 October 2026 using the isolated Python 3.12 environment and versions
recorded in `requirements.lock.txt`.

All 10 invariant tests pass, with no skips. They check metadata rejection,
conflicting labels, duplicate grouping, chronological ordering and timestamp ties,
training-only preprocessing, causal graph windows, known one/two-hop exposure,
nested arc retention, degree preservation and independence of earlier training
predictions from future-flow feature changes.

Completed synthetic end-to-end runs:

| Scenario | Model conditions | Test flows per condition |
| --- | ---: | ---: |
| Random, two seeds, five levels, original/rewired | 26 | 240 |
| Separate-graph inductive, original/rewired | 5 | 240 |
| Chronological, original/rewired, boundary gap | 5 | 240 |
| Degree-preserving cross-edge reduction | 6 | 240 |
| Final integration check | 5 | 240 |

Each scenario includes Random Forest, MLP and the no-neighbour GraphSAGE control.
All outputs are marked synthetic. The random sweep's prediction records were
checked for identical IDs across conditions. Model checkpoints, audit manifests,
preprocessing, results, summaries and plots were written successfully.

The zero-cross-edge target was infeasible for one degree-controlled synthetic
condition: 229 undirected cross edges remained from 3,281 initially. Its diagnostic
records `target_reached: false`. Two-hop exposure can saturate even when cross
connectivity decreases; requested retention levels are not exposure measurements.

Earlier runs encountered a plotting-cache write warning while still producing
plots successfully. New runs keep that cache inside their output folder.

No CICIDS2017 research run has been performed. The official download form returned
a server error. Synthetic validation does not establish the scientific hypothesis,
publication suitability or patentability. Dataset curation, equal-budget model
tuning, repeated research experiments and statistical uncertainty analysis remain.
