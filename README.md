# Structural exposure in graph based intrusion detection

Research prototype for the Computer Networks project. It compares Random Forest,
a parameter-matched MLP, mean GraphSAGE and GraphSAGE without neighbour messages.
The question is how graph exposure affects the GraphSAGE–MLP performance gap.
This is an experimental pipeline, not a live network detector.

## First run on Windows

An isolated Python 3.12 environment is configured at `.venv`.
`requirements.lock.txt` records the tested dependency versions. On another machine,
create a Python 3.12 environment and install that lock file with pip.
From this project folder in PowerShell:

```powershell
.\.venv\Scripts\python.exe -m exposure_ids.cli demo --out data/synthetic
.\.venv\Scripts\python.exe -m exposure_ids.cli run --data data/synthetic --out runs/demo-random --protocol random --seeds 1 2 --epochs 30 --rewire
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The synthetic CSV created during setup is already available; omit the `demo`
command if it exists. `Run-Demo.ps1` handles this automatically and chooses a fresh
output folder:

```powershell
.\Run-Demo.ps1
```

Synthetic scores only validate software; they cannot support the project claim.
Use a fresh output folder for each run. Existing results are not overwritten.

## Obtain the research dataset

Use the [official CICIDS2017 page](https://www.unb.ca/cic/datasets/ids-2017.html).
Choose **GeneratedLabelledFlows**, retaining source/destination IP, Timestamp,
Flow Duration and Label. Do not substitute a stripped MachineLearningCSV export
or an undocumented mirror. As checked on 8 October 2026, the linked official
download form returns a server error. No real dataset is bundled with this repo.
If a download form needs personal information, enter your own details yourself.

Extract the actual labelled CSVs into `data/raw/cicids2017/`. The reader accepts
common Source IP/Src IP and Destination IP/Dst IP column spellings. Timestamp
parsing defaults to day-first; inspect the audit and use `--monthfirst` if the
source release requires it. Flow Duration is interpreted as microseconds.
Real-data metadata, labels, durations and duplicates must pass the audit.
Invalid metadata or conflicting duplicate annotations stop execution; they are
not silently removed or relabelled.

```powershell
.\.venv\Scripts\python.exe -m exposure_ids.cli audit --data data/raw/cicids2017 --out runs/initial-audit --protocol chronological
```

`--per-file-limit 5000` is a prefix-per-file development limit, **not representative
sampling**. Do not publish that subset's scores as full-dataset findings. The
full-batch graph model is intended for a manageable subset initially; millions
of flows need a reviewed sampling scheme and mini-batch training before use.
Loading currently concatenates CSVs in memory; even an audit of the complete
dataset needs sufficient RAM. There is no automatic download or PCAP re-extraction.

## Experiments

```powershell
# Fixed record split; nested deletion levels; repeated model seeds
.\.venv\Scripts\python.exe -m exposure_ids.cli run --data data/raw/cicids2017 --out runs/random --protocol random --rewire

# Additional degree-preserving connectivity control on the same split
.\.venv\Scripts\python.exe -m exposure_ids.cli run --data data/raw/cicids2017 --out runs/random-degree --protocol random --degree-controlled

# Graphs independently constructed within each partition
.\.venv\Scripts\python.exe -m exposure_ids.cli run --data data/raw/cicids2017 --out runs/inductive --protocol inductive --rewire

# Earlier completed flows only; optional boundary embargo
.\.venv\Scripts\python.exe -m exposure_ids.cli run --data data/raw/cicids2017 --out runs/chronological --protocol chronological --gap-seconds 60 --rewire
```

Defaults: 60/20/20 record-group split, 60-second graph window, 12 candidate
neighbours, 32 GraphSAGE hidden units, 100 epochs, patience 15, five model seeds.
Split seed and graph seed are separate and remain fixed across model seeds.
Connectivity levels are 0%, 25%, 50%, 75%, 100% retained candidate cross edges;
they are **not** percentages of neighbourhood exposure. Threshold is fixed at
0.5, not optimized against test labels. Numerical preprocessing, loss weights and
early stopping use only training or validation labels as appropriate.

## Information access rules

Every node is a completed flow. Numeric features exclude IPs, IDs, absolute times
and labels. Edges join flows sharing any endpoint host within the window.
Candidate selection is bounded, most-recent-first and label-independent.

* **Random:** transductive diagnostic. Unlabelled test features may influence
  training through train/test graph edges. Validation uses a separate train/val
  view built independently, fixed across test connectivity conditions. Validation and test have no
  graph paths through each other in the respective views. This declared setting
  must not be presented as a deployment-realistic result.
* **Inductive:** build graphs after splitting, independently per partition.
  No held-out message path reaches training. Hosts can recur; this does not mean
  hosts are disjoint. Only training labels contribute to supervised gradients.
* **Chronological:** partition by flow completion time, keeping timestamp ties
  together. Directed message arcs run from earlier to later completed flows;
  held-out future flows cannot affect training predictions, including through
  multiple layers. Earlier validation flows can be unlabelled history for test
  predictions. Graph features use no global/batch normalization. This is a
  flow-completion detector, not detection before a flow finishes.

Exact duplicate flow identities stay together. The audit separately reports
feature-identical test rows seen in training. Those are not automatically the same
flow, and they are not silently removed. A single-class chronological training
partition fails instead of switching to class-stratified time splitting.

## Measurements and controls

Host reuse is distinct test hosts seen in training divided by all test hosts.
Cross-split arc fraction is cross-partition directed message arcs divided by
all arcs in the declared view. One/two-hop exposure counts test nodes reachable
from training through one/two directed message steps. An empty test class gives
an undefined recall/FPR/AP rather than a misleading perfect score.

GraphSAGE implements the mean aggregation equation with sparse native PyTorch
operations, without PyG or learned IP embeddings:

`h_next = ReLU(W_self h + W_neighbor mean(incoming neighbour h))`

Two such layers feed a flow classifier. MLP width is selected by nearest active
parameter count to full GraphSAGE. The no-neighbour control freezes the unused
neighbour weights; total and active parameter counts are reported separately.

Degree-preserving rewiring retains temporal eligibility and partition-pair edge
counts while disrupting shared-host relationships. It does not guarantee equal
two-hop exposure: inspect the measured change. In random degree-controlled
sweeps, swapping two train/test edges into within-partition edges preserves
degree but changes relational semantics. Targets may be infeasible; the achieved
cross-edge count and swap success are logged. Neither method alone establishes
a pure causal effect of host familiarity.

## Outputs

Each run writes:

* `audit.json`: hashes, row/feature counts, labels, duplicates, missing values,
  partition support and unseen test labels.
* `split_manifest.csv`: exact records, flow identity groups, hosts, times and splits.
* `preprocessing.json`, `config.json`, `versions.json`: reproducible settings.
* `prepared.npz` and condition edge arrays: the numerical dataset and graphs.
* Model weights, validation histories and rewiring/control diagnostics.
* `results.csv` / `results.json`: macro-F1, recall, FPR, **average precision**,
  confusion matrices, graph exposure and per-seed paired gap to MLP.
* `summary.csv`: per-condition means and seed standard deviations (not sampling
  confidence intervals).
* `predictions.csv`: probabilities on identical test records for every model.
* `exposure_gap.png`: measured exposure against performance difference, labelled
  as synthetic when applicable.

Average precision is reported explicitly; it is not trapezoidal PR-AUC.
Rewiring currently uses a fixed original validation graph to keep early stopping
conditions fixed. This policy must be stated in the paper, with a sensitivity
check using corresponding rewired validation graphs before strong conclusions.

## Before conference conclusions

The prototype uses fixed hyperparameters, not equal-budget tuning. Complete the
real-data audit, meaningful subset design, five-seed principal runs, second-dataset
replication, tuning budget comparison, and host/session/time-block uncertainty
analysis. Seed variability is not a population confidence interval. Review
degree-control feasibility and exposure drift in topology ablations. Tests check
software invariants; they do not establish research novelty or the hypothesis.

References: [GraphSAGE](https://arxiv.org/abs/1706.02216),
[mean aggregation implementation reference](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.SAGEConv.html),
[average precision definition](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html).
