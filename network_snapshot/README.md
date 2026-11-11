# KT2028 network snapshot — 2026-11-11

This branch archives the current feature-based prediction networks and experiment
drivers. Its new commit contains code, software tests, registered configurations
and anonymous aggregate logs. The existing repository history is retained.

## Networks and entry points

| Module | Role |
| --- | --- |
| `kt2028/survival_network.py` | R3: coarse context, coarse-query fine attention, cross-gated dual streams, five-interval survival head |
| `kt2028/multicancer_network.py` | Independent, shared and cancer-conditioned four-expert prediction |
| `kt2028/selection_network.py` | 128-d predictor, fixed-budget region selector and attention-output gate ablation |
| `kt2028/dsca_source_adapter.py` | Pristine official DSCA modules with missing-child masking and our annual-target adapter |
| `kt2028/run_multicancer.py` | Registered five-fold, four-cancer independent/shared/MoE suite |
| `kt2028/run_selection_ablation.py` | Fixed 20% parent-budget development, fixed-selector and alternating updates; inner validation only |
| `kt2028/run_dsca_comparison.py` | Official-recipe four-cancer fold-0 comparison and two CRC training controls |

The feature loader, likelihood and censoring-metric helpers are included through
the local dependency closure recorded in `source_manifest.json`. UNI itself is
frozen; this snapshot contains neither pretrained weights nor feature extraction
assets. EAGLE-Net is not implemented in this snapshot.

## Environment and checks

Install `network_snapshot/requirements.txt` into a separate environment. PyTorch's
CUDA wheel must match the host. The dependency list covers feature-based training;
it does not install the optional UNI pixel-encoding stack.

From the repository root:

```bash
python -m unittest discover -s tests -p test_multicancer.py
python -m unittest discover -s tests -p test_selection.py
DSCA_SOURCE=/path/to/pristine/DSCA python -m unittest discover -s tests -p test_dsca_source.py
```

For DSCA, obtain the author's repository, https://github.com/liupei101/DSCA,
and check out `a749074cb3415c3a6eb83bee39191bdb69968dcc` unchanged. Verify the
65 source-file hashes in `configs/dsca.json`. Third-party DSCA source and its
example patient splits are not vendored here.

## Private run directory contract

Each driver accepts `--root /path/to/run`. Prepare that directory privately:

* `config.json`: copy the corresponding registered config from `configs/`.
* `patients_private.json`: local patient/slide manifest compatible with
  `run_multicancer.Dataset` and `mixed_survival_data.inspect_asset`; the same
  patient key must never cross partitions. No placeholder clinical labels are
  supplied by this archive.
* `features/<asset_id>`: existing paired Zarr assets with coarse/fine embeddings,
  coordinates, nominal physical geometry and fine-parent mapping. The existing
  UNI weight hash and geometry checks remain in force.
* DSCA additionally requires the unchanged `official/` source snapshot and, when
  using separate dependencies, `pydeps/`.

The main multicancer and selection drivers start training directly when invoked
with a prepared root. Their inherited `training_default` config text is not an
execution guard. Only the DSCA driver additionally enforces `--execute`.
The multicancer suite uses all five outer folds; do not rerun it simply to view
logs. Selection is a single-slate REINFORCE policy, not PPO or a world model.

## Logs and evidence

`logs/training_summary_w143.json` is the dated, anonymous W143 result snapshot:
17/30 main fits, 10/10 selection fits and 6/6 DSCA/control fits completed at that
time. It records fold counts, selected epochs and aggregate metrics. It is not a
final five-fold result and is not refreshed automatically. `validation.json`
records software checks of this source snapshot, not another training run.

Private raw logs, per-patient predictions, labels/splits, features, checkpoints,
WSIs, server addresses and access credentials are excluded from the new commit.
Training code writes patient-level outputs under the private run root; keep that
root outside the checkout. The original development checkout and live training
processes were not changed for this publication.
