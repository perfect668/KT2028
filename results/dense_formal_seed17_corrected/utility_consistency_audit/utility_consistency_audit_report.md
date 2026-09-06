# Dense PPO Utility Consistency Audit

- Scope: read-only audit plus interface repair; no training, warm-start, Validation or Test access.
- Execution source: **confirmed**. Server `formal_dense.py` before repair matched archived commit `e1a6ec086d02ff2c29d26f0f4e216da7ce9495b2` (SHA256 `4d3f8909af406fa3691230b610cf92d243a02a099b605529ba72382610e8c918`).
- Checkpoint status: `final_dense_seed17_balanced.pt` is a **protocol-affected historical result** because its action gains, warm-start targets and lambda quantiles were generated through the inconsistent local utility. It was not modified or retrained.

## Findings

`formal_dense.py::phi` returned positive BCE plus penalties, while `rl_core.core::evidence_potential` returns negative BCE plus the same penalties. The environment used `evidence_potential`; `action_gains`, warm-start targets and lambda quantiles used the local `phi`. Thus an improvement toward the true label had the wrong sign in the action-generation path.

Independent hand cases (labels 0 and 1, fixed disagreement, zero cost) pass for the unified function: correct-direction probability changes have positive gain and deterioration has negative gain. This test is independent of the old implementation.

## Patient/action comparison

Using 8 Train patients, k=0/1/2/4, all legal actions (1,992 rows):

- maximum old-vs-unified gain difference: **0.8360993967409187**
- sign mismatch rate: **99.1466%**
- best-action change rate: **100%**
- unified gain of actions selected by old rule: **-0.0495886** mean
- mean regret relative to unified best action: **0.0400688**

The comparison is repeated for the four existing lambda values in `lambda_net_reward_comparison.csv`; cached net reward is exactly the environment formula (`evidence_potential` delta minus lambda times cost), with maximum absolute error **0.0**.

## Cost interface

The non-zero-history warm-start observation set `cumulative_cost` to zero. For k>0, the selected-region cost sum is non-zero, so this is a confirmed interface inconsistency. The repair is recorded; historical checkpoints remain unchanged.

## Repair

`formal_dense.py` now imports and calls the unique `evidence_potential` implementation and no longer defines a duplicate `phi`. No network, reward, STOP, GAE or PPO behavior was changed.

Verification: `py_compile` passed; `python -m MSRLSelect.rl_core.test_core` passed; no training entrypoint was run.
