# Position mechanism: discovery protocol v1

## Question and scope

What internal changes transmit sensitivity to the order of alternatives in Laya?
Primary model: the multilingual public Laya checkpoint pinned by src/activations.py
(7b928d828b7b0e022f929d9bd2e44165aa270148), laya 0.4.30, sequential option layout.
Dataset: LocalLLaMA/typed-decisions, pinned c76749ec58bd8c3d2ea706b31c333a9059c38f90.
Only `choice` is included. No new SAE or decision-model training is required.
This is a discovery study in one checkpoint, not a universal explanation of System One.

## Separation of data and inference

Reuse the project split by case: 1,020 discovery / 180 dev / 400 test cases.
All permutations of a case belong to the same partition. Discovery case ordering is a
fixed hash independent of predictions and gold labels. Use all discovery choice decisions
for behavior. Patch all choice decisions in the first 64 discovery cases by that hash,
including decisions with no flips. The pilot is the first four discovery cases, patching
the first two; it checks execution, not scientific thresholds.

Do not run the test in this collection. Candidate components and additional interventions
will be selected after discovery; freeze them before dev verification and test confirmation.
The original test was examined in earlier group studies and is not described as unseen.
State-swap donors have identical question specifications, are different cases, and never
cross the discovery/dev/test boundary.

## Treatments and outcomes

For k <= 4 use all k! permutations. For k > 4 use cyclic rotations and reversed rotations,
deduplicated; every alternative occupies every slot equally often. This is not an exhaustive
permutation study for k > 4. Keep IDs and descriptions unchanged in the primary treatment.
The package's option_order contains integer canonical indices, not IDs. It unpermutes
probabilities on output, but raw logits and marker activations remain in slot order;
explicitly align those by semantic identity.

Collect four separate conditions: original, empty state, swapped state, renamed IDs opt_1..k.
Renaming retains the canonical semantic indices and is not confounded with presentation slot.
Renaming/empty/swapped controls change content or tokenization and are diagnostic contrasts,
not pure interventions on ordinal position.

Save raw logits, official probabilities/choice, identity mapping, marker positions, token
spans and lengths. Stop if state/options are truncated or alternatives collapse. Assert
that reordering preserves token spans by alternative, instruction tokens, and state tokens.
Use 1024/256 budgets, single-question FP32, no TF32, eval/inference mode, seed 0.

## Causal localization without SAE

For every selected decision use the fixed pair canonical order and rotation by one,
in both directions. Do not select the pair by maximal observed effect.
Sweep all encoder layers, final normalization, and both head layers. Transfer marker
vectors from donor to recipient aligned by option identity. Interventions per layer:

- all markers;
- one marker, identity selected by a fixed hash independent of outcomes;
- wrong-identity donor marker vectors (cyclic remapping), a disruptive comparison;
- four isotropic random-delta controls, matching the true delta norm per marker;
- same-order self patch (must reproduce the recipient within 1e-5 logit).

Final-head all-marker transfer must recover donor logits within 1e-4: this is a positive
implementation control and is deliberately trivial, not evidence about the origin of bias.
Official unhooked and hooked probabilities must agree within 0.00011, with identical
official selected IDs. Disable the MHA fused fast path uniformly so hooks do not change
computation paths between reference and intervention.

Primary saved effect: project the centered patched-minus-recipient logit vector onto
the centered donor-minus-recipient vector. Save numerator, squared denominator, ratio,
absolute change norm, orthogonal norm, distance to donor, and all raw score vectors.
Centering removes arbitrary shared logit offsets. Ratio is undefined when denominator
<= 1e-8 (numerical guard only); retain these cases for absolute effects. Ratios can be
negative or >1 and are not percentages of causal mediation. Establish any larger
scientific effect floor using discovery before confirmation.

## Later analysis (not automatic model selection)

Report flip rates, aligned probability variation, centered logit variation, and accuracy
across orders, both overall and by workflow/option count. Aggregate permutations and
questions within case; bootstrap cases, not permutations. Report effect-size distributions,
denominators/exclusions, with/without flips. Four random directions are controls, not
enough to claim precise empirical p-values. Control multiple comparisons for layer scans
or designate replicated candidates before confirmation.

Patching locates transmission; it does not prove where the effect first originated.
Follow up with attention/MLP/path interventions in localized components. Distinguish
ordinal slot from absolute token position, distances, neighbourhood and lexical IDs using
additional controlled-length/neighbour experiments. These are follow-ups, not implemented
claims of this first collection. Low sensitivity caused by uniform outputs is not success:
track discrimination, entropy and accuracy whenever proposing a mitigation.

## Execution, storage and resume

GPU inference only via Sauron jobs. Reuse uv.lock and uv_env.sh; preserve HF_HOME/TORCH_HOME.
Use a separate project ~/workspace/sae-position-mechanism. One GPU in the 89-GB class,
4 CPUs and 16g RAM initially; pilot validates the envelope. Actual elapsed time and memory
must guide subsequent sizing; no measured full-runtime estimate is claimed yet.

`position-pilot.yaml` is a 30-minute standalone pilot. `position-discovery.yaml` is an
8-hour bounded pipeline that repeats the pilot, checks COMPLETE, then runs discovery only.
This allows handing off a queued job without a long-lived external monitor. A failed
pilot prevents the collection from starting. The pilot subprocess has a 15-minute bound,
including any checkpoint loading not already cached. Both jobs use the same script.

Atomic per-decision JSON outputs permit safe resume; manifests reject changed code,
arguments, model/data versions or case lists. A repeated submission can resume the same
outputs. COMPLETE is written only after every selected decision file exists. Do not edit
source files while a job using this workspace is queued/running. Results and venv are not
committed. Keep an external copy of results after the run; Sauron provides no backup.

## Commands

```bash
cd ~/workspace/sae-position-mechanism
sauron submit -c scripts/sauron/jobs/position-pilot.yaml scripts/sauron/position.sauron
# Bounded discovery pipeline, with pilot gate:
sauron submit -c scripts/sauron/jobs/position-discovery.yaml scripts/sauron/position.sauron
sauron logs JOB_ID -f
sauron status JOB_ID
```
