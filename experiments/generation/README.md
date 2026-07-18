# Grounded-generation evaluation

`benchmark-spec.json` freezes the minimum case mix, objective metrics, blind
review fields and hard-failure gates. `benchmark_results` is empty: no approved
live model, historian-authored generation set, or human grades exist.

The executable contract is `wic_history.generation_benchmark`. A dataset case
contains an exact `ScenarioContextBundle`, task, bounded history, gold
answerability and expected support-region IDs. Only the context/task/history
are passed to the provider. The gold fields remain outside the prompt.

An eligible dataset requires at least 30 cases, two historian authors, five
unanswerable cases, five cases for each generation task, all seven research
question categories, unique task/context/history inputs, complete scan provenance, and expected support that is
either directly reviewed or historian-selected gold. Eligibility is recomputed
from the case evidence when a dataset is loaded; its Boolean cannot be flipped
manually.

Run one pinned provider over the frozen contexts:

```bash
uv run wic-generation-benchmark run \
  --dataset artifacts/generation-benchmark/dataset-v1.json \
  --code-revision 0000000000000000000000000000000000000000 \
  --output artifacts/generation-benchmark/model-a.predictions.json
```

Replace the zero revision with the exact 40-character project commit. The
provider uses the environment contract in
[`docs/generation-operations.md`](../../docs/generation-operations.md). An
ineligible technical fixture is refused unless
`--allow-ineligible-technical-run` is explicit; that flag does not make its
scores scientific.

Calculate structural/citation metrics:

```bash
uv run wic-generation-benchmark score-objective \
  --dataset artifacts/generation-benchmark/dataset-v1.json \
  --predictions artifacts/generation-benchmark/model-a.predictions.json \
  --output artifacts/generation-benchmark/model-a.objective.json
```

These metrics establish whether an answer passed the evidence validator and
cited expected regions. They do not establish entailment, completeness,
historical safety, usefulness, or whether an answer to a negative question was
appropriate. Token totals and estimated cost are reported only when the
provider supplies complete usage metadata; missing usage is not imputed.

Export a model-blind packet:

```bash
uv run wic-generation-benchmark export-blind \
  --dataset artifacts/generation-benchmark/dataset-v1.json \
  --predictions artifacts/generation-benchmark/model-a.predictions.json \
  --output artifacts/generation-benchmark/blind-batch-a.json
```

The packet removes model/provider/configuration, internal case ID, question
category and gold answerability, then orders cases by artifact-specific blind
hash. It retains a pseudonymous hash of the exact task/context/history input so
separately graded model reports can later be paired without revealing the
internal case ID or gold fields. Two independent historians score every blind case on five 1–5 scales,
unsupported claims and pass/fail/needs-discussion. A third person adjudicates
every case and must be distinct from both reviewers. Review JSON must validate
as `HumanGenerationGradeSet` and cite the exact dataset, prediction-artifact
and blinded-packet SHA-256 values.

Aggregate adjudicated grades and reviewer agreement:

```bash
uv run wic-generation-benchmark score-human \
  --packet artifacts/generation-benchmark/blind-batch-a.json \
  --grades artifacts/generation-benchmark/blind-batch-a.grades.json \
  --output artifacts/generation-benchmark/model-a.human.json
```

Models must ultimately be compared on paired cases with uncertainty intervals
and absolute safety/quality review. A higher mean alone is not a selection.

After both model-blind batches have been independently reviewed and
adjudicated, compare their reports on identical pseudonymous inputs:

```bash
uv run wic-generation-benchmark compare-human \
  --report-a artifacts/generation-benchmark/model-a.human.json \
  --label-a candidate-a \
  --report-b artifacts/generation-benchmark/model-b.human.json \
  --label-b candidate-b \
  --bootstrap-seed 17 --bootstrap-resamples 5000 \
  --output artifacts/generation-benchmark/a-vs-b.paired.json
```

The reported 95% interval is a deterministic case-level bootstrap of the
adjudicated five-scale mean difference. It does not override the absolute
unsupported-claim and historical-safety gates.
