# Regenerating the thesis evaluation tables

From the repository root:

```sh
python3 evaluation/regenerate_tables.py
```

The exporter uses only the Python standard library and retained JSON/CSV/log
artifacts. It does not run the SHAPE pipeline, call language models, query Qdrant,
or access the FHIR server. It can also be invoked by absolute path from another
directory.

## Reference and counting conventions

- P07/A3 is `ActivityDefinition/1058` (Request Clinical Observation).
- Activity grounding and confidence count one observation per evaluable reference
  action. A generated action matched to several reference actions contributes to
  each correspondence; confidence records must not be keyed only by generated ID.
- Parameter identification retains the notebook's existing conditional subset and
  TP/FP/FN matching convention. This correction adds one FP for Mixtral-8x7B and
  one FP for Llama; no expected parameter is added to P07/A3.
- Ontology grounding is checked against all saved candidate/selection records.
  The 90 configuration counts remain unchanged. In term isolation, k counts term
  records, not distinct concepts. Accepted alternatives come from the reference
  value/possible_values annotations.
- Final encoding uses Exact/Different/Missing for the 26 expected temporal
  constraints and 77 expected parameters. Ontology values compare system/code
  against accepted alternatives, ValueSet values compare the selected code,
  and other parameter values compare decoded contents. Display and confidence
  annotations do not affect coded-value agreement.

## Outputs

The existing RQ2 activity-grounding, confidence and parameter-identification CSVs
are rebuilt in `evaluation/rq2/`. The ontology summary and 18 configuration CSVs
are re-exported after checking N/H/C against the raw saved records.

`evaluation/rq3/e2e_analysis.csv` now exports final-encoding counts. Its columns
are `llm`, `temporal_exact`, `temporal_different`, `temporal_missing`,
`parameter_exact`, `parameter_different`, and `parameter_missing`.
This intentionally replaces the legacy aggregate TP/FP/FN/F1 schema, which did
not implement the comparison criteria used in the thesis.

`evaluation/details/` contains:

- MI07: intermediate temporal cases with matching warning file/line references,
  counts per model, and reconciliation with missing reference constraints.
- MI12: every reference-action/candidate/selection case and aggregate counts.
- MI15: all 77 Llama reference parameters and its 26 Different cases; differences
  in ontology codes are not automatically treated as semantic equivalence.
- MI14: the complete 90 configurations, 2,286 parameter/configuration cases and
  an appendix-ready LaTeX longtable.
- `thesis_table_rows.tex`: data rows for the existing thesis tables; headers and
  captions remain the responsibility of the thesis source.
- `regeneration_manifest.json`: SHA-256 fingerprints of source annotations,
  generated outputs and logs, verified unchanged during regeneration.

Construction warnings do not preserve detailed ValidationError objects. Temporal
omissions are attributed only when the warning payload matches the retained
intermediate value; repeated warnings and the two complete log runs do not count
as additional constraints. Offsets without an incoming relationship are reported
separately.

The notebook uses the same exporter for RQ3, preserves one confidence record per
reference action, and chooses the ontology-selection denominator from k rather
than the representation-strategy name. Its four affected table displays are
refreshed from the regenerated CSVs without running inference.
