# Semantic Parser / Planner v4

## Ownership boundary

| Stage | Input | Owns | Must not own |
|---|---|---|---|
| Parser | User text only | Split user goals, normalize each goal to the closed semantic ontology, preserve explicit order and uncertainty | Primitive names, mechanisms, masks, geometry, budgets, feasibility |
| Program Planner | Structured semantic request + four-axis case identity | Produce one step per user intent and a bounded primitive candidate set | Pixels, coordinates, tool parameters, unsupported clinical inference |
| Current-state preflight | Current tissue/nuclei masks, typed auxiliaries and skill contracts | Remove infeasible primitive/mechanism pairs and compile certified candidates | Reinterpret the user's biological intent |
| Deterministic evaluator | Compiled candidates and hard-gate reports | Accept only hard-gate-passing candidates and break ties by budget error | LLM judgment or a claim of visual/pathology realism |

For an ordered request, execution is transactional:

1. execute one step against the current masks;
2. require every deterministic hard gate to pass;
3. commit the selected tissue and nuclei masks;
4. rebuild the scene from the committed masks;
5. run the next step;
6. stop immediately on failure, review, or clarification.

## Parser system prompt

The runtime source of truth is `SEMANTIC_REQUEST_SYSTEM_PROMPT` in
`phase3_joint_edit_refine/semantic_request.py`:

```text
You are the instruction-only Semantic Request Parser for a pathology mask editor.

Your sole responsibility is to translate the user's Chinese or English language into the caller's closed semantic ontology. A request may contain one intent or several intents. Split only user-stated biological intentions; preserve explicit order words such as first, then, after, and finally. An implementation detail needed to realize one biological change is not a second user intent.

For every intent, extract the biological target, requested operation, clinical context, spatial scope, morphology, explicitly named cell class, strength, literal source span, and uncertainty. Use `direct_edit` for an explicit requested change and `clinical_trajectory` for progression, regression, treatment-response, residual-disease, or recurrence language. Normalize paraphrases into the supplied enum values.

Never select, name, rank, or suggest an edit primitive or pathology mechanism. Never inspect or infer image morphology, annotation labels, coordinates, masks, connected components, area, cell count, density, tool parameters, or feasibility. Do not invent an order that the user did not state. Preserve negation and do not convert post-treatment context into improvement unless the user states response or regression. If the biological direction is genuinely missing, use `unspecified` and record the uncertainty instead of guessing.

Return only JSON conforming to the supplied strict schema.
```

The API Parser is called with `image_paths=()`. Its schema recursively rejects
`primitive_id`, `primitive_hypotheses`, and `mechanism_id`.

## Examples

### One explicit edit

User:

```text
让肿瘤边界连续、黏附性地向外扩张。
```

Parser output (abridged):

```json
{
  "intents": [{
    "intent_id": "intent-001",
    "intent_type": "direct_edit",
    "target": "tumor_extent",
    "operation": "increase",
    "polarity": "affirmed",
    "clinical_context": "none",
    "spatial_scope": "boundary",
    "morphology": "cohesive",
    "cell_class": null
  }],
  "relations": []
}
```

Breast/PANDA Planner output:

```json
{
  "step_id": "step-001",
  "selected_primitive_id": "cohesive-boundary-expansion-v1",
  "status": "planned"
}
```

### Ordered multi-intent treatment edit

User:

```text
先在治疗后缩小肿瘤，然后减少残余肿瘤细胞。
```

Parser output (abridged):

```json
{
  "intents": [
    {"intent_id": "intent-001", "target": "tumor_extent", "operation": "decrease", "clinical_context": "post_treatment"},
    {"intent_id": "intent-002", "target": "neoplastic_cell_population", "operation": "decrease", "clinical_context": "residual_disease", "cell_class": "neoplastic"}
  ],
  "relations": [
    {"before_intent_id": "intent-001", "after_intent_id": "intent-002", "relation_type": "explicit_sequence"}
  ]
}
```

Breast Planner output:

```json
{
  "steps": [
    {"step_id": "step-001", "selected_primitive_id": "invasive-tumor-footprint-decrease-v1", "depends_on": []},
    {"step_id": "step-002", "selected_primitive_id": "neoplastic-cell-abundance-decrease-v1", "depends_on": ["step-001"]}
  ]
}
```

Step 2 receives the committed masks and digests from step 1, not the original
masks.

### Morphology left genuinely unresolved

User:

```text
增加浸润。
```

Parser emits `target=invasion_pattern`, `operation=increase`, and
`morphology=unspecified`. The Planner keeps the organ-compatible invasion
primitives as candidates. Current-mask preflight removes impossible choices. If
two materially different primitive meanings still survive, execution returns a
digest-bound clarification request instead of guessing.

### Contradictory unordered request

User:

```text
增加肿瘤面积，并且减少肿瘤面积。
```

The Parser preserves two unordered intents. The Program Planner marks their
opposing operations as a conflict and returns `clarification_required`. The same
two operations are allowed when the user explicitly states their order.

## Evaluator boundary

`DeterministicMaskProgramEvaluator` is code, not an LLM. It accepts a candidate
only when its required gate report passes, then ranks surviving candidates by
distance to the system-owned count/area budget. It also verifies the digest
chain between sequential steps. Its approval scope is only
`mask-contract execution`; `visual_pathology_approval` is always false and must
remain a separate downstream visual audit.
