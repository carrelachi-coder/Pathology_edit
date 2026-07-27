# Pathology World-Model Narrative

## 1. Narrative objective

The manuscript should not present the system as a collection of image-editing
modules. Its central scientific object is an **intervention-conditioned
pathology world model**: given a real histopathology observation and a
directional biological intervention, the system constructs, realizes and
verifies a counterfactual morphological state transition.

The preferred one-sentence description is:

> Given a real histopathology state and a specified biological intervention,
> the model constructs a plausible next morphological state, propagates the
> transition from tissue architecture to cellular organization, renders the
> resulting state as H&E and verifies both the intended change and the
> structures that should remain invariant.

Recommended manuscript positioning:

> An agentic pathology world model for controllable morphological evolution.

The qualifying terms are important. The system is:

- intervention-conditioned rather than an unconditional generator;
- morphological rather than molecular;
- counterfactual rather than a patient-specific forecast;
- single-step in the formal benchmark rather than a validated long-horizon
  simulator;
- multiscale because it represents both tissue and nuclei states;
- closed-loop because generated observations are re-perceived and checked.

## 2. Formal model

Let the explicit pathological state be

```text
Z_t = (T_t, N_t),
```

where `T_t` is the tissue state and `N_t` is the nuclei state. Let `X_t` be the
observable H&E image and let the intervention be

```text
a = (biological process, direction, dose, spatial scope).
```

The system implements

```text
Z_(t+1) = F(Z_t, a)
X_(t+1) = G(Z_(t+1) | X_t)
V(X_(t+1), Z_(t+1), Z_t) -> accept, alternate realization, or review
```

`F` is the structured transition model. It translates the intervention into a
target tissue state and propagates that change into a compatible nuclei state.
`G` is the observation model. It realizes the same target state through local
synthesis or reference-guided global reconstruction. `V` is the closed-loop
perception and consistency model. It tests whether the intended transition is
observable in the generated H&E and whether prespecified invariants are
retained.

The notation `t -> t+1` denotes a counterfactual state transition or
pseudo-time step. It must not be described as observed chronological disease
progression.

## 3. Central scientific contrast

Conventional generative pathology asks:

> Can a model synthesize a realistic pathology image?

The proposed world model asks:

> Given this specimen and this biological intervention, what morphologically
> compatible state can follow while unrelated properties remain invariant?

The distinction rests on four explicit elements that ordinary generation does
not jointly provide:

1. a defined current state;
2. a biologically interpretable action;
3. an explicit target state and invariant set;
4. a perception-based test of whether the transition was realized.

Realism remains necessary, but it is not sufficient evidence of a valid state
transition.

## 4. Terminology map

| Implementation term | Manuscript term | Scientific role |
|---|---|---|
| Edit instruction | Biological intervention | Specifies which biological process should change |
| Edit primitive | Biological transition operator | Defines an allowed direction of morphological evolution |
| Edit direction | Transition direction | Increase, decrease or phenotype conversion |
| Edit strength | Transition dose | Controls the magnitude of a counterfactual step; not elapsed time |
| Reference tissue/nuclei masks | Current structural state | Explicit multiscale state before intervention |
| Target tissue mask | Target tissue state | Macroscopic result of the transition |
| Nuclei-layout construction | Cross-scale state propagation | Makes the cellular state compatible with the new tissue state |
| Binary change mask | Transition support | Region in which the state is permitted to change |
| Unchanged region | State invariants | Features that should remain stable through the transition |
| Local inpainting | Local observation model | Realizes compact state changes using surrounding image context |
| Cross-v1 plus reference refinement | Global observation model | Realizes distributed state changes and restores reference-compatible appearance |
| Re-segmentation | State re-observation | Recovers tissue and nuclei evidence from the generated H&E |
| Verification | Closed-loop transition verification | Tests target attainment and invariant preservation |
| Alternate route | Alternative state realization | Re-renders the same target state without changing the intervention |
| Utility analysis | Representation-space trajectory validation | Tests whether realized transitions retain dose and primitive semantics |

Use `editing` when referring to the user interaction or implementation
interface. Use `state transition`, `controlled morphological evolution` or
`biological transition` for the scientific interpretation.

## 5. Six-part Results logic

### Result 1: The system is an intervention-conditioned pathology world model

Introduce the complete state-action-transition-observation-verification loop.
The input image supplies the initial observation. Tissue and nuclei conditions
form the explicit structural state. Natural language supplies the biological
action. The model plans a target state, realizes it using one of two observation
routes and re-perceives the output before acceptance.

Main claim:

> The system makes both the intended transition and the required invariants
> explicit before image synthesis.

### Result 2: Language defines directional, localized and dose-controlled
transition operators

The mask benchmark is not merely a prompt-following benchmark. It validates the
macroscopic transition model `F_T`: the requested process, direction, location
and dose are converted into a legal next tissue state without unrelated label
changes.

The strength series should be interpreted as a counterfactual dose trajectory,
not as patient time. Validator-guided replanning demonstrates that invalid
state proposals can be rejected and repaired before rendering.

Main claim:

> Natural language can specify executable biological transition operators with
> explicit direction, support, dose and invariants.

### Result 3: Tissue transitions propagate into compatible cellular states

This section explains the micro-scale transition `F_N`. Counts and type quotas
are determined from the current patch and dataset--tissue statistics.
CellDistNet adds a tissue- and boundary-aware spatial prior, and geometric
sampling realizes a safe nuclei state.

Do not claim that CellDistNet discovers a complete cellular point process. The
scientific role is narrower and more defensible:

> It propagates an accepted tissue transition into an engineering-valid
> cellular state while preserving retained nuclei.

### Result 4: Scale-adaptive observation models realize and verify the same
target state

Local and global synthesis are not different biological mechanisms. They are
two observation models for realizing the same structural endpoint. Routing
selects the appropriate realization scale; it does not change the biological
action or target state.

The verifier functions as state re-observation. It tests whether the rendered
H&E expresses the planned tissue and nuclei state and whether route-specific
invariants are preserved. Alternate-route recovery is constrained re-rendering
of the same state, not model-driven alteration of the endpoint.

Main claim:

> A target morphological state can be realized through scale-matched
> observation models and subjected to closed-loop perceptual constraints.

### Result 5: The complete world model produces faithful and realistic
counterfactual states

Comparator fairness must follow input contracts. Models without target
geometry cannot be ranked for strict state-transition fidelity. Patho-KID and
expert review address observational realism, while re-segmentation addresses
target attainment and invariant preservation.

Main claim:

> The complete system should be evaluated by transition validity, preservation
> and realism rather than visual realism alone.

This claim remains prospective until the generation-consistency, Patho-KID and
expert-review benchmarks are frozen.

### Result 6: Controlled biological transitions form dose-extending and
primitive-specific representation trajectories

This is the mechanistic culmination of the paper, not a generic downstream
utility benchmark.

The logic is:

1. Within each primitive, Moderate-to-Significant continuation tests whether
   transition dose extends an established state displacement.
2. Agreement between local and global observation models tests whether the
   displacement is reproducible beyond one rendering implementation.
3. Agreement between UNI-2h and CONCH tests whether the result is not confined
   to one pathology representation model.
4. The shared U1/U2 component captures common tissue remodelling.
5. Positive own-axis margins identify primitive-specific residual directions.
6. High-mask-overlap sensitivity shows that primitive separation is not
   explained only by spatially disjoint transition support.

Supported conclusion:

> Controlled tumour expansion and stromal immune infiltration form
> dose-extending representation trajectories with a shared tissue-remodelling
> component and primitive-specific residual directions.

Do not relabel this as proof that the system has learned a universal biological
law.

## 6. Introduction logic

The Introduction should follow this sequence:

1. Histopathology slides are static observations of evolving multicellular
   systems. Real paired observations that differ in only one biological process
   are generally unavailable.
2. Existing text-to-image and editing models operate mainly in observation
   space. They can produce plausible images without defining the underlying
   state transition or its invariants.
3. World models represent the state of an environment and predict the outcome
   of actions. In pathology, the relevant action is a biological intervention
   and the relevant next state is a counterfactual tissue-and-cellular
   morphology.
4. Histopathology requires an explicit multiscale state because tissue
   architecture, nuclei composition and appearance are coupled.
5. Introduce the proposed agentic world model and its transition,
   observation and verification components.
6. State the benchmark questions in the same order as the Results.

Relevant framing references:

- Ha and Schmidhuber, *World Models* (2018).
- Hafner et al., *Mastering diverse control tasks through world models*,
  *Nature* (2025).
- Palma et al., *Predicting cell morphological responses to perturbations
  using generative modeling*, *Nature Communications* (2025).

The first two establish the state/action/predicted-outcome world-model concept.
The third supports the broader idea that counterfactual image generation can
represent morphological responses to biological perturbations.

## 7. Figure logic

### Figure 1

Panel A should be read as a world-model loop:

```text
initial observation
    -> explicit multiscale state
    -> biological intervention
    -> target state transition
    -> observation model
    -> re-observation and verification
```

Panel B should evaluate:

1. action interpretation;
2. target-state construction;
3. observation fidelity and realism;
4. representation-space trajectory semantics.

### Language-editing figure

Describe it as the construction and validation of a tissue-state transition,
not simply mask editing.

### Nuclei and synthesis figure

Describe CellDistNet as cross-scale state propagation. Describe FLUX--ControlNet
and Cross-v1/pix2pix as local and global observation models.

### Representation trajectory figure

This figure is evidence that generated counterfactual states are organized by
transition dose and biological process. Local/global agreement supports a
renderer-shared semantic component; it does not imply renderer invariance.

## 8. Claim boundaries

### Supported

- intervention-conditioned morphological state transition;
- counterfactual single-step evolution;
- explicit tissue and nuclei states;
- state invariants and closed-loop verification;
- dose-dependent representation response;
- transition directions reproduced across two observation models and two
  pathology encoders;
- shared tissue-remodelling and primitive-specific representation components.

### Not supported

- prediction of a patient's future slide;
- natural-history or treatment-response forecasting;
- temporal calibration of Moderate and Significant strengths;
- a molecular or causal disease mechanism;
- a universal biological simulator;
- validated long-horizon rollout;
- a pathology digital twin;
- improved clinical prediction or downstream task performance.

If the paper uses the term `evolution`, pair it with `controlled`,
`counterfactual` or `morphological`. Avoid unqualified `disease evolution`.

## 9. Recommended title and central claim

Preferred title:

> **An agentic pathology world model for controllable morphological evolution**

Alternative, more conservative title:

> **Intervention-conditioned modelling of multiscale morphological transitions
> in histopathology**

Preferred central claim:

> We formulate histopathology editing as intervention-conditioned state
> transition and build an agentic world model that plans, realizes and verifies
> multiscale morphological evolution.

Preferred concluding statement:

> Controllable generative pathology should be evaluated not only by whether an
> image appears real, but by whether a specified biological transition is
> realized, unrelated state variables remain invariant and the resulting
> counterfactual states retain the semantics of transition identity and dose.
