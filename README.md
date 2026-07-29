# Hybrid Fuzzy–LLM Decision Support System

Research prototype for lifecycle-oriented supply-chain strategy using a
deterministic zero-order Sugeno fuzzy inference system and an optional LLM
language-rendering layer.

## Methodological boundary

The fuzzy engine is the only decision authority. It produces the scores,
rankings, membership degrees, activated rules, and numerical trace. The LLM
cannot calculate or modify those results; it can only render the frozen
evidence in natural language. The application therefore remains operational
and reproducible without an API key.

The v2.1 rule base and input-specific membership breakpoints are explicitly marked as
**provisional**. The constructs and directional associations are literature
informed, while the numerical calibration requires structured expert
elicitation and sensitivity validation before industrial-validity claims are
made.

## Repository contents

- `app.py`: Streamlit interface and optional OpenRouter narrative layer.
- `llm_grounding.py`: API-independent semantic grounding validator.
- `fuzzy_engine.py`: API-independent Sugeno inference engine and trace.
- `decision_model.py`: baselines, 5S associations, lifecycle associations, and scoring.
- `benchmarks.json`: contextual benchmark ranges shown in the interface.
- `run_replication.py`: command-line reproduction without Streamlit or an LLM.
- `config/fuzzy_model.json`: complete memberships, consequents, rule weights
  and confidences, applicability convention, baseline matrices, associations,
  epsilon, and 27 explicit rules.
- `config/example_scenario.json`: frozen example inputs.
- `expected_results/example_replication_output.json`: reference output.
- `tests/test_fuzzy_engine.py`: deterministic coverage, range, and repeatability tests.
- `METHODS_AND_REPLICATION.md`: reporting and replication specification.
- `CALIBRATION_PROTOCOL.md`: prospective expert elicitation, calibration, and
  holdout-validation protocol.
- `KPI_MAPPING.md`: complete 30-KPI cross-configuration baseline matrix and
  applicability policy.

## Local installation

```bash
python -m venv .venv
```

Activate the environment and install dependencies:

```bash
pip install -r requirements.txt
```

Run the app:

```bash
streamlit run app.py
```

No credential is needed for the deterministic DSS. When an OpenRouter key is
available, clicking `Analyze` generates the LLM narrative automatically. If the
key is absent, the service fails, or the model returns reasoning/prompt text,
the app automatically displays the deterministic explanation. To enable
OpenRouter locally, copy `.streamlit/secrets.toml.example` to
`.streamlit/secrets.toml` and add the key. Never commit that file.

## Exact computational replication

From the repository root, run:

```bash
python run_replication.py \
  --scenario config/example_scenario.json \
  --output replication_output.json
```

Then run the tests:

```bash
python -m unittest discover -s tests -v
```

The generated fuzzy scores, applicability decisions, and traces are exactly reproducible for the same
code, configuration, inputs, and Python dependencies. LLM prose is not part of
the numerical replication target.

## Optional OpenRouter layer

The interface uses:

```python
model="openrouter/free"
```

The underlying free model may vary between requests. The app records the model
returned by the router, temperature, token limit, and prompt hash. This router
is appropriate for accessibility and demonstrations, but a controlled paper
experiment should either pin one model or archive the evaluated outputs.

Every optional narrative is checked against its supplied payload. The
validator rejects reasoning leakage, unsupported numbers and rule identifiers,
item–score or item–rule mismatches, ranking inversions, and unsupported
normative, risk, causal, performance, or outcome claims. Rejected sections are
replaced by deterministic trace explanations. The run JSON records every
attempt, actual routed model, rejection reason, and fallback section.

## Benchmark caution

`benchmarks.json` contains broad consultancy-attributed ranges supplied for the
prototype interface. These values are contextual references, not plant-level
validation data and not evidence that the fuzzy mappings have been calibrated.
Before publication, replace abbreviated source labels with complete,
page-verifiable citations or clearly label the ranges as illustrative.

## Reproducibility statement

Scientific reproducibility applies to the deterministic fuzzy scores, rule
activations, rankings, and robustness results. Exact textual identity is not
expected from the optional LLM renderer. A deterministic explanation remains
available when the external service is unavailable.
