# Self vs. Therapist: Multi-Agent Personality-Disorder Simulation

A multi-agent "Self" — five OCEAN trait agents plus a consensus step — is interviewed by an LLM "Therapist" agent across DSM-5-TR personality-disorder criteria. Each simulated session is scored against a held-out ground-truth diagnosis, validated against an independent rater baseline on textbook vignettes, and averaged across independent rounds to control for LLM sampling noise.

See `PROJECT_DESCRIPTION.md` for the full write-up, or the sections below for a practical map of the repo.

---

## Why

Single-model role-play tends to collapse into a flat, caricatured "persona" — one voice pretending to have five different traits. This project instead builds "Self" out of five separate trait-specialist agents (Openness, Conscientiousness, Extraversion, Agreeableness, Neuroticism) whose individual reactions are merged by a dedicated consensus step, aiming for more internally varied, less caricatured behavior across a full clinical interview.

## Architecture

```
Therapist agent  <──dialogue──>  Self (outer loop)
                                     │
                         ┌───────────┴───────────┐
                         │   interpret question   │
                         └───────────┬───────────┘
                     ┌───────────────┼───────────────┐
                 O agent   C agent   E agent   A agent   N agent
                     └───────────────┬───────────────┘
                              consensus agent
                                     │
                              Self's reply
```

- **Therapist ↔ Self loop** — a LangGraph state machine (`StateGraph`) drives a multi-turn clinical interview (6–8 turns per case).
- **Self's inner subsystem** — each turn: the question is interpreted, all five OCEAN trait agents react independently, and a consensus agent merges those five reactions into Self's single reply. Every trait agent's raw reaction is preserved in `ocean_log` before merging, so the formation of the final reply can be audited turn-by-turn after the fact.
- **Report generation** — after the interview, a separate report-writer model reads the full transcript and assigns a 0–100% likelihood to each of the ten DSM-5-TR personality disorders.

## Case pipeline

Raw clinical vignettes (e.g. textbook case studies) are run through an extraction step that produces structured case records — persona background, a list of memory/behavior snippets, and a held-out ground-truth diagnosis — while stripping any wording that would hand the diagnosis to the models directly. Extracted cases are cached in `case_cache.json` so re-running the pipeline doesn't re-extract from scratch.

## Evaluation framework

Because `recall@k` with a fixed `k` behaves oddly on cases with a single ground-truth diagnosis, several complementary metrics are computed per case and aggregated:

- `recall@k`, `precision@k`, `F1@k`
- confidence-weighted recall (recall weighted by the model's own assigned likelihood, not just rank)
- **R-Precision** and **MAP** — rank-sensitive metrics that don't require a fixed `k`
- **GTLS** (ground-truth likelihood score) — mean likelihood assigned to the true diagnosis/diagnoses
- **Brier score** / **log loss** — calibration of the full probability vector against the true labels
- **top-1 accuracy** — whether the single highest-probability disorder was correct

Two additional pieces separate *modeling* error from *measurement* noise:

- **Rater-validation baseline** — the same report-writer/judge setup is run once on a small set of single-turn, open-text textbook control vignettes (not the multi-agent simulation), to check whether the judging process itself can recognize disorders reliably before trusting it to judge the simulated interviews.
- **Relative Simulation Fidelity (RSF)** — compares the self-agent's per-disorder recognition against an independent therapist-baseline read of the *same* cases, expressed as a ratio, so results aren't reported as raw numbers in isolation.

Results are reported as **mean ± stdev across 5 independent rounds** rather than a single run, since each round's therapist↔self dialogue is freshly sampled and a single run is noisy.

## Repo contents

| File | What it is |
|---|---|
| `self_vs_therapist_agents_final.ipynb` | Main notebook: architecture, case pipeline, scoring, and all three experiments |
| `case_cache.json` | Extracted case records (persona, memories, ground-truth diagnoses) |
| `experiment2_checkpoint.json` | Saved results of the self-agent simulation (Experiment 2) |
| `therapist_baseline_results.json` | Saved results of the independent therapist baseline (Experiment 1) |
| `experiment2_multi_round_results.json` | Per-round results + mean/stdev for the 5-round Experiment 2 average |
| `exp2_round{1..5}.json` | Per-round checkpoints for the multi-round Experiment 2 run |
| `results_dashboard.html` | Standalone HTML dashboard visualizing self-agent vs. therapist results |
| `build_dashboard.py`, `dashboard_template.html` | Regenerate the dashboard from updated result JSONs |
| `PROJECT_DESCRIPTION.md` | Full written project description |

## Notebook structure

1. Setup — OpenRouter client, model config, `chat()` helper (with retry/backoff)
2. Self's default persona
3. Therapist & report prompts (DSM-5-TR grounded)
4. The five OCEAN trait agents
5. Self's inner subsystem: interpret → 5 trait agents → consensus
6. Outer graph: therapist ↔ self loop
7. Report generation (ensemblable) & the compiled graph
8. Load cases from file (with extraction caching)
9. Scoring: matching predicted vs. ground-truth diagnoses
10. Ensembled per-case evaluation
11. Aggregate accuracy: recall@k, R-Precision, MAP
12. Rater-validation baseline: textbook control vignettes
13. Run the full study, averaged over independent rounds (`run_full_study_multi_round`)
    - **13b.** Cheaper 5-round average for Experiment 2 alone (`run_experiment2_multi_round`) — skips re-running the control baseline every round
14. **Experiment 1** — independent therapist baseline on real clinical cases
15. **Experiment 2** — self-agent simulation
16. **Experiment 3** — Relative Simulation Fidelity (self-agent vs. therapist baseline)

## Setup

```bash
pip install openai langgraph python-dotenv
```

Create a `.env` file in the project root:

```
OPENROUTER_API_KEY=sk-or-...
USE_PAID_MODELS=1   # set to 0 to fall back to a free model + reduced call volume
```

Models are routed through [OpenRouter](https://openrouter.ai). In paid mode, Self/consensus/report use `openai/gpt-oss-120b` and the Therapist/judge use `google/gemini-3.7-flash` by default (configurable at the top of the setup cell). Paid mode is required for full 4000-token structured reports and the full 5-trait-agent ensemble; free mode caps output tokens and disables the trait ensemble to stay under free-tier daily request limits.

## Running the study

Run cells 1–12 once to define everything. Then:

- **Single-round check**: run cells for Experiment 1, then Experiment 2, then Experiment 3 in order — each later cell depends on variables (`therapist_baseline_results`, `self_agent_results`) assigned by the earlier ones in the *same kernel session*. If you restart the kernel, re-run those cells before Experiment 3, or Experiment 3 will raise `NameError`.
- **Reportable 5-round average**: run cell 13b (`run_experiment2_multi_round(n_rounds=5, n_repeats=1)`). This is the number to put in a report — a single round is noisy. It's substantially cheaper than section 13's `run_full_study_multi_round`, which redundantly reruns the Experiment 1 control baseline every round.

### Cost and rate limits

The self-agent simulation is the expensive part of the pipeline: each interview turn fires the interpret step + all five trait agents + the consensus step (≈7 calls/turn) across 6–8 turns, per case, per round. A full 5-round Experiment 2 run means 5× the full 10-case dialogue cost.

- `chat()` already retries HTTP 429/500/502/503/504 with exponential backoff + jitter. HTTP 402 (no remaining OpenRouter credit) is deliberately **not** retried — it raises immediately with a message telling you to check your OpenRouter balance.
- Each round of the multi-round experiments is checkpointed independently (`exp2_round{N}.json`), so an interrupted or failed run only requires re-running the affected round, not the whole study — **but** a case that failed with an error is still recorded as "done" in its round's checkpoint and will be silently skipped on resume unless you strip error-marked entries from that round's JSON file first.

## Known limitations

- **LLM judging LLM.** Both the diagnosis-matching and report-writing steps are themselves LLMs, not human clinicians. The rater-validation baseline measures part of this risk but doesn't eliminate it.
- **Small case count.** A limited number of cases means any single mislabeled or unusual case has an outsized effect on the averaged results; generalizing beyond this case set should be done cautiously.
- **Persona name collision risk.** The default demo persona is named "Larry" — if a real case in `cases.json` also uses that name, verify `case_cache.json`'s `case_id` → background/memories mapping after extraction to make sure they weren't merged.
- **Computational cost scales linearly with rounds and repeats.** Increasing `n_rounds` or `n_repeats` directly increases API spend; see the cost/rate-limit note above.

## Suggested future work

- Add one or more human clinician raters and compare their judgments against the automated evaluation.
- Expand the case set in size and diversity to reduce sensitivity to any single case.
- Run an ablation comparing the 5-agent OCEAN "Self" against a single-model baseline to quantify the multi-agent design's effect on recognizability.
- Test different model families for each role (Self, consensus, Therapist, report-writer) to measure sensitivity to model choice.
