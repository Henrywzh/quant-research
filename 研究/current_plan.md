# Work Procedure (Next Phase): A-Share Large/Small Index Rotation → HSCI Multifactor

## Objective
Reorder execution to prioritize **A-share large/small index rotation** first, then return to **HSCI multifactor**.  
Rationale: HSCI single-factor research is already completed recently; momentum is highest if we convert the A-share prototype into a robust, explainable v2 before context-switching back.

---

## Track 1 (Primary): A-Share Large/Small Index Rotation — v2 Delivery

### Scope (freeze)
- Do **not** add new indices/pairs beyond your current tested set unless required for validation.
- Focus on converting prototype → **regime-aware, explainable strategy**.

### Deliverables (end of Track 1)
1. **v2 strategy**: rotation + regime gating/scaling (risk-on/off)
2. **Evidence package**:
   - equity curve vs benchmark
   - drawdown chart
   - regime split performance (bull / bear / sideways)
   - a short “why it works” narrative backed by 1–2 diagnostic charts
3. **Reproducibility**:
   - single config file for parameters
   - single notebook/script to run and export artifacts

### Work Steps (in order)

#### Step 1 — Baseline Lock (no changes)
- Fix the exact baseline:
  - universe: the indices you already tested
  - rotation rule: current selection logic
  - rebalance frequency
  - transaction cost assumption
- Export baseline artifacts:
  - equity curve
  - drawdown
  - yearly returns / regime split (even if rough)
- **Acceptance (DoD)**: baseline run is one-command reproducible and produces artifacts to `/artifacts/ashare_rotation/v1/`.

#### Step 2 — Identify “Bull-Market Dependence” Mechanism
- Quantify concentration of returns:
  - contribution by market regime (bull/bear/sideways)
  - worst drawdowns and when they occur
- Produce two diagnostics:
  - rolling volatility (or correlation) of target indices
  - dispersion proxy (optional) to support rotation rationale
- **Acceptance (DoD)**: you can point to 1–2 clear regime conditions where the strategy works/fails.

#### Step 3 — Add Minimal Regime Overlay (v2)
Choose **one** regime proxy (keep it simple):
- Example candidates (pick one):
  - market volatility level (e.g., rolling vol of CSI300)
  - correlation/risk concentration proxy (rolling corr between large/small indices)
  - trend filter on a broad market index
- Implement a simple overlay:
  - **Risk-ON**: execute rotation normally
  - **Risk-OFF**: reduce exposure / switch to defensive index / cash proxy
- **Acceptance (DoD)**: v2 reduces drawdown or improves stability in non-bull periods without killing bull upside completely.

#### Step 4 — Robustness Mini-Suite (minimal but mandatory)
- Holdout validation:
  - at least one walk-forward or pre/post split
- Parameter sanity:
  - rebalance freq sensitivity (e.g., 15/20/30 days)
  - overlay threshold sensitivity (small grid)
- Cost sensitivity:
  - low/medium/high cost scenarios
- **Acceptance (DoD)**: conclusions unchanged under small perturbations; no single parameter “knife-edge.”

#### Step 5 — Write v2 Summary (1–2 pages)
- What the strategy is (rotation + regime overlay)
- Where it works (bull) and what v2 improved (non-bull)
- Remaining failure cases (explicit)
- **Acceptance (DoD)**: publishable internal note + artifact links.

---

## Track 2 (Secondary): CICC Quant Handbook Reading — Only as Needed During Track 1
### Rule
Reading is allowed only if it directly supports a concrete implementation step in Track 1.

### Output requirement
Every reading session must produce:
- 5–10 bullet “executable takeaways”
- one proposed test or diagnostic to run next

---

## Track 3 (After Track 1): Return to HSCI Multifactor — v2 Combination

### Entry condition (when to switch back)
Switch back to HSCI only when A-share rotation v2 meets:
- artifacts exported
- regime overlay implemented
- robustness mini-suite done
- short write-up completed

### HSCI v2 scope (controlled)
- Combine existing signals only (Trend / Vol / Liquidity)
- Two combinations only:
  1) rank-average
  2) IC-weight
- Two variants:
  - raw combination (exposures allowed)
  - neutralised combination (residual alpha diagnostic)
- Liquidity factor likely as constraint/risk input if neutralised alpha remains weak

### HSCI v2 deliverables
- equity curve, drawdown, table summary vs single factors
- short conclusion: “what improves from combining?”

---

## Weekly Cadence (recommended)
- **2 build sessions** for Track 1 (deep work, code + experiments)
- **1 validation session** (robustness + diagnostics)
- **1 writing session** (update summary note + decisions)
- Optional: **1 reading session** (only if it unlocks Track 1)

---

## Definition of Done (overall)
- A-share rotation v2: reproducible + explainable + regime-aware
- HSCI multifactor v2: reproducible combined factor baseline ready for further research
