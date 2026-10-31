# Literature Classification Labeling Guideline

## 1. Objective
Determine whether a peer-reviewed publication genuinely satisfies a specific experimental criterion regarding a target entity, rather than merely mentioning it as a reagent, starting material, solvent, substrate, internal standard, or background literature reference.

## 2. Core Rule ("Words of Stone")
> **Qualifies only if the paper reports making, producing, extracting, or isolating the entity as a direct result of its own experimental work.**

## 3. Label Definitions
- `qualifies`: The publication contains primary experimental evidence that the authors synthesized, biosynthesized, extracted, fermented, purified, or harvested the target entity.
- `does_not_qualify`: Any of the following:
  1. **Reagent / Starting Material / Substrate:** The target entity was purchased, provided, or used as an input to synthesize something else.
  2. **Analytical Standard / Matrix Spike:** The target entity was used to calibrate an instrument (e.g. HPLC/GC-MS) or validate an analytical method.
  3. **Pharmacological / Biological Subject:** The target entity was administered to cells, animals, or humans to test its biological activity, without reporting its production.
  4. **Background Literature Review:** The target entity is cited in the introduction or discussion as prior work.
  5. **Not Found / Insufficient Text:** The entity does not appear or there is insufficient text to establish synthesis.

## 4. Conflict Resolution & Quality Control
A 20% random sample is independently labeled by a second reviewer. Inter-annotator agreement is evaluated via Cohen's kappa ($\kappa$). If $\kappa < 0.60$, the criteria are disambiguated and conflicting labels are reconciled before training or evaluation.

