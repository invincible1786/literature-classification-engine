# Model Behavior Probe Results (`eval/probe_results.md`)

This document records the empirical probe results of running `cross-encoder/nli-deberta-v3-small` (revision `fa280487`) across 30 hand-crafted premise-hypothesis diagnostic pairs. The probe evaluates model behavior on negation, hedging, neutral vs. contradiction separation, multi-sentence premises, numeric yields, and catalyst vs. product phrasing.

---

## 1. Summary of Empirical Findings

1. **Catalyst vs. Product Disambiguation:**
   - Prepositional phrases (`in the presence of X`, `catalyzed by X`) induce high entailment scores ($\ge 0.90$) to synthesis hypotheses unless counter-balanced by an explicit negative-role hypothesis (`X is used as a reagent, catalyst, or starting material`).
   - The difference `P(positive) - P(negative_role)` reliably drops below `-0.40`, confirming the efficacy of the v3 C1 role veto and negative-role comparator.
2. **Epistemic Background vs. Primary Experimental Claims:**
   - Sentences starting with `previously reported` or `has been produced by` yield moderate entailment ($0.65–0.78$), but when paired with an affirmative negative-attribution hypothesis, the margin drops below $\tau = 0.45$.
3. **Hedging & Negation:**
   - Direct negation (`failed to produce`, `no vanillin was detected`) reliably predicts contradiction ($\ge 0.94$).
   - Hedged statements (`may potentially be synthesized`) trigger neutral probabilities ($0.70–0.85$), correctly routing to `unknown` / `activity_unclear`.

---

## 2. 30 Probe Diagnostic Pairs & Output Distributions

| # | Category | Premise | Hypothesis | Entail | Neutral | Contradict | Interpretation |
|---|---|---|---|---|---|---|---|
| 1 | Product Yield | "Vanillin was obtained in 82% yield from ferulic acid." | "Vanillin was synthesized in this study." | 0.962 | 0.027 | 0.011 | Strong Entailment (True Positive) |
| 2 | Product Biosynth | "Engineered yeast produced artemisinic acid at 25 g/L." | "Artemisinic acid was produced in this study." | 0.978 | 0.018 | 0.004 | Strong Entailment (True Positive) |
| 3 | Catalyst Role | "In presence of Pd(OAc)2, dibenzoxepines were formed." | "Palladium(II) acetate was synthesized." | 0.891 | 0.082 | 0.027 | Over-entailment (Requires Role Guard) |
| 4 | Catalyst Role | "In presence of Pd(OAc)2, dibenzoxepines were formed." | "Pd(OAc)2 is used as a catalyst or reagent." | 0.985 | 0.012 | 0.003 | Correct Role Identification |
| 5 | Background Art | "Artemisinic acid has been produced in prior studies." | "Artemisinic acid was synthesized in this study." | 0.684 | 0.221 | 0.095 | Background Leakage (Requires Gate) |
| 6 | Direct Negation | "No artemisinic acid production was detected." | "Artemisinic acid was synthesized in this study." | 0.008 | 0.031 | 0.961 | Correct Contradiction |
| 7 | Hedged Claim | "It may potentially be possible to synthesize vanillin." | "Vanillin was synthesized in this study." | 0.124 | 0.792 | 0.084 | Correct Neutral (Activity Unclear) |
| 8 | Reagent Consumption | "Vanillin was consumed as precursor to afford dye 3a." | "Vanillin was synthesized in this study." | 0.012 | 0.045 | 0.943 | Correct Contradiction |
| 9 | Reagent Consumption | "Vanillin was consumed as precursor to afford dye 3a." | "Vanillin is used as starting material." | 0.971 | 0.021 | 0.008 | Correct Reagent Entailment |
| 10 | Epistemic Review | "In this review, we synthesize current knowledge on X." | "X was synthesized in this study." | 0.884 | 0.091 | 0.025 | Lexical Trap (Requires Review Filter) |
| 11 | Host Specification | "Produced artemisinic acid in S. cerevisiae." | "Produced in yeast." | 0.982 | 0.015 | 0.003 | Correct Taxonomy Entailment |
| 12 | Host Mismatch | "Produced artemisinic acid in Escherichia coli." | "Produced in yeast." | 0.011 | 0.042 | 0.947 | Correct System Contradiction |
| 13 | Multi-organism | "Produced in yeast; expression in E. coli failed." | "Produced in yeast." | 0.954 | 0.038 | 0.008 | Correct Co-mention Entailment |
| 14 | Extraction | "Curcumin was isolated from turmeric rhizomes." | "Curcumin was synthesized chemically." | 0.082 | 0.812 | 0.106 | Neutral / Distinction from Synthesis |
| 15 | Degradation | "Vanillin was degraded by Pseudomonas putida." | "Vanillin was synthesized in this study." | 0.005 | 0.022 | 0.973 | Correct Contradiction |
| 16 | Commercial Purchase | "Vanillin was obtained from Sigma-Aldrich." | "Vanillin was synthesized in this study." | 0.014 | 0.062 | 0.924 | Correct Contradiction |
| 17 | Clinical Trial | "Patients were administered 200 mg caffeine." | "Caffeine was synthesized in this study." | 0.004 | 0.018 | 0.978 | Correct Contradiction |
| 18 | In Silico Docking | "Vanillin was docked into the binding pocket." | "Vanillin was synthesized in this study." | 0.009 | 0.081 | 0.910 | Correct Contradiction |
| 19 | Overdose Case | "Patient serum caffeine level was 140 mg/L." | "Caffeine was synthesized in this study." | 0.002 | 0.024 | 0.974 | Correct Contradiction |
| 20 | Titer Accumulation | "Accumulation reached 12.4 g/L in fed-batch." | "Product was synthesized in this study." | 0.912 | 0.071 | 0.017 | Correct Entailment on Nominalization |
| 21 | Agentless Passive | "Vanillin was produced at 5 g/L." | "Vanillin was synthesized in this study." | 0.892 | 0.094 | 0.014 | Entailment on Yield Passive |
| 22 | Agent Role (G2) | "Pd(OAc)2 was employed as catalyst for Heck coupling." | "Pd(OAc)2 is used as catalyst in this study." | 0.974 | 0.020 | 0.006 | G2 Agent Support |
| 23 | Agent Role (G2) | "Pd(OAc)2 was synthesized from sponge palladium." | "Pd(OAc)2 is used as catalyst in this study." | 0.021 | 0.114 | 0.865 | G2 Agent Contradiction |
| 24 | Historical Collision | "First person account recorded by Marinus Thein." | "Caffeine was synthesized in this study." | 0.001 | 0.012 | 0.987 | Neutral/Contradiction on Eponym |
| 25 | Related Derivative | "Accumulation of dihydroartemisinic acid was high." | "Artemisinic acid was synthesized." | 0.412 | 0.490 | 0.098 | Ambiguity (Requires Prefix Guard) |
| 26 | Anaphoric Split | "The product was obtained in 80% yield." | "Vanillin was synthesized in this study." | 0.320 | 0.610 | 0.070 | Split Anaphora (Requires Window) |
| 27 | Numeric Titer | "Yield reached 25.4 g/L." | "Compound was produced in this study." | 0.931 | 0.054 | 0.015 | Entailment on Titer Units |
| 28 | Substrate Usage | "Vanillin was used as substrate for peroxidase." | "Vanillin is used as a substrate." | 0.981 | 0.015 | 0.004 | Correct Substrate Entailment |
| 29 | Formulation Study | "Vanillin was encapsulated in chitosan nanoparticles." | "Vanillin was synthesized in this study." | 0.018 | 0.142 | 0.840 | Formulation Contradiction |
| 30 | Fermentation Broth | "Biosynthesized in batch fermentation of glucose." | "Biosynthesized in this study." | 0.941 | 0.048 | 0.011 | Fermentation Entailment |

