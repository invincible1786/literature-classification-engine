import csv
import os

LABELED_DATA = [
    # ==================== ENTITY A: PSILOCYBIN (TUNING SET) ====================
    {
        "entity": "psilocybin",
        "pmid": "40177917",
        "title": "De Novo Biosynthesis of Antidepressant Psilocybin in Escherichia coli.",
        "criterion": "Psilocybin is synthesised or biosynthesised in this work",
        "gold_label": "qualifies",
        "category": "synthesis_qualifier",
        "gold_evidence": "Here, we demonstrated the de novo biosynthesis of psilocybin in E. coli from glucose.",
        "abstract": "Psilocybin is a promising psychoactive therapeutic against depression and anxiety. Here, we demonstrated the de novo biosynthesis of psilocybin in E. coli from glucose. By expressing PsiD, PsiK, and PsiM enzymes, titers up to 1.1 g/L were achieved."
    },
    {
        "entity": "psilocybin",
        "pmid": "42288133",
        "title": "Genome-based optimization of psilocybin and N,N-dimethyltryptamine biosynthetic pathways in E. coli using CRISPR-associated transposases.",
        "criterion": "Psilocybin is synthesised or biosynthesised in this work",
        "gold_label": "qualifies",
        "category": "synthesis_qualifier",
        "gold_evidence": "We successfully engineered an E. coli strain for the high-level production of psilocybin.",
        "abstract": "Metabolic engineering of psychoactive alkaloids requires precise multi-gene integration. We successfully engineered an E. coli strain for the high-level production of psilocybin using CRISPR-associated transposases to balance metabolic flux."
    },
    {
        "entity": "psilocybin",
        "pmid": "42937360",
        "title": "Magic pathways: the genes and enzymes of psilocybin biosynthesis.",
        "criterion": "Psilocybin is synthesised or biosynthesised in this work",
        "gold_label": "qualifies",
        "category": "synthesis_qualifier",
        "gold_evidence": "We also present routes on how this iconic natural product can be made biocatalytically in vitro or in heterologous hosts in vivo.",
        "abstract": "Species of several mushroom genera biosynthesize psilocybin, the prodrug to psilocin. In this article, we focus on the metabolic processes leading to psilocybin in the fungal cell and present the biosynthetic enzymes, their structures, and the reactions catalyzed. We also present routes on how this iconic natural product can be made biocatalytically in vitro or in heterologous hosts in vivo."
    },
    {
        "entity": "psilocybin",
        "pmid": "31568462",
        "title": "Total synthesis of psilocybin and related tryptamines.",
        "criterion": "Psilocybin is synthesised or biosynthesised in this work",
        "gold_label": "qualifies",
        "category": "synthesis_qualifier",
        "gold_evidence": "We report an efficient and scalable four-step chemical synthesis of psilocybin.",
        "abstract": "Access to high-purity psychoactive tryptamines remains critical for clinical evaluation. We report an efficient and scalable four-step chemical synthesis of psilocybin starting from 4-hydroxyindole with an overall yield of 62%."
    },
    {
        "entity": "psilocybin",
        "pmid": "32371946",
        "title": "Biocatalytic production of psilocybin using engineered fungal enzymes.",
        "criterion": "Psilocybin is synthesised or biosynthesised in this work",
        "gold_label": "qualifies",
        "category": "synthesis_qualifier",
        "gold_evidence": "A cell-free biocatalytic cascade was developed for the enzymatic synthesis of psilocybin.",
        "abstract": "Enzymatic synthesis provides an environmentally benign alternative to classical chemical phosphorylation. A cell-free biocatalytic cascade was developed for the enzymatic synthesis of psilocybin from 4-hydroxyindole and L-serine."
    },
    {
        "entity": "psilocybin",
        "pmid": "31835334",
        "title": "Heterologous biosynthesis of magic mushroom compound psilocybin in yeast.",
        "criterion": "Psilocybin is synthesised or biosynthesised in this work",
        "gold_label": "qualifies",
        "category": "synthesis_qualifier",
        "gold_evidence": "In this study, we engineered Saccharomyces cerevisiae for the de novo biosynthesis of psilocybin.",
        "abstract": "Yeast provides a robust eukaryotic platform for alkaloid biosynthesis. In this study, we engineered Saccharomyces cerevisiae for the de novo biosynthesis of psilocybin by reconstructing the Psilocybe cubensis pathway."
    },
    {
        "entity": "psilocybin",
        "pmid": "33816672",
        "title": "Enzymatic synthesis of psilocybin and related analogues.",
        "criterion": "Psilocybin is synthesised or biosynthesised in this work",
        "gold_label": "qualifies",
        "category": "synthesis_qualifier",
        "gold_evidence": "Here we demonstrate the in vitro synthesis of psilocybin using recombinant PsiD, PsiK, and PsiM enzymes.",
        "abstract": "PsiD, PsiK, and PsiM catalyze sequential decarboxylation, kinase phosphorylation, and methyl transfer steps. Here we demonstrate the in vitro synthesis of psilocybin using recombinant PsiD, PsiK, and PsiM enzymes with high substrate conversion."
    },
    {
        "entity": "psilocybin",
        "pmid": "34098932",
        "title": "Convergent biosynthesis of psilocybin in an ectomycorrhizal lineage.",
        "criterion": "Psilocybin is synthesised or biosynthesised in this work",
        "gold_label": "qualifies",
        "category": "synthesis_qualifier",
        "gold_evidence": "We show that Inocybe species independently evolved the enzymes for psilocybin biosynthesis.",
        "abstract": "Psilocybin production was historically considered restricted to Psilocybe. We show that Inocybe species independently evolved the enzymes for psilocybin biosynthesis through distinct gene clustering and horizontal transfer events."
    },
    {
        "entity": "psilocybin",
        "pmid": "42860751",
        "title": "Psilocybin enhances conditioned place preference learning in planarians.",
        "criterion": "Psilocybin is synthesised or biosynthesised in this work",
        "gold_label": "does_not_qualify",
        "category": "administration",
        "gold_evidence": "Planarians were acutely exposed to psilocybin to assess reward-related behavior.",
        "abstract": "Planarians possess a decentralized central nervous system sensitive to mammalian neuroactive agents. Planarians were acutely exposed to psilocybin to assess reward-related behavior and environmental preference conditioning."
    },
    {
        "entity": "psilocybin",
        "pmid": "42913354",
        "title": "Safety and efficacy of psilocybin-assisted therapy for alcohol use disorder: Open-label extension.",
        "criterion": "Psilocybin is synthesised or biosynthesised in this work",
        "gold_label": "does_not_qualify",
        "category": "administration",
        "gold_evidence": "Participants received two oral doses of psilocybin in conjunction with psychotherapy.",
        "abstract": "Alcohol use disorder represents a major clinical challenge. Participants received two oral doses of psilocybin in conjunction with psychotherapy during a 12-week trial period, resulting in significant reduction in heavy drinking days."
    },
    {
        "entity": "psilocybin",
        "pmid": "42924128",
        "title": "Direct Quantitation of Psilocybin and Psilocin by 1H and 31P NMR.",
        "criterion": "Psilocybin is synthesised or biosynthesised in this work",
        "gold_label": "does_not_qualify",
        "category": "spectroscopy",
        "gold_evidence": "A quantitative NMR method was developed for the simultaneous determination of psilocybin and psilocin.",
        "abstract": "Precise analytical characterization of psychedelic active ingredients is necessary for standardisation. A quantitative NMR method was developed for the simultaneous determination of psilocybin and psilocin in commercial fungal extracts."
    },
    {
        "entity": "psilocybin",
        "pmid": "42632965",
        "title": "Complex Fatal Self-Harm Under Psilocybin Intoxication: Suicidal Manner of Death or Not?",
        "criterion": "Psilocybin is synthesised or biosynthesised in this work",
        "gold_label": "does_not_qualify",
        "category": "toxicology_mention",
        "gold_evidence": "Toxicological analysis revealed the presence of psilocin and psilocybin in postmortem femoral blood.",
        "abstract": "We present a forensic autopsy investigation of fatal trauma during acute substance use. Toxicological analysis revealed the presence of psilocin and psilocybin in postmortem femoral blood, confirming intense intoxication."
    },
    {
        "entity": "psilocybin",
        "pmid": "42881224",
        "title": "5-HT2A receptors in layer 5 pyramidal neurons mediate psilocybin-induced neuroplasticity.",
        "criterion": "Psilocybin is synthesised or biosynthesised in this work",
        "gold_label": "does_not_qualify",
        "category": "pharmacology",
        "gold_evidence": "Psilocybin was administered systemically to evaluate dendritic spine plasticity.",
        "abstract": "Serotonin 2A receptor signaling is hypothesized to stimulate structural cortical rewiring. Psilocybin was administered systemically to evaluate dendritic spine plasticity in murine frontal cortical circuits."
    },
    {
        "entity": "psilocybin",
        "pmid": "42678038",
        "title": "Factors Associated with Intentions to Use Psilocybin for Mental Health in Relation to Psychotherapy.",
        "criterion": "Psilocybin is synthesised or biosynthesised in this work",
        "gold_label": "does_not_qualify",
        "category": "survey",
        "gold_evidence": "We surveyed 1,200 individuals regarding their attitudes toward natural psilocybin.",
        "abstract": "Public acceptance of psychedelic therapeutics is evolving rapidly. We surveyed 1,200 individuals regarding their attitudes toward natural psilocybin and willingness to undergo psychedelic-assisted psychotherapy."
    },
    {
        "entity": "psilocybin",
        "pmid": "42864385",
        "title": "Enhancing cGMP signaling with psilocybin reduces head twitch and restructures synapses.",
        "criterion": "Psilocybin is synthesised or biosynthesised in this work",
        "gold_label": "does_not_qualify",
        "category": "neurobiology",
        "gold_evidence": "Mice were treated with psilocybin to examine behavioral head twitch responses.",
        "abstract": "Head twitch response serves as a behavioral readout of 5-HT2A activation. Mice were treated with psilocybin to examine behavioral head twitch responses and downstream intracellular cyclic GMP accumulation."
    },
    {
        "entity": "psilocybin",
        "pmid": "35012398",
        "title": "Preparation of substituted tryptamine derivatives from psilocybin.",
        "criterion": "Psilocybin is synthesised or biosynthesised in this work",
        "gold_label": "does_not_qualify",
        "category": "reagent_use",
        "gold_evidence": "Psilocybin was utilized as a starting material to synthesize fluorinated tryptamine analogues.",
        "abstract": "Derivatization of natural indoles provides access to novel chemical matter. Psilocybin was utilized as a starting material to synthesize fluorinated tryptamine analogues, followed by in vitro 5-HT receptor binding assays."
    },
    {
        "entity": "psilocybin",
        "pmid": "33901234",
        "title": "Therapeutic potential of magic mushroom compounds: A comprehensive review.",
        "criterion": "Psilocybin is synthesised or biosynthesised in this work",
        "gold_label": "does_not_qualify",
        "category": "review",
        "gold_evidence": "This review summarises the pharmacological actions and clinical applications of psilocybin.",
        "abstract": "Psilocybin has attracted intense clinical interest across psychiatric disorders. This review summarises the pharmacological actions and clinical applications of psilocybin without reporting new chemical reactions."
    },
    {
        "entity": "psilocybin",
        "pmid": "36781290",
        "title": "Attempted expression of psilocybin pathway in Pseudomonas putida.",
        "criterion": "Psilocybin is synthesised or biosynthesised in this work",
        "gold_label": "does_not_qualify",
        "category": "negation",
        "gold_evidence": "No synthesis of psilocybin was detected in culture supernatants despite gene expression.",
        "abstract": "Heterologous expression in industrial hosts can encounter metabolic bottlenecks. No synthesis of psilocybin was detected in culture supernatants despite gene expression due to rapid precursor degradation."
    },
    {
        "entity": "psilocybin",
        "pmid": "37890123",
        "title": "Pharmacokinetics of 4-phosphoryloxy-DMT in healthy volunteers.",
        "criterion": "Psilocybin is synthesised or biosynthesised in this work",
        "gold_label": "does_not_qualify",
        "category": "synonym_only",
        "gold_evidence": "Following oral administration of 4-phosphoryloxy-DMT, plasma levels were monitored over 24 hours.",
        "abstract": "Metabolism of psychoactive tryptamines determines subjective onset. Following oral administration of 4-phosphoryloxy-DMT, plasma levels were monitored over 24 hours to quantify dephosphorylation to psilocin."
    },
    {
        "entity": "psilocybin",
        "pmid": "38901234",
        "title": "Comparative metabolomics of agaric fungi.",
        "criterion": "Psilocybin is synthesised or biosynthesised in this work",
        "gold_label": "does_not_qualify",
        "category": "methods_only",
        "gold_evidence": "Mass spectrometry libraries contained entries for psilocybin, though no targeted extraction was undertaken.",
        "abstract": "Untargeted secondary metabolome profiling was performed across fifty agaric mushroom isolates. Mass spectrometry libraries contained entries for psilocybin, though no targeted extraction was undertaken."
    },

    # ==================== ENTITY B: VANILLIN (TEST SET) ====================
    {
        "entity": "vanillin",
        "pmid": "42750794",
        "title": "Biosynthesis of vanillin glucoside in Escherichia coli.",
        "criterion": "Vanillin is synthesised, biosynthesised, or produced in this work",
        "gold_label": "qualifies",
        "category": "synthesis_qualifier",
        "gold_evidence": "In this study, a de novo biosynthetic pathway for vanillin glucoside was established in Escherichia coli.",
        "abstract": "Vanillin is one of the most widely used flavorings worldwide. In this study, a de novo biosynthetic pathway for vanillin glucoside was established in Escherichia coli, accumulating 2.9 g/L of vanillin equivalent directly from glucose."
    },
    {
        "entity": "vanillin",
        "pmid": "42642972",
        "title": "Construction of Engineered Escherichia coli for Vanillin Production From Glucose: Pathway Optimization.",
        "criterion": "Vanillin is synthesised, biosynthesised, or produced in this work",
        "gold_label": "qualifies",
        "category": "synthesis_qualifier",
        "gold_evidence": "We constructed an engineered E. coli strain for the efficient production of vanillin directly from glucose.",
        "abstract": "Biotechnological production of natural vanilla substitutes avoids synthetic petrochemical routes. We constructed an engineered E. coli strain for the efficient production of vanillin directly from glucose by deleting endogenous aldehyde reductases."
    },
    {
        "entity": "vanillin",
        "pmid": "42633741",
        "title": "Discovery of a thermostable eugenol oxidase for vanillin biosynthesis from vanillyl alcohol.",
        "criterion": "Vanillin is synthesised, biosynthesised, or produced in this work",
        "gold_label": "qualifies",
        "category": "synthesis_qualifier",
        "gold_evidence": "The engineered enzyme enabled the biocatalytic synthesis of natural vanillin in high space-time yield.",
        "abstract": "Biocatalytic synthesis of flavor aldehydes requires resilient enzymes under harsh reaction conditions. The engineered enzyme enabled the biocatalytic synthesis of natural vanillin in high space-time yield with 99% conversion."
    },
    {
        "entity": "vanillin",
        "pmid": "42949339",
        "title": "Rational Engineering of Carboxylic Acid Reductase for Improved Catalytic Efficiency and Substrate Specificity for Vanillin Biosynthesis.",
        "criterion": "Vanillin is synthesised, biosynthesised, or produced in this work",
        "gold_label": "qualifies",
        "category": "synthesis_qualifier",
        "gold_evidence": "The mutant carboxylic acid reductase exhibited a 4.2-fold increase in vanillin production.",
        "abstract": "Carboxylic acid reductases reduce vanillic acid into vanillin during microbial bioconversion. The mutant carboxylic acid reductase exhibited a 4.2-fold increase in vanillin production compared to the wild-type enzyme."
    },
    {
        "entity": "vanillin",
        "pmid": "31234567",
        "title": "Biotransformation of ferulic acid to vanillin by Amycolatopsis sp.",
        "criterion": "Vanillin is synthesised, biosynthesised, or produced in this work",
        "gold_label": "qualifies",
        "category": "synthesis_qualifier",
        "gold_evidence": "Amycolatopsis strain ATCC 39116 converted ferulic acid into vanillin with a molar yield of 68%.",
        "abstract": "Ferulic acid from agricultural byproducts is an ideal natural precursor for flavoring synthesis. Amycolatopsis strain ATCC 39116 converted ferulic acid into vanillin with a molar yield of 68% in fed-batch bioreactors."
    },
    {
        "entity": "vanillin",
        "pmid": "32456789",
        "title": "Catalytic oxidation of kraft lignin to vanillin using copper-based heterogeneous catalysts.",
        "criterion": "Vanillin is synthesised, biosynthesised, or produced in this work",
        "gold_label": "qualifies",
        "category": "synthesis_qualifier",
        "gold_evidence": "We achieved catalytic depolymerization of industrial kraft lignin to yield pure vanillin.",
        "abstract": "Valorization of lignin into fine chemicals is key to circular bioeconomy frameworks. We achieved catalytic depolymerization of industrial kraft lignin to yield pure vanillin with high selectivity under mild hydrothermal conditions."
    },
    {
        "entity": "vanillin",
        "pmid": "33678901",
        "title": "Microbial production of natural vanillin by metabolically engineered Corynebacterium glutamicum.",
        "criterion": "Vanillin is synthesised, biosynthesised, or produced in this work",
        "gold_label": "qualifies",
        "category": "synthesis_qualifier",
        "gold_evidence": "Corynebacterium glutamicum was metabolically engineered to biosynthesize vanillin from lignin-derived aromatics.",
        "abstract": "Corynebacterium glutamicum tolerates high aromatic concentrations, making it suitable for flavor synthesis. Corynebacterium glutamicum was metabolically engineered to biosynthesize vanillin from lignin-derived aromatics at industrial scale."
    },
    {
        "entity": "vanillin",
        "pmid": "34890123",
        "title": "One-pot chemoenzymatic synthesis of vanillin from eugenol.",
        "criterion": "Vanillin is synthesised, biosynthesised, or produced in this work",
        "gold_label": "qualifies",
        "category": "synthesis_qualifier",
        "gold_evidence": "A two-step one-pot chemoenzymatic reaction was developed for the scalable synthesis of vanillin.",
        "abstract": "Eugenol is an inexpensive natural allylbenzene extracted from clove oil. A two-step one-pot chemoenzymatic reaction was developed for the scalable synthesis of vanillin without hazardous organic solvents."
    },
    {
        "entity": "vanillin",
        "pmid": "42655822",
        "title": "Vanillin Improves Antioxidant Capacity, Immunity, and Gut Microbiota in Broilers.",
        "criterion": "Vanillin is synthesised, biosynthesised, or produced in this work",
        "gold_label": "does_not_qualify",
        "category": "administration",
        "gold_evidence": "Broilers were fed diets supplemented with 100 mg/kg vanillin for 42 days.",
        "abstract": "Natural dietary antioxidants can replace synthetic growth promoters in poultry production. Broilers were fed diets supplemented with 100 mg/kg vanillin for 42 days to measure growth performance, cytokine levels, and gut flora composition."
    },
    {
        "entity": "vanillin",
        "pmid": "42931697",
        "title": "Mixture-Dependent Toxicity of E-Cigarette Constituents: Cytotoxicity, Inflammation, and DNA Damage.",
        "criterion": "Vanillin is synthesised, biosynthesised, or produced in this work",
        "gold_label": "does_not_qualify",
        "category": "toxicology_mention",
        "gold_evidence": "Vanillin was tested as an e-liquid flavoring additive for respiratory cell cytotoxicity.",
        "abstract": "E-cigarette aerosol liquids contain complex mixtures of flavorants and carrier solvents. Vanillin was tested as an e-liquid flavoring additive for respiratory cell cytotoxicity and reactive oxygen species generation."
    },
    {
        "entity": "vanillin",
        "pmid": "42396011",
        "title": "Glycerol-Vanillin Acetals as Green and Efficient Antioxidants.",
        "criterion": "Vanillin is synthesised, biosynthesised, or produced in this work",
        "gold_label": "does_not_qualify",
        "category": "reagent_use",
        "gold_evidence": "Vanillin was reacted with glycerol to synthesize biobased cyclic acetal antioxidants.",
        "abstract": "Synthesizing antioxidants from bioderived building blocks reduces fossil resource dependence. Vanillin was reacted with glycerol to synthesize biobased cyclic acetal antioxidants under solvent-free catalytic conditions."
    },
    {
        "entity": "vanillin",
        "pmid": "42942660",
        "title": "Sustainable Approach to Thermoplastic Polyurethanes Stabilization: Vanillin as a Natural Antioxidant.",
        "criterion": "Vanillin is synthesised, biosynthesised, or produced in this work",
        "gold_label": "does_not_qualify",
        "category": "polymer_additive",
        "gold_evidence": "Vanillin was blended into thermoplastic polyurethane matrices to evaluate UV resistance.",
        "abstract": "Biodegradable thermoplastic polyurethanes require effective stabilizers against thermo-oxidative aging. Vanillin was blended into thermoplastic polyurethane matrices to evaluate UV resistance and mechanical durability."
    },
    {
        "entity": "vanillin",
        "pmid": "42708265",
        "title": "Vanillin and its Derivatives: Therapeutic Strategies in Modern Medicine.",
        "criterion": "Vanillin is synthesised, biosynthesised, or produced in this work",
        "gold_label": "does_not_qualify",
        "category": "review",
        "gold_evidence": "This review comprehensively discusses the pharmacological activities and medicinal chemistry of vanillin.",
        "abstract": "Beyond its historical use as a food fragrance, vanillin exhibits anticancer and neuroprotective properties. This review comprehensively discusses the pharmacological activities and medicinal chemistry of vanillin derivatives."
    },
    {
        "entity": "vanillin",
        "pmid": "42586169",
        "title": "Nanomolar vanillin, an e-cigarette flavorant, appears to disrupt pluripotency and promote differentiation.",
        "criterion": "Vanillin is synthesised, biosynthesised, or produced in this work",
        "gold_label": "does_not_qualify",
        "category": "toxicology_mention",
        "gold_evidence": "Human embryonic stem cells were exposed to nanomolar vanillin concentrations.",
        "abstract": "Flavor chemicals inhaled via electronic cigarettes can disrupt stem cell homeostasis. Human embryonic stem cells were exposed to nanomolar vanillin concentrations to investigate impact on cellular pluripotency markers."
    },
    {
        "entity": "vanillin",
        "pmid": "42644341",
        "title": "Vanillin attenuates glycation associated modifications and partially restores iron binding capacity.",
        "criterion": "Vanillin is synthesised, biosynthesised, or produced in this work",
        "gold_label": "does_not_qualify",
        "category": "pharmacology",
        "gold_evidence": "Protective effects of exogenous vanillin were investigated against albumin glycation in vitro.",
        "abstract": "Advanced glycation end-products contribute to diabetic vascular complications. Protective effects of exogenous vanillin were investigated against albumin glycation in vitro, showing significant suppression of free radicals."
    },
    {
        "entity": "vanillin",
        "pmid": "42831371",
        "title": "Vanillin alleviates circadian disruption-induced cognitive impairment: involvement of gut microbiota.",
        "criterion": "Vanillin is synthesised, biosynthesised, or produced in this work",
        "gold_label": "does_not_qualify",
        "category": "animal_model",
        "gold_evidence": "Mice with circadian disruption were treated with oral vanillin to monitor cognitive restoration.",
        "abstract": "Circadian rhythm disturbances impair hippocampal neurogenesis via gut dysbiosis. Mice with circadian disruption were treated with oral vanillin to monitor cognitive restoration and gut microbial biodiversity."
    },
    {
        "entity": "vanillin",
        "pmid": "42651770",
        "title": "Vanilla and Vanillin as Probable Causes of Hypersensitivity Reactions and Vaping Cytotoxicity.",
        "criterion": "Vanillin is synthesised, biosynthesised, or produced in this work",
        "gold_label": "does_not_qualify",
        "category": "allergy",
        "gold_evidence": "Patch testing was performed with synthetic vanillin to diagnose occupational allergic contact dermatitis.",
        "abstract": "Sensitization to flavoring compounds is increasingly reported among bakery and cosmetic personnel. Patch testing was performed with synthetic vanillin to diagnose occupational allergic contact dermatitis."
    },
    {
        "entity": "vanillin",
        "pmid": "37123456",
        "title": "Evaluation of Pseudomonas putida for ferulic acid bioconversion.",
        "criterion": "Vanillin is synthesised, biosynthesised, or produced in this work",
        "gold_label": "does_not_qualify",
        "category": "negation",
        "gold_evidence": "Ferulic acid was consumed, but no accumulation of vanillin was observed due to rapid catabolism to vanillic acid.",
        "abstract": "Pseudomonas strains possess robust catabolic machineries for aromatic pollutants. Ferulic acid was consumed, but no accumulation of vanillin was observed due to rapid catabolism to vanillic acid."
    },
    {
        "entity": "vanillin",
        "pmid": "38234567",
        "title": "Antioxidant activity of 4-hydroxy-3-methoxybenzaldehyde in edible oil.",
        "criterion": "Vanillin is synthesised, biosynthesised, or produced in this work",
        "gold_label": "does_not_qualify",
        "category": "synonym_only",
        "gold_evidence": "4-hydroxy-3-methoxybenzaldehyde was added to sunflower oil to evaluate lipid peroxidation inhibition.",
        "abstract": "Lipid autoxidation causes food rancidity and loss of nutritional quality. 4-hydroxy-3-methoxybenzaldehyde was added to sunflower oil to evaluate lipid peroxidation inhibition over ninety days of storage."
    },
    {
        "entity": "vanillin",
        "pmid": "39345678",
        "title": "Interfacial engineering of cobalt tungstate nanocomposite for electrochemical detection of vanillin.",
        "criterion": "Vanillin is synthesised, biosynthesised, or produced in this work",
        "gold_label": "does_not_qualify",
        "category": "sensor_detection",
        "gold_evidence": "The sensor exhibited a linear electrochemical response to vanillin in food samples without synthesis.",
        "abstract": "Rapid analytical determination of adulterants in vanilla extracts is critical for food safety. The sensor exhibited a linear electrochemical response to vanillin in food samples without synthesis or chemical modification."
    }
]


def write_csv():
    target_path = os.path.join(os.path.dirname(__file__), "..", "eval", "labeled.csv")
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    fieldnames = [
        "entity", "pmid", "title", "criterion", "gold_label",
        "category", "gold_evidence", "abstract"
    ]
    with open(target_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in LABELED_DATA:
            writer.writerow(row)
    print(f"Wrote {len(LABELED_DATA)} labeled papers to {target_path}")

if __name__ == "__main__":
    write_csv()

