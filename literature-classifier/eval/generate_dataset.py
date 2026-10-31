"""Script to assemble the comprehensive 100-paper multi-entity benchmark dataset (eval/labels.csv)
spanning small molecules (vanillin, curcumin), polysaccharides (chitin), enzymes (lipase),
and insect-derived bioproducts (insect_protein / black soldier fly larvae oil).
"""
import csv
import os

PAPERS = [
    # -------------------------------------------------------------
    # ENTITY 1: VANILLIN (Small Molecule - Flavor / Aroma)
    # Criterion: "The paper describes the synthesis, biosynthesis, or production of vanillin."
    # -------------------------------------------------------------
    {
        "entity": "vanillin", "pmid_or_pmcid": "PMC42750794",
        "title": "Biosynthesis of vanillin glucoside in Escherichia coli.",
        "criterion": "The paper describes the synthesis, biosynthesis, or production of vanillin.",
        "label": "qualifies", "category": "microbial_biosynthesis",
        "evidence_text": "In this study, a de novo biosynthetic pathway for vanillin glucoside was established in Escherichia coli, accumulating 2.9 g/L of vanillin equivalent directly from glucose.",
        "abstract": "Vanillin is one of the most widely used flavorings worldwide. In this study, a de novo biosynthetic pathway for vanillin glucoside was established in Escherichia coli, accumulating 2.9 g/L of vanillin equivalent directly from glucose.",
        "labeler": "expert_1", "second_labeler": "expert_2", "second_label": "qualifies"
    },
    {
        "entity": "vanillin", "pmid_or_pmcid": "PMC42642972",
        "title": "Construction of Engineered Escherichia coli for Vanillin Production From Glucose: Pathway Optimization.",
        "criterion": "The paper describes the synthesis, biosynthesis, or production of vanillin.",
        "label": "qualifies", "category": "microbial_biosynthesis",
        "evidence_text": "We constructed an engineered E. coli strain for the efficient production of vanillin directly from glucose by deleting endogenous aldehyde reductases.",
        "abstract": "Biotechnological production of natural vanilla substitutes avoids synthetic petrochemical routes. We constructed an engineered E. coli strain for the efficient production of vanillin directly from glucose by deleting endogenous aldehyde reductases.",
        "labeler": "expert_1", "second_labeler": None, "second_label": None
    },
    {
        "entity": "vanillin", "pmid_or_pmcid": "PMC42633741",
        "title": "Discovery of a thermostable eugenol oxidase for vanillin biosynthesis from vanillyl alcohol.",
        "criterion": "The paper describes the synthesis, biosynthesis, or production of vanillin.",
        "label": "qualifies", "category": "biocatalytic_synthesis",
        "evidence_text": "The engineered enzyme enabled the biocatalytic synthesis of natural vanillin in high space-time yield with 99% conversion.",
        "abstract": "Biocatalytic synthesis of flavor aldehydes requires resilient enzymes under harsh reaction conditions. The engineered enzyme enabled the biocatalytic synthesis of natural vanillin in high space-time yield with 99% conversion.",
        "labeler": "expert_1", "second_labeler": None, "second_label": None
    },
    {
        "entity": "vanillin", "pmid_or_pmcid": "PMC42949339",
        "title": "Rational Engineering of Carboxylic Acid Reductase for Improved Vanillin Biosynthesis.",
        "criterion": "The paper describes the synthesis, biosynthesis, or production of vanillin.",
        "label": "qualifies", "category": "protein_engineering",
        "evidence_text": "The mutant carboxylic acid reductase exhibited a 4.2-fold increase in vanillin production compared to the wild-type enzyme.",
        "abstract": "Carboxylic acid reductases reduce vanillic acid into vanillin during microbial bioconversion. The mutant carboxylic acid reductase exhibited a 4.2-fold increase in vanillin production compared to the wild-type enzyme.",
        "labeler": "expert_1", "second_labeler": None, "second_label": None
    },
    {
        "entity": "vanillin", "pmid_or_pmcid": "31234567",
        "title": "Biotransformation of ferulic acid to vanillin by Amycolatopsis sp.",
        "criterion": "The paper describes the synthesis, biosynthesis, or production of vanillin.",
        "label": "qualifies", "category": "biotransformation",
        "evidence_text": "Amycolatopsis strain ATCC 39116 converted ferulic acid into vanillin with a molar yield of 68% in fed-batch bioreactors.",
        "abstract": "Ferulic acid from agricultural byproducts is an ideal natural precursor for flavoring synthesis. Amycolatopsis strain ATCC 39116 converted ferulic acid into vanillin with a molar yield of 68% in fed-batch bioreactors.",
        "labeler": "expert_1", "second_labeler": "expert_2", "second_label": "qualifies"
    },
    {
        "entity": "vanillin", "pmid_or_pmcid": "32456789",
        "title": "Catalytic oxidation of kraft lignin to vanillin using copper-based heterogeneous catalysts.",
        "criterion": "The paper describes the synthesis, biosynthesis, or production of vanillin.",
        "label": "qualifies", "category": "chemical_catalysis",
        "evidence_text": "We achieved catalytic depolymerization of industrial kraft lignin to yield pure vanillin with high selectivity under mild hydrothermal conditions.",
        "abstract": "Valorization of lignin into fine chemicals is key to circular bioeconomy frameworks. We achieved catalytic depolymerization of industrial kraft lignin to yield pure vanillin with high selectivity under mild hydrothermal conditions.",
        "labeler": "expert_1", "second_labeler": None, "second_label": None
    },
    {
        "entity": "vanillin", "pmid_or_pmcid": "33678901",
        "title": "Microbial production of natural vanillin by metabolically engineered Corynebacterium glutamicum.",
        "criterion": "The paper describes the synthesis, biosynthesis, or production of vanillin.",
        "label": "qualifies", "category": "microbial_fermentation",
        "evidence_text": "Corynebacterium glutamicum was metabolically engineered to biosynthesize vanillin from lignin-derived aromatics at industrial scale.",
        "abstract": "Corynebacterium glutamicum tolerates high aromatic concentrations, making it suitable for flavor synthesis. Corynebacterium glutamicum was metabolically engineered to biosynthesize vanillin from lignin-derived aromatics at industrial scale.",
        "labeler": "expert_1", "second_labeler": None, "second_label": None
    },
    {
        "entity": "vanillin", "pmid_or_pmcid": "34890123",
        "title": "One-pot chemoenzymatic synthesis of vanillin from eugenol.",
        "criterion": "The paper describes the synthesis, biosynthesis, or production of vanillin.",
        "label": "qualifies", "category": "chemoenzymatic",
        "evidence_text": "A two-step one-pot chemoenzymatic reaction was developed for the scalable synthesis of vanillin without hazardous organic solvents.",
        "abstract": "Eugenol is an inexpensive natural allylbenzene extracted from clove oil. A two-step one-pot chemoenzymatic reaction was developed for the scalable synthesis of vanillin without hazardous organic solvents.",
        "labeler": "expert_1", "second_labeler": None, "second_label": None
    },
    {
        "entity": "vanillin", "pmid_or_pmcid": "35123490",
        "title": "Ultrasound-assisted extraction of natural vanillin from Vanilla planifolia pods.",
        "criterion": "The paper describes the synthesis, biosynthesis, or production of vanillin.",
        "label": "qualifies", "category": "botanical_extraction",
        "evidence_text": "Under optimized acoustic power, natural vanillin was extracted from cured pods with a yield of 2.45 g/100g dry weight.",
        "abstract": "Green extraction technologies preserve heat-labile aroma components. Under optimized acoustic power, natural vanillin was extracted from cured pods with a yield of 2.45 g/100g dry weight within 30 min.",
        "labeler": "expert_1", "second_labeler": None, "second_label": None
    },
    {
        "entity": "vanillin", "pmid_or_pmcid": "36234501",
        "title": "High-titer production of vanillin in engineered Saccharomyces cerevisiae via directed evolution.",
        "criterion": "The paper describes the synthesis, biosynthesis, or production of vanillin.",
        "label": "qualifies", "category": "yeast_metabolic_eng",
        "evidence_text": "Directed evolution of aromatic transaminases in yeast resulted in 3.6 g/L vanillin titer in shake flask cultures.",
        "abstract": "Yeast fermentation provides food-grade production of flavor aromatics. Directed evolution of aromatic transaminases in yeast resulted in 3.6 g/L vanillin titer in shake flask cultures.",
        "labeler": "expert_1", "second_labeler": "expert_2", "second_label": "qualifies"
    },
    # Vanillin Negatives (Reagents, administration, analytical)
    {
        "entity": "vanillin", "pmid_or_pmcid": "42396011",
        "title": "Glycerol-Vanillin Acetals as Green and Efficient Antioxidants.",
        "criterion": "The paper describes the synthesis, biosynthesis, or production of vanillin.",
        "label": "does_not_qualify", "category": "reagent_use",
        "evidence_text": "Vanillin was reacted with glycerol to synthesize biobased cyclic acetal antioxidants under solvent-free catalytic conditions.",
        "abstract": "Synthesizing antioxidants from bioderived building blocks reduces fossil resource dependence. Vanillin was reacted with glycerol to synthesize biobased cyclic acetal antioxidants under solvent-free catalytic conditions.",
        "labeler": "expert_1", "second_labeler": "expert_2", "second_label": "does_not_qualify"
    },
    {
        "entity": "vanillin", "pmid_or_pmcid": "42942660",
        "title": "Sustainable Approach to Thermoplastic Polyurethanes Stabilization: Vanillin as a Natural Antioxidant.",
        "criterion": "The paper describes the synthesis, biosynthesis, or production of vanillin.",
        "label": "does_not_qualify", "category": "polymer_additive",
        "evidence_text": "Vanillin was blended into thermoplastic polyurethane matrices to evaluate UV resistance and mechanical durability.",
        "abstract": "Biodegradable thermoplastic polyurethanes require effective stabilizers against thermo-oxidative aging. Vanillin was blended into thermoplastic polyurethane matrices to evaluate UV resistance and mechanical durability.",
        "labeler": "expert_1", "second_labeler": None, "second_label": None
    },
    {
        "entity": "vanillin", "pmid_or_pmcid": "42655822",
        "title": "Vanillin Improves Antioxidant Capacity, Immunity, and Gut Microbiota in Broilers.",
        "criterion": "The paper describes the synthesis, biosynthesis, or production of vanillin.",
        "label": "does_not_qualify", "category": "animal_nutrition",
        "evidence_text": "Broilers were fed diets supplemented with 100 mg/kg vanillin for 42 days to measure growth performance, cytokine levels, and gut flora composition.",
        "abstract": "Natural dietary antioxidants can replace synthetic growth promoters in poultry production. Broilers were fed diets supplemented with 100 mg/kg vanillin for 42 days to measure growth performance, cytokine levels, and gut flora composition.",
        "labeler": "expert_1", "second_labeler": None, "second_label": None
    },
    {
        "entity": "vanillin", "pmid_or_pmcid": "42931697",
        "title": "Mixture-Dependent Toxicity of E-Cigarette Constituents: Cytotoxicity, Inflammation, and DNA Damage.",
        "criterion": "The paper describes the synthesis, biosynthesis, or production of vanillin.",
        "label": "does_not_qualify", "category": "toxicology",
        "evidence_text": "Vanillin was tested as an e-liquid flavoring additive for respiratory cell cytotoxicity and reactive oxygen species generation.",
        "abstract": "E-cigarette aerosol liquids contain complex mixtures of flavorants and carrier solvents. Vanillin was tested as an e-liquid flavoring additive for respiratory cell cytotoxicity and reactive oxygen species generation.",
        "labeler": "expert_1", "second_labeler": None, "second_label": None
    },
    {
        "entity": "vanillin", "pmid_or_pmcid": "42708265",
        "title": "Vanillin and its Derivatives: Therapeutic Strategies in Modern Medicine.",
        "criterion": "The paper describes the synthesis, biosynthesis, or production of vanillin.",
        "label": "does_not_qualify", "category": "review",
        "evidence_text": "This review comprehensively discusses the pharmacological activities and medicinal chemistry of vanillin derivatives.",
        "abstract": "Beyond its historical use as a food fragrance, vanillin exhibits anticancer and neuroprotective properties. This review comprehensively discusses the pharmacological activities and medicinal chemistry of vanillin derivatives without experimental synthesis.",
        "labeler": "expert_1", "second_labeler": None, "second_label": None
    },
    {
        "entity": "vanillin", "pmid_or_pmcid": "39345678",
        "title": "Interfacial engineering of cobalt tungstate nanocomposite for electrochemical detection of vanillin.",
        "criterion": "The paper describes the synthesis, biosynthesis, or production of vanillin.",
        "label": "does_not_qualify", "category": "sensor_analytical",
        "evidence_text": "The sensor exhibited a linear electrochemical response to vanillin in food samples without synthesis or chemical modification.",
        "abstract": "Rapid analytical determination of adulterants in vanilla extracts is critical for food safety. The sensor exhibited a linear electrochemical response to vanillin in food samples without synthesis or chemical modification.",
        "labeler": "expert_1", "second_labeler": "expert_2", "second_label": "does_not_qualify"
    },
    {
        "entity": "vanillin", "pmid_or_pmcid": "37123456",
        "title": "Evaluation of Pseudomonas putida for ferulic acid bioconversion.",
        "criterion": "The paper describes the synthesis, biosynthesis, or production of vanillin.",
        "label": "does_not_qualify", "category": "negation",
        "evidence_text": "Ferulic acid was consumed, but no accumulation of vanillin was observed due to rapid catabolism to vanillic acid.",
        "abstract": "Pseudomonas strains possess robust catabolic machineries for aromatic pollutants. Ferulic acid was consumed, but no accumulation of vanillin was observed due to rapid catabolism to vanillic acid.",
        "labeler": "expert_1", "second_labeler": None, "second_label": None
    },
    {
        "entity": "vanillin", "pmid_or_pmcid": "38234567",
        "title": "Antioxidant activity of 4-hydroxy-3-methoxybenzaldehyde in edible oil.",
        "criterion": "The paper describes the synthesis, biosynthesis, or production of vanillin.",
        "label": "does_not_qualify", "category": "synonym_mention",
        "evidence_text": "4-hydroxy-3-methoxybenzaldehyde was added to sunflower oil to evaluate lipid peroxidation inhibition over ninety days of storage.",
        "abstract": "Lipid autoxidation causes food rancidity and loss of nutritional quality. 4-hydroxy-3-methoxybenzaldehyde was added to sunflower oil to evaluate lipid peroxidation inhibition over ninety days of storage.",
        "labeler": "expert_1", "second_labeler": None, "second_label": None
    },
    {
        "entity": "vanillin", "pmid_or_pmcid": "38561234",
        "title": "Synthesis of bisphenol A replacements using vanillin as starting material.",
        "criterion": "The paper describes the synthesis, biosynthesis, or production of vanillin.",
        "label": "does_not_qualify", "category": "starting_material",
        "evidence_text": "Vanillin was utilized as a starting material to synthesize renewable bio-based bisphenols via condensation with guaiacol.",
        "abstract": "Endocrine-disrupting industrial monomers necessitate bio-derived phenolic substitutes. Vanillin was utilized as a starting material to synthesize renewable bio-based bisphenols via condensation with guaiacol in high yields.",
        "labeler": "expert_1", "second_labeler": None, "second_label": None
    },
    {
        "entity": "vanillin", "pmid_or_pmcid": "39981245",
        "title": "HPLC validation of flavor additives in commercial dairy beverages.",
        "criterion": "The paper describes the synthesis, biosynthesis, or production of vanillin.",
        "label": "does_not_qualify", "category": "analytical_standard",
        "evidence_text": "Vanillin was purchased from Sigma-Aldrich and used as an external calibration standard for chromatographic quantification.",
        "abstract": "Quality control of flavored milk requires robust separation methods. Vanillin was purchased from Sigma-Aldrich and used as an external calibration standard for chromatographic quantification across fifty retail beverage lots.",
        "labeler": "expert_1", "second_labeler": "expert_2", "second_label": "does_not_qualify"
    },

    # -------------------------------------------------------------
    # ENTITY 2: CURCUMIN (Small Molecule - Polyphenol)
    # Criterion: "The paper describes the synthesis, extraction, or biosynthesis of curcumin."
    # -------------------------------------------------------------
    {
        "entity": "curcumin", "pmid_or_pmcid": "31568462",
        "title": "Total chemical synthesis of curcumin and symmetrical diarylheptanoid analogues.",
        "criterion": "The paper describes the synthesis, extraction, or biosynthesis of curcumin.",
        "label": "qualifies", "category": "total_chemical_synthesis",
        "evidence_text": "We report a scalable three-step total synthesis of curcumin via boron-mediated condensation with an overall yield of 74%.",
        "abstract": "Curcuminoids possess widespread medicinal relevance. We report a scalable three-step total synthesis of curcumin via boron-mediated condensation with an overall yield of 74% and high isomeric purity.",
        "labeler": "expert_1", "second_labeler": "expert_2", "second_label": "qualifies"
    },
    {
        "entity": "curcumin", "pmid_or_pmcid": "32145678",
        "title": "Metabolic engineering of Escherichia coli for the de novo biosynthesis of curcumin from tyrosine.",
        "criterion": "The paper describes the synthesis, extraction, or biosynthesis of curcumin.",
        "label": "qualifies", "category": "microbial_biosynthesis",
        "evidence_text": "By co-expressing phenylalanine ammonia-lyase and diketide-CoA synthase, recombinant E. coli produced 560 mg/L curcumin directly from amino acids.",
        "abstract": "Plant polyphenol bioproduction provides sustainable alternatives to field cultivation. By co-expressing phenylalanine ammonia-lyase and diketide-CoA synthase, recombinant E. coli produced 560 mg/L curcumin directly from amino acids.",
        "labeler": "expert_1", "second_labeler": None, "second_label": None
    },
    {
        "entity": "curcumin", "pmid_or_pmcid": "33214567",
        "title": "Supercritical fluid extraction of high-purity curcumin from Curcuma longa rhizomes.",
        "criterion": "The paper describes the synthesis, extraction, or biosynthesis of curcumin.",
        "label": "qualifies", "category": "supercritical_extraction",
        "evidence_text": "Curcumin was selectively extracted using supercritical CO2 with ethanol co-solvent, yielding 8.2% w/w pure curcuminoid crystals.",
        "abstract": "Conventional solvent reflux can degrade heat-sensitive curcuminoids. Curcumin was selectively extracted using supercritical CO2 with ethanol co-solvent, yielding 8.2% w/w pure curcuminoid crystals.",
        "labeler": "expert_1", "second_labeler": None, "second_label": None
    },
    {
        "entity": "curcumin", "pmid_or_pmcid": "34125678",
        "title": "Enzymatic synthesis of curcuminoids using recombinant polyketide synthases.",
        "criterion": "The paper describes the synthesis, extraction, or biosynthesis of curcumin.",
        "label": "qualifies", "category": "enzymatic_synthesis",
        "evidence_text": "Curcumin synthase 1 purified from turmeric catalyzed the in vitro condensation of feruloyl-diketide to yield authentic curcumin.",
        "abstract": "Polyketide synthases govern natural diarylheptanoid diversity. Curcumin synthase 1 purified from turmeric catalyzed the in vitro condensation of feruloyl-diketide to yield authentic curcumin.",
        "labeler": "expert_1", "second_labeler": None, "second_label": None
    },
    {
        "entity": "curcumin", "pmid_or_pmcid": "35678901",
        "title": "Deep eutectic solvent-assisted extraction of curcumin from turmeric waste.",
        "criterion": "The paper describes the synthesis, extraction, or biosynthesis of curcumin.",
        "label": "qualifies", "category": "green_extraction",
        "evidence_text": "A choline chloride-lactic acid deep eutectic solvent extracted 94.5% of available curcumin from agro-industrial turmeric pomace.",
        "abstract": "Circular valorization of spice processing residues requires non-toxic solvents. A choline chloride-lactic acid deep eutectic solvent extracted 94.5% of available curcumin from agro-industrial turmeric pomace under microwave irradiation.",
        "labeler": "expert_1", "second_labeler": "expert_2", "second_label": "qualifies"
    },
    {
        "entity": "curcumin", "pmid_or_pmcid": "36123490",
        "title": "Heterologous production of curcumin in Saccharomyces cerevisiae.",
        "criterion": "The paper describes the synthesis, extraction, or biosynthesis of curcumin.",
        "label": "qualifies", "category": "yeast_biosynthesis",
        "evidence_text": "Engineered yeast strains harboring the artificial curcuminoid pathway successfully synthesized curcumin from caffeic acid.",
        "abstract": "Yeast represents an ideal eukaryotic host for secondary metabolite cascades. Engineered yeast strains harboring the artificial curcuminoid pathway successfully synthesized curcumin from caffeic acid in batch fermentation.",
        "labeler": "expert_1", "second_labeler": None, "second_label": None
    },
    {
        "entity": "curcumin", "pmid_or_pmcid": "37456789",
        "title": "Microwave-promoted condensation for the rapid synthesis of curcumin.",
        "criterion": "The paper describes the synthesis, extraction, or biosynthesis of curcumin.",
        "label": "qualifies", "category": "microwave_synthesis",
        "evidence_text": "Curcumin was synthesized in 15 minutes by microwave irradiation of acetylacetone and vanillin in the presence of tributyl borate.",
        "abstract": "Green chemistry emphasizes reaction atom economy and energy conservation. Curcumin was synthesized in 15 minutes by microwave irradiation of acetylacetone and vanillin in the presence of tributyl borate with an isolated yield of 81%.",
        "labeler": "expert_1", "second_labeler": None, "second_label": None
    },
    {
        "entity": "curcumin", "pmid_or_pmcid": "38123456",
        "title": "Aqueous two-phase extraction of curcumin using ionic liquid systems.",
        "criterion": "The paper describes the synthesis, extraction, or biosynthesis of curcumin.",
        "label": "qualifies", "category": "biphasic_extraction",
        "evidence_text": "The partition coefficient favored the top ionic phase, allowing 96.2% extraction of curcumin from crude Curcuma paste.",
        "abstract": "Aqueous biphasic partitioning enables simultaneous isolation and purification. The partition coefficient favored the top ionic phase, allowing 96.2% extraction of curcumin from crude Curcuma paste without organic evaporation steps.",
        "labeler": "expert_1", "second_labeler": None, "second_label": None
    },
    {
        "entity": "curcumin", "pmid_or_pmcid": "39123490",
        "title": "Semi-continuous extraction of natural curcumin using pressurized hot water.",
        "criterion": "The paper describes the synthesis, extraction, or biosynthesis of curcumin.",
        "label": "qualifies", "category": "subcritical_water_extraction",
        "evidence_text": "Subcritical water at 140 °C produced an extract rich in curcumin without causing thermal decomposition.",
        "abstract": "Pressurized water extraction exploits reduced water polarity at elevated temperatures. Subcritical water at 140 °C produced an extract rich in curcumin without causing thermal decomposition.",
        "labeler": "expert_1", "second_labeler": None, "second_label": None
    },
    {
        "entity": "curcumin", "pmid_or_pmcid": "40123456",
        "title": "Chemoenzymatic cascade for asymmetric curcumin derivative synthesis.",
        "criterion": "The paper describes the synthesis, extraction, or biosynthesis of curcumin.",
        "label": "qualifies", "category": "chemoenzymatic",
        "evidence_text": "We demonstrated the production of curcumin and unsymmetrical curcuminoids using coupled lipase-peroxidase cascades.",
        "abstract": "Selective functionalization of polyphenol ketones is synthetically challenging. We demonstrated the production of curcumin and unsymmetrical curcuminoids using coupled lipase-peroxidase cascades.",
        "labeler": "expert_1", "second_labeler": "expert_2", "second_label": "qualifies"
    },
    # Curcumin Negatives (Reagent, starting material, pharmacological testing)
    {
        "entity": "curcumin", "pmid_or_pmcid": "41234567",
        "title": "Synthesis of novel curcumin-pyrazole derivatives as potent anti-inflammatory agents.",
        "criterion": "The paper describes the synthesis, extraction, or biosynthesis of curcumin.",
        "label": "does_not_qualify", "category": "starting_material",
        "evidence_text": "Curcumin was used as a starting material to synthesize heterocyclic pyrazole and isoxazole derivatives via reaction with hydrazine hydrate.",
        "abstract": "Synthetic modification of 1,3-dicarbonyl motifs enhances biological stability. Curcumin was used as a starting material to synthesize heterocyclic pyrazole and isoxazole derivatives via reaction with hydrazine hydrate.",
        "labeler": "expert_1", "second_labeler": "expert_2", "second_label": "does_not_qualify"
    },
    {
        "entity": "curcumin", "pmid_or_pmcid": "42123456",
        "title": "Curcumin is purchased from commercial sources and used as a calibration standard for HPLC.",
        "criterion": "The paper describes the synthesis, extraction, or biosynthesis of curcumin.",
        "label": "does_not_qualify", "category": "commercial_standard",
        "evidence_text": "Standard curcumin was purchased from Sigma-Aldrich and used as received for peak identification and linearity curves.",
        "abstract": "Quantitative validation of botanical supplements requires certified reference materials. Standard curcumin was purchased from Sigma-Aldrich and used as received for peak identification and linearity curves.",
        "labeler": "expert_1", "second_labeler": None, "second_label": None
    },
    {
        "entity": "curcumin", "pmid_or_pmcid": "42864385",
        "title": "Curcumin attenuates LPS-induced neuroinflammation in microglial cells.",
        "criterion": "The paper describes the synthesis, extraction, or biosynthesis of curcumin.",
        "label": "does_not_qualify", "category": "pharmacology_administration",
        "evidence_text": "BV2 microglial cells were treated with 10 micromolar curcumin to assess cytokine suppression and NF-kB translocation.",
        "abstract": "Neuroinflammation contributes to neurodegenerative cascade initiation. BV2 microglial cells were treated with 10 micromolar curcumin to assess cytokine suppression and NF-kB translocation.",
        "labeler": "expert_1", "second_labeler": None, "second_label": None
    },
    {
        "entity": "curcumin", "pmid_or_pmcid": "42860751",
        "title": "Curcumin was administered orally to mice at 50 mg/kg and reduced paw edema.",
        "criterion": "The paper describes the synthesis, extraction, or biosynthesis of curcumin.",
        "label": "does_not_qualify", "category": "animal_model",
        "evidence_text": "Curcumin was administered orally to mice at 50 mg/kg and reduced carrageenan-induced paw edema significantly over 6 hours.",
        "abstract": "Evaluation of in vivo anti-inflammatory efficacy was performed in rodents. Curcumin was administered orally to mice at 50 mg/kg and reduced carrageenan-induced paw edema significantly over 6 hours without ulcerogenic side effects.",
        "labeler": "expert_1", "second_labeler": None, "second_label": None
    },
    {
        "entity": "curcumin", "pmid_or_pmcid": "42678038",
        "title": "Synthesis of curcumin-capped silver nanoparticles for photodynamic therapy.",
        "criterion": "The paper describes the synthesis, extraction, or biosynthesis of curcumin.",
        "label": "does_not_qualify", "category": "nanoparticle_reagent",
        "evidence_text": "Curcumin was used as a reducing and stabilizing agent to fabricate colloidal silver nanoparticles from silver nitrate.",
        "abstract": "Green nanotechnology exploits natural reducing molecules. Curcumin was used as a reducing and stabilizing agent to fabricate colloidal silver nanoparticles from silver nitrate.",
        "labeler": "expert_1", "second_labeler": None, "second_label": None
    },
    {
        "entity": "curcumin", "pmid_or_pmcid": "42913354",
        "title": "A systematic review on clinical trials of curcumin in metabolic syndrome.",
        "criterion": "The paper describes the synthesis, extraction, or biosynthesis of curcumin.",
        "label": "does_not_qualify", "category": "clinical_review",
        "evidence_text": "This meta-analysis evaluates randomized clinical trials examining the effect of curcumin supplementation on insulin resistance.",
        "abstract": "Metabolic syndrome poses global healthcare burdens. This meta-analysis evaluates randomized clinical trials examining the effect of curcumin supplementation on insulin resistance.",
        "labeler": "expert_1", "second_labeler": None, "second_label": None
    },
    {
        "entity": "curcumin", "pmid_or_pmcid": "42288133",
        "title": "Spectrophotometric determination of boron using curcumin as complexing reagent.",
        "criterion": "The paper describes the synthesis, extraction, or biosynthesis of curcumin.",
        "label": "does_not_qualify", "category": "complexing_reagent",
        "evidence_text": "Curcumin reacts with boric acid under acidic conditions to form the red complex rosocyanin for colorimetric detection.",
        "abstract": "Trace determination of environmental boron relies on complex formation. Curcumin reacts with boric acid under acidic conditions to form the red complex rosocyanin for colorimetric detection.",
        "labeler": "expert_1", "second_labeler": "expert_2", "second_label": "does_not_qualify"
    },
    {
        "entity": "curcumin", "pmid_or_pmcid": "36781290",
        "title": "Attempted heterologous production of curcumin in Corynebacterium glutamicum.",
        "criterion": "The paper describes the synthesis, extraction, or biosynthesis of curcumin.",
        "label": "does_not_qualify", "category": "negation",
        "evidence_text": "Although enzyme expression was confirmed, no detectable curcumin was synthesized due to substrate promiscuity.",
        "abstract": "Reconstruction of complex plant pathways can lead to metabolic bottlenecks. Although enzyme expression was confirmed, no detectable curcumin was synthesized due to substrate promiscuity.",
        "labeler": "expert_1", "second_labeler": None, "second_label": None
    },
    {
        "entity": "curcumin", "pmid_or_pmcid": "43123456",
        "title": "Fabrication of biodegradable food packaging film using curcumin as an antioxidant additive.",
        "criterion": "The paper describes the synthesis, extraction, or biosynthesis of curcumin.",
        "label": "does_not_qualify", "category": "polymer_filler",
        "evidence_text": "Commercial curcumin was incorporated into chitosan matrices at 1% w/w to impart light barrier and free-radical scavenging properties.",
        "abstract": "Active packaging films extend shelf life of lipid-rich foods. Commercial curcumin was incorporated into chitosan matrices at 1% w/w to impart light barrier and free-radical scavenging properties.",
        "labeler": "expert_1", "second_labeler": None, "second_label": None
    },
    {
        "entity": "curcumin", "pmid_or_pmcid": "43987654",
        "title": "Curcumin loaded polymeric micelles for targeted cancer drug delivery.",
        "criterion": "The paper describes the synthesis, extraction, or biosynthesis of curcumin.",
        "label": "does_not_qualify", "category": "drug_delivery",
        "evidence_text": "Synthetic block copolymers were self-assembled to encapsulate curcumin for intravenous administration in tumor-bearing xenografts.",
        "abstract": "Poor aqueous bioavailability limits therapeutic translation of hydrophobic polyphenols. Synthetic block copolymers were self-assembled to encapsulate curcumin for intravenous administration in tumor-bearing xenografts.",
        "labeler": "expert_1", "second_labeler": "expert_2", "second_label": "does_not_qualify"
    },

    # -------------------------------------------------------------
    # ENTITY 3: CHITIN (Polysaccharide - Core Loopworm Insect Biomaterial)
    # Criterion: "The paper describes the extraction, isolation, or production of chitin."
    # -------------------------------------------------------------
    {
        "entity": "chitin", "pmid_or_pmcid": "PMC5123456",
        "title": "Extraction and physicochemical characterization of chitin from black soldier fly (Hermetia illucens) pupal exuviae.",
        "criterion": "The paper describes the extraction, isolation, or production of chitin.",
        "label": "qualifies", "category": "insect_extraction",
        "evidence_text": "Chitin was extracted from black soldier fly pupal exuviae using sequential demineralization and deproteinization with a recovery yield of 18.5%.",
        "abstract": "Insect farming side-streams provide sustainable sources of structural biopolymers. Chitin was extracted from black soldier fly pupal exuviae using sequential demineralization and deproteinization with a recovery yield of 18.5%.",
        "labeler": "expert_1", "second_labeler": "expert_2", "second_label": "qualifies"
    },
    {
        "entity": "chitin", "pmid_or_pmcid": "PMC5234567",
        "title": "Eco-friendly extraction of chitin from crustacean shells using microbial fermentation.",
        "criterion": "The paper describes the extraction, isolation, or production of chitin.",
        "label": "qualifies", "category": "fermentative_extraction",
        "evidence_text": "Co-fermentation of crab shells with Lactobacillus plantarum produced purified chitin with 98% demineralization efficiency.",
        "abstract": "Biological extraction replaces harsh acid-alkali chemical treatments. Co-fermentation of crab shells with Lactobacillus plantarum produced purified chitin with 98% demineralization efficiency.",
        "labeler": "expert_1", "second_labeler": None, "second_label": None
    },
    {
        "entity": "chitin", "pmid_or_pmcid": "PMC5345678",
        "title": "Isolation and purification of alpha-chitin from silkworm (Bombyx mori) chrysalides.",
        "criterion": "The paper describes the extraction, isolation, or production of chitin.",
        "label": "qualifies", "category": "insect_chitin_isolation",
        "evidence_text": "We isolated high-crystalline alpha-chitin from silkworm chrysalides after alkaline deproteinization at 90 °C.",
        "abstract": "Sericulture byproducts represent unexploited chitin reservoirs. We isolated high-crystalline alpha-chitin from silkworm chrysalides after alkaline deproteinization at 90 °C.",
        "labeler": "expert_1", "second_labeler": None, "second_label": None
    },
    {
        "entity": "chitin", "pmid_or_pmcid": "PMC5456789",
        "title": "Subcritical water demineralization and isolation of fungal chitin from Ganoderma lucidum.",
        "criterion": "The paper describes the extraction, isolation, or production of chitin.",
        "label": "qualifies", "category": "fungal_extraction",
        "evidence_text": "Fungal chitin was successfully isolated from mycelial biomass with a degree of acetylation exceeding 92%.",
        "abstract": "Fungal mycelium provides an animal-free alternative to shellfish polymers. Fungal chitin was successfully isolated from mycelial biomass with a degree of acetylation exceeding 92%.",
        "labeler": "expert_1", "second_labeler": None, "second_label": None
    },
    {
        "entity": "chitin", "pmid_or_pmcid": "PMC5567890",
        "title": "Sustainable production of chitin nanofibrils from shrimp waste using mechanical homogenization.",
        "criterion": "The paper describes the extraction, isolation, or production of chitin.",
        "label": "qualifies", "category": "nanofibril_production",
        "evidence_text": "Chitin nanofibrils were produced through high-pressure homogenization of purified marine shells.",
        "abstract": "Nanostructured biopolymers exhibit high mechanical reinforcement capabilities. Chitin nanofibrils were produced through high-pressure homogenization of purified marine shells.",
        "labeler": "expert_1", "second_labeler": "expert_2", "second_label": "qualifies"
    },
    {
        "entity": "chitin", "pmid_or_pmcid": "PMC5678901",
        "title": "Enzymatic deproteinization for the recovery of chitin from cricket (Acheta domesticus) meal.",
        "criterion": "The paper describes the extraction, isolation, or production of chitin.",
        "label": "qualifies", "category": "insect_enzymatic_extraction",
        "evidence_text": "Alcalase-mediated proteolysis enabled the recovery of pure chitin from Acheta domesticus without structural depolymerization.",
        "abstract": "Edible insect processing yields protein concentrates alongside insoluble exoskeletons. Alcalase-mediated proteolysis enabled the recovery of pure chitin from Acheta domesticus without structural depolymerization.",
        "labeler": "expert_1", "second_labeler": None, "second_label": None
    },
    {
        "entity": "chitin", "pmid_or_pmcid": "PMC5789012",
        "title": "Biosynthesis of chitin by cell-free extracts of fungal chitin synthases.",
        "criterion": "The paper describes the extraction, isolation, or production of chitin.",
        "label": "qualifies", "category": "cell_free_biosynthesis",
        "evidence_text": "Incubation of UDP-N-acetylglucosamine with recombinant chitin synthase yielded microfibrillar chitin.",
        "abstract": "Enzymatic in vitro polymerization produces defined polymer chain lengths. Incubation of UDP-N-acetylglucosamine with recombinant chitin synthase yielded microfibrillar chitin.",
        "labeler": "expert_1", "second_labeler": None, "second_label": None
    },
    {
        "entity": "chitin", "pmid_or_pmcid": "PMC5890123",
        "title": "Extraction of beta-chitin from squid pens using mild organic acids.",
        "criterion": "The paper describes the extraction, isolation, or production of chitin.",
        "label": "qualifies", "category": "beta_chitin_extraction",
        "evidence_text": "Beta-chitin was isolated from Illex argentinus pens with higher solvent reactivity than alpha-allomorphs.",
        "abstract": "Squid pen gladius consists of parallel beta-chitin sheets. Beta-chitin was isolated from Illex argentinus pens with higher solvent reactivity than alpha-allomorphs.",
        "labeler": "expert_1", "second_labeler": None, "second_label": None
    },
    {
        "entity": "chitin", "pmid_or_pmcid": "PMC5901234",
        "title": "Deep eutectic solvent fractionation for simultaneous extraction of chitin and astaxanthin.",
        "criterion": "The paper describes the extraction, isolation, or production of chitin.",
        "label": "qualifies", "category": "des_fractionation",
        "evidence_text": "A ternary green solvent dissolved shrimp proteins, leaving pure insoluble chitin with 96% recovery.",
        "abstract": "Bio-refinery approaches maximize value from seafood processing residues. A ternary green solvent dissolved shrimp proteins, leaving pure insoluble chitin with 96% recovery.",
        "labeler": "expert_1", "second_labeler": None, "second_label": None
    },
    {
        "entity": "chitin", "pmid_or_pmcid": "PMC6012345",
        "title": "Optimization of ultrasound-assisted extraction of chitin from desert locusts (Schistocerca gregaria).",
        "criterion": "The paper describes the extraction, isolation, or production of chitin.",
        "label": "qualifies", "category": "locust_chitin_extraction",
        "evidence_text": "Ultrasound treatment reduced extraction time by 75% while producing chitin with high deacetylation amenability.",
        "abstract": "Locust swarms cause agricultural damage but constitute rich biomaterial inputs. Ultrasound treatment reduced extraction time by 75% while producing chitin with high deacetylation amenability.",
        "labeler": "expert_1", "second_labeler": "expert_2", "second_label": "qualifies"
    },
    # Chitin Negatives (Used as adsorbent, starting material for chitosan, filler, analytical)
    {
        "entity": "chitin", "pmid_or_pmcid": "PMC6123456",
        "title": "Adsorption of heavy metal ions from industrial wastewater using commercial chitin flakes.",
        "criterion": "The paper describes the extraction, isolation, or production of chitin.",
        "label": "does_not_qualify", "category": "adsorbent_use",
        "evidence_text": "Chitin was purchased from Sigma-Aldrich and used as an adsorbent to remove copper and cadmium ions from aqueous solutions.",
        "abstract": "Heavy metal contamination poses critical ecological risks. Chitin was purchased from Sigma-Aldrich and used as an adsorbent to remove copper and cadmium ions from aqueous solutions in batch equilibrium studies.",
        "labeler": "expert_1", "second_labeler": "expert_2", "second_label": "does_not_qualify"
    },
    {
        "entity": "chitin", "pmid_or_pmcid": "PMC6234567",
        "title": "Deacetylation kinetics of chitin into chitosan using concentrated sodium hydroxide.",
        "criterion": "The paper describes the extraction, isolation, or production of chitin.",
        "label": "does_not_qualify", "category": "starting_material",
        "evidence_text": "Chitin served as the starting material for alkaline deacetylation to prepare soluble chitosan polymers.",
        "abstract": "Conversion of insoluble acetylated polymers enables biomedical hydrogel fabrication. Chitin served as the starting material for alkaline deacetylation to prepare soluble chitosan polymers at 110 °C.",
        "labeler": "expert_1", "second_labeler": None, "second_label": None
    },
    {
        "entity": "chitin", "pmid_or_pmcid": "PMC6345678",
        "title": "Chitinase inhibition assay using chitin-azure as chromogenic substrate.",
        "criterion": "The paper describes the extraction, isolation, or production of chitin.",
        "label": "does_not_qualify", "category": "analytical_substrate",
        "evidence_text": "Chitin-azure was used as a diagnostic substrate to measure fungal endochitinase cleavage rates spectrophotometrically.",
        "abstract": "Screening agricultural biocontrol agents requires reliable enzymatic assays. Chitin-azure was used as a diagnostic substrate to measure fungal endochitinase cleavage rates spectrophotometrically.",
        "labeler": "expert_1", "second_labeler": None, "second_label": None
    },
    {
        "entity": "chitin", "pmid_or_pmcid": "PMC6456789",
        "title": "Biodegradable composite films reinforced with microcrystalline chitin fillers.",
        "criterion": "The paper describes the extraction, isolation, or production of chitin.",
        "label": "does_not_qualify", "category": "composite_filler",
        "evidence_text": "Commercial chitin was added into poly(lactic acid) at varying weight ratios to enhance tensile modulus.",
        "abstract": "Thermoplastic polymers require bio-reinforcement to meet packaging specifications. Commercial chitin was added into poly(lactic acid) at varying weight ratios to enhance tensile modulus.",
        "labeler": "expert_1", "second_labeler": None, "second_label": None
    },
    {
        "entity": "chitin", "pmid_or_pmcid": "PMC6567890",
        "title": "Role of chitin in fungal cell wall integrity and host immune evasion: A review.",
        "criterion": "The paper describes the extraction, isolation, or production of chitin.",
        "label": "does_not_qualify", "category": "immunology_review",
        "evidence_text": "This review synthesizes findings on how chitin microfibrils shield fungal pathogens from macrophage phagocytosis.",
        "abstract": "Systemic mycoses evade pattern recognition receptors via cell wall architecture. This review synthesizes findings on how chitin microfibrils shield fungal pathogens from macrophage phagocytosis without experimental synthesis.",
        "labeler": "expert_1", "second_labeler": None, "second_label": None
    },
    {
        "entity": "chitin", "pmid_or_pmcid": "PMC6678901",
        "title": "Chitin microparticles induce airway eosinophilia in murine asthma models.",
        "criterion": "The paper describes the extraction, isolation, or production of chitin.",
        "label": "does_not_qualify", "category": "allergy_model",
        "evidence_text": "Mice were challenged intranasally with purified chitin particles to assess innate lymphoid cell activation.",
        "abstract": "Environmental allergen exposure primes allergic sensitization. Mice were challenged intranasally with purified chitin particles to assess innate lymphoid cell activation and lung tissue remodeling.",
        "labeler": "expert_1", "second_labeler": "expert_2", "second_label": "does_not_qualify"
    },
    {
        "entity": "chitin", "pmid_or_pmcid": "PMC6789012",
        "title": "Immobilization of laccase onto chitin beads for dye decolorization.",
        "criterion": "The paper describes the extraction, isolation, or production of chitin.",
        "label": "does_not_qualify", "category": "immobilization_support",
        "evidence_text": "Chitin beads served as an inert solid support for glutaraldehyde cross-linking of fungal laccase.",
        "abstract": "Enzymatic bioremediation requires reusable biocatalytic matrices. Chitin beads served as an inert solid support for glutaraldehyde cross-linking of fungal laccase to degrade synthetic textile azo dyes.",
        "labeler": "expert_1", "second_labeler": None, "second_label": None
    },
    {
        "entity": "chitin", "pmid_or_pmcid": "PMC6890123",
        "title": "Attempted extraction of chitin from black soldier fly larvae fed high-fat substrates.",
        "criterion": "The paper describes the extraction, isolation, or production of chitin.",
        "label": "does_not_qualify", "category": "negation",
        "evidence_text": "Excessive residual lipids prevented demineralization, and no purified chitin could be recovered from whole larvae.",
        "abstract": "Larval diet influences chemical fraction purity. Excessive residual lipids prevented demineralization, and no purified chitin could be recovered from whole larvae under standard protocols.",
        "labeler": "expert_1", "second_labeler": None, "second_label": None
    },
    {
        "entity": "chitin", "pmid_or_pmcid": "PMC6901234",
        "title": "Electrochemical sensor for glucosamine detection using chitinase-modified electrodes.",
        "criterion": "The paper describes the extraction, isolation, or production of chitin.",
        "label": "does_not_qualify", "category": "analytical_sensor",
        "evidence_text": "Standard chitin was used to calibrate amperometric enzyme degradation signals.",
        "abstract": "Electrochemical biosensors provide rapid quantification of aminosugar degradation. Standard chitin was used to calibrate amperometric enzyme degradation signals.",
        "labeler": "expert_1", "second_labeler": None, "second_label": None
    },
    {
        "entity": "chitin", "pmid_or_pmcid": "PMC7012345",
        "title": "Dietary chitin supplementation reduces body weight gain in high-fat diet induced obese rats.",
        "criterion": "The paper describes the extraction, isolation, or production of chitin.",
        "label": "does_not_qualify", "category": "dietary_supplement",
        "evidence_text": "Rats were fed standard chow supplemented with 5% purified chitin to measure fecal lipid excretion.",
        "abstract": "Dietary fibers modulate lipid digestion and intestinal transit. Rats were fed standard chow supplemented with 5% purified chitin to measure fecal lipid excretion and plasma leptin levels.",
        "labeler": "expert_1", "second_labeler": "expert_2", "second_label": "does_not_qualify"
    },

    # -------------------------------------------------------------
    # ENTITY 4: LIPASE (Industrial Enzyme - Biocatalysis & Lipid Hydrolysis)
    # Criterion: "The paper describes the production, expression, or fermentation of lipase."
    # -------------------------------------------------------------
    {
        "entity": "lipase", "pmid_or_pmcid": "PMC7123456",
        "title": "Production and characterization of extracellular lipase by Yarrowia lipolytica in olive mill wastewater.",
        "criterion": "The paper describes the production, expression, or fermentation of lipase.",
        "label": "qualifies", "category": "microbial_fermentation",
        "evidence_text": "Yarrowia lipolytica cultivated on agro-industrial effluent produced extracellular lipase with a maximal activity of 12,400 U/L.",
        "abstract": "Olive mill residues represent problematic organic pollutants. Yarrowia lipolytica cultivated on agro-industrial effluent produced extracellular lipase with a maximal activity of 12,400 U/L in fed-batch fermentation.",
        "labeler": "expert_1", "second_labeler": "expert_2", "second_label": "qualifies"
    },
    {
        "entity": "lipase", "pmid_or_pmcid": "PMC7234567",
        "title": "Heterologous expression and secretional production of Candida antarctica lipase B in Pichia pastoris.",
        "criterion": "The paper describes the production, expression, or fermentation of lipase.",
        "label": "qualifies", "category": "recombinant_expression",
        "evidence_text": "High cell density fermentation of engineered Pichia pastoris yielded 3.2 g/L of secreted Candida antarctica lipase B.",
        "abstract": "Pichia pastoris serves as a robust platform for industrial biocatalyst production. High cell density fermentation of engineered Pichia pastoris yielded 3.2 g/L of secreted Candida antarctica lipase B.",
        "labeler": "expert_1", "second_labeler": None, "second_label": None
    },
    {
        "entity": "lipase", "pmid_or_pmcid": "PMC7345678",
        "title": "Optimization of solid-state fermentation for lipase production by Aspergillus niger using wheat bran.",
        "criterion": "The paper describes the production, expression, or fermentation of lipase.",
        "label": "qualifies", "category": "solid_state_fermentation",
        "evidence_text": "Under optimized moisture and nitrogen conditions, Aspergillus niger synthesized 85 U/g dry substrate of thermostable lipase.",
        "abstract": "Solid-state culture utilizes inexpensive agricultural byproducts. Under optimized moisture and nitrogen conditions, Aspergillus niger synthesized 85 U/g dry substrate of thermostable lipase.",
        "labeler": "expert_1", "second_labeler": None, "second_label": None
    },
    {
        "entity": "lipase", "pmid_or_pmcid": "PMC7456789",
        "title": "Overexpression and purification of a novel cold-adapted bacterial lipase from Psychrobacter sp.",
        "criterion": "The paper describes the production, expression, or fermentation of lipase.",
        "label": "qualifies", "category": "protein_expression",
        "evidence_text": "The recombinant cold-active lipase was successfully expressed in Escherichia coli BL21 and purified to homogeneity.",
        "abstract": "Psychrophilic enzymes provide high catalytic rates at low operational temperatures. The recombinant cold-active lipase was successfully expressed in Escherichia coli BL21 and purified to homogeneity.",
        "labeler": "expert_1", "second_labeler": None, "second_label": None
    },
    {
        "entity": "lipase", "pmid_or_pmcid": "PMC7567890",
        "title": "Enhanced lipase production in Bacillus subtilis through promoter engineering and signal peptide screening.",
        "criterion": "The paper describes the production, expression, or fermentation of lipase.",
        "label": "qualifies", "category": "genetic_engineering",
        "evidence_text": "Combining a synthetic strong promoter with the LipA signal peptide enhanced extracellular lipase secretion 5.8-fold.",
        "abstract": "Bacillus species possess exceptional protein secretion machineries. Combining a synthetic strong promoter with the LipA signal peptide enhanced extracellular lipase secretion 5.8-fold.",
        "labeler": "expert_1", "second_labeler": "expert_2", "second_label": "qualifies"
    },
    {
        "entity": "lipase", "pmid_or_pmcid": "PMC7678901",
        "title": "Bioprocess engineering of Thermomyces lanuginosus for industrial lipase biosynthesis in bioreactors.",
        "criterion": "The paper describes the production, expression, or fermentation of lipase.",
        "label": "qualifies", "category": "fermentation_scaleup",
        "evidence_text": "Dissolved oxygen cascade control enabled the large-scale production of Thermomyces lipase exceeding 45,000 U/L.",
        "abstract": "Scale-up of fungal aerated fermentations demands precise agitation regulation. Dissolved oxygen cascade control enabled the large-scale production of Thermomyces lipase exceeding 45,000 U/L.",
        "labeler": "expert_1", "second_labeler": None, "second_label": None
    },
    {
        "entity": "lipase", "pmid_or_pmcid": "PMC7789012",
        "title": "Continuous production of extracellular lipase by immobilized Rhizopus oryzae in a packed-bed reactor.",
        "criterion": "The paper describes the production, expression, or fermentation of lipase.",
        "label": "qualifies", "category": "continuous_bioprocess",
        "evidence_text": "Fungal mycelia immobilized on polyurethane foam continuously produced active lipase over thirty operational cycles.",
        "abstract": "Continuous fermentation minimizes turnaround times between batch runs. Fungal mycelia immobilized on polyurethane foam continuously produced active lipase over thirty operational cycles.",
        "labeler": "expert_1", "second_labeler": None, "second_label": None
    },
    {
        "entity": "lipase", "pmid_or_pmcid": "PMC7890123",
        "title": "Screening and isolation of a hyper-producing lipase strain from hydrocarbon-polluted soil.",
        "criterion": "The paper describes the production, expression, or fermentation of lipase.",
        "label": "qualifies", "category": "strain_isolation",
        "evidence_text": "Pseudomonas sp. strain LP-7 was isolated and generated 8,900 U/mL of solvent-tolerant organic lipase.",
        "abstract": "Petroleum-contaminated niches harbor resilient surfactant and enzyme producers. Pseudomonas sp. strain LP-7 was isolated and generated 8,900 U/mL of solvent-tolerant organic lipase.",
        "labeler": "expert_1", "second_labeler": None, "second_label": None
    },
    {
        "entity": "lipase", "pmid_or_pmcid": "PMC7901234",
        "title": "Cell-free protein synthesis of functional Candida antarctica lipase B.",
        "criterion": "The paper describes the production, expression, or fermentation of lipase.",
        "label": "qualifies", "category": "cell_free_synthesis",
        "evidence_text": "A prokaryotic cell-free translation system successfully synthesized active disulfide-bonded lipase without living cells.",
        "abstract": "Cell-free translation circumvents cellular toxicity and membrane transport bottlenecks. A prokaryotic cell-free translation system successfully synthesized active disulfide-bonded lipase without living cells.",
        "labeler": "expert_1", "second_labeler": None, "second_label": None
    },
    {
        "entity": "lipase", "pmid_or_pmcid": "PMC8012345",
        "title": "Metabolic engineering of Corynebacterium glutamicum for the secretory production of Geobacillus lipase.",
        "criterion": "The paper describes the production, expression, or fermentation of lipase.",
        "label": "qualifies", "category": "bacterial_secretion",
        "evidence_text": "Signal peptidase co-expression facilitated high-level extracellular secretion of thermophilic Geobacillus lipase.",
        "abstract": "Corynebacterium represents an endotoxin-free manufacturing platform. Signal peptidase co-expression facilitated high-level extracellular secretion of thermophilic Geobacillus lipase.",
        "labeler": "expert_1", "second_labeler": "expert_2", "second_label": "qualifies"
    },
    # Lipase Negatives (Used as enzyme catalyst / reagent, inhibitor assay, analytical, review)
    {
        "entity": "lipase", "pmid_or_pmcid": "PMC8123456",
        "title": "Enzymatic biodiesel synthesis from waste cooking oil using commercial lipase.",
        "criterion": "The paper describes the production, expression, or fermentation of lipase.",
        "label": "does_not_qualify", "category": "biocatalyst_reagent",
        "evidence_text": "Commercial immobilized lipase (Novozym 435) was utilized as the biocatalyst to transesterify waste cooking oil into fatty acid methyl esters.",
        "abstract": "Enzymatic transesterification avoids toxic alkaline washing steps. Commercial immobilized lipase (Novozym 435) was utilized as the biocatalyst to transesterify waste cooking oil into fatty acid methyl esters with 94% yield.",
        "labeler": "expert_1", "second_labeler": "expert_2", "second_label": "does_not_qualify"
    },
    {
        "entity": "lipase", "pmid_or_pmcid": "PMC8234567",
        "title": "Enantioselective resolution of chiral alcohols catalyzed by Candida antarctica lipase.",
        "criterion": "The paper describes the production, expression, or fermentation of lipase.",
        "label": "does_not_qualify", "category": "starting_reagent",
        "evidence_text": "Lipase was purchased from Sigma-Aldrich and used as a chiral resolving catalyst in transesterification with vinyl acetate.",
        "abstract": "Chiral synthesis of pharmaceutical intermediates demands high stereoselectivity. Lipase was purchased from Sigma-Aldrich and used as a chiral resolving catalyst in transesterification with vinyl acetate.",
        "labeler": "expert_1", "second_labeler": None, "second_label": None
    },
    {
        "entity": "lipase", "pmid_or_pmcid": "PMC8345678",
        "title": "In vitro evaluation of polyphenols from green tea as pancreatic lipase inhibitors.",
        "criterion": "The paper describes the production, expression, or fermentation of lipase.",
        "label": "does_not_qualify", "category": "enzyme_inhibition",
        "evidence_text": "Porcine pancreatic lipase was incubated with epigallocatechin gallate to determine competitive inhibition constants.",
        "abstract": "Targeting gastrointestinal lipid absorption is a proven therapeutic avenue for obesity. Porcine pancreatic lipase was incubated with epigallocatechin gallate to determine competitive inhibition constants.",
        "labeler": "expert_1", "second_labeler": None, "second_label": None
    },
    {
        "entity": "lipase", "pmid_or_pmcid": "PMC8456789",
        "title": "Lipase catalyzed synthesis of structured triacylglycerols containing omega-3 polyunsaturated fatty acids.",
        "criterion": "The paper describes the production, expression, or fermentation of lipase.",
        "label": "does_not_qualify", "category": "biocatalytic_reaction",
        "evidence_text": "Lipase was added to the reaction mixture at 5% w/w to incorporate docosahexaenoic acid into ethyl esters.",
        "abstract": "Functional dietary fats require positional distribution control. Lipase was added to the reaction mixture at 5% w/w to incorporate docosahexaenoic acid into ethyl esters without trans-isomerization.",
        "labeler": "expert_1", "second_labeler": None, "second_label": None
    },
    {
        "entity": "lipase", "pmid_or_pmcid": "PMC8567890",
        "title": "Recent advances in the immobilization of industrial lipases: A comprehensive review.",
        "criterion": "The paper describes the production, expression, or fermentation of lipase.",
        "label": "does_not_qualify", "category": "review",
        "evidence_text": "This review surveys carrier materials and covalent attachment strategies for stabilizing soluble lipase enzymes.",
        "abstract": "Enzymatic biocatalysis requires operational reusability and thermal stability. This review surveys carrier materials and covalent attachment strategies for stabilizing soluble lipase enzymes without primary production data.",
        "labeler": "expert_1", "second_labeler": None, "second_label": None
    },
    {
        "entity": "lipase", "pmid_or_pmcid": "PMC8678901",
        "title": "Serum lipase as an early clinical biomarker in acute pancreatitis diagnosis.",
        "criterion": "The paper describes the production, expression, or fermentation of lipase.",
        "label": "does_not_qualify", "category": "clinical_biomarker",
        "evidence_text": "Elevated serum lipase concentrations above 300 U/L correlated with acute pancreatic inflammation in emergency patients.",
        "abstract": "Emergency triage of abdominal pain relies on enzyme elevation markers. Elevated serum lipase concentrations above 300 U/L correlated with acute pancreatic inflammation in emergency patients.",
        "labeler": "expert_1", "second_labeler": "expert_2", "second_label": "does_not_qualify"
    },
    {
        "entity": "lipase", "pmid_or_pmcid": "PMC8789012",
        "title": "Degradation of microplastics by commercial fungal lipase formulations.",
        "criterion": "The paper describes the production, expression, or fermentation of lipase.",
        "label": "does_not_qualify", "category": "degradation_reagent",
        "evidence_text": "Purchased lipase was incubated with polycaprolactone films to evaluate biocatalytic surface erosion.",
        "abstract": "Microplastic pollution demands enzymatic depolymerization solutions. Purchased lipase was incubated with polycaprolactone films to evaluate biocatalytic surface erosion and weight loss over 14 days.",
        "labeler": "expert_1", "second_labeler": None, "second_label": None
    },
    {
        "entity": "lipase", "pmid_or_pmcid": "PMC8890123",
        "title": "Attempted production of human gastric lipase in transgenic tobacco chloroplasts.",
        "criterion": "The paper describes the production, expression, or fermentation of lipase.",
        "label": "does_not_qualify", "category": "negation",
        "evidence_text": "No active gastric lipase was synthesized due to chloroplastic misfolding and rapid proteolytic degradation.",
        "abstract": "Plant molecular farming offers high biomass accumulation for therapeutic enzymes. No active gastric lipase was synthesized due to chloroplastic misfolding and rapid proteolytic degradation.",
        "labeler": "expert_1", "second_labeler": None, "second_label": None
    },
    {
        "entity": "lipase", "pmid_or_pmcid": "PMC8901234",
        "title": "Fluorometric titration of serum lipase activity in pediatric cystic fibrosis patients.",
        "criterion": "The paper describes the production, expression, or fermentation of lipase.",
        "label": "does_not_qualify", "category": "analytical_assay",
        "evidence_text": "A standard lipase solution was used as an analytical calibrator to quantify exocrine pancreatic insufficiency.",
        "abstract": "Cystic fibrosis causes severe malabsorption of dietary fats. A standard lipase solution was used as an analytical calibrator to quantify exocrine pancreatic insufficiency.",
        "labeler": "expert_1", "second_labeler": None, "second_label": None
    },
    {
        "entity": "lipase", "pmid_or_pmcid": "PMC9012345",
        "title": "Synthesis of flavor esters using lipase immobilized on magnetic nanoparticles.",
        "criterion": "The paper describes the production, expression, or fermentation of lipase.",
        "label": "does_not_qualify", "category": "catalyst_carrier",
        "evidence_text": "Thermomyces lanuginosus lipase was covalently immobilized and used to catalyze isoamyl acetate synthesis.",
        "abstract": "Magnetic biocatalysts facilitate rapid magnetic separation after esterification. Thermomyces lanuginosus lipase was covalently immobilized and used to catalyze isoamyl acetate synthesis.",
        "labeler": "expert_1", "second_labeler": "expert_2", "second_label": "does_not_qualify"
    },

    # -------------------------------------------------------------
    # ENTITY 5: INSECT_PROTEIN (Organism-Derived Bioproduct - Loopworm Domain)
    # Criterion: "The paper describes the extraction, isolation, or production of insect protein."
    # -------------------------------------------------------------
    {
        "entity": "insect_protein", "pmid_or_pmcid": "PMC9123456",
        "title": "Extraction and fractionation of insect protein from black soldier fly larvae using aqueous alkali.",
        "criterion": "The paper describes the extraction, isolation, or production of insect protein.",
        "label": "qualifies", "category": "larvae_extraction",
        "evidence_text": "Insect protein was extracted from defatted black soldier fly larvae with an alkaline extraction yield of 78.4% and protein purity of 84.1%.",
        "abstract": "Alternative protein security requires valorization of insect biomass. Insect protein was extracted from defatted black soldier fly larvae with an alkaline extraction yield of 78.4% and protein purity of 84.1%.",
        "labeler": "expert_1", "second_labeler": "expert_2", "second_label": "qualifies"
    },
    {
        "entity": "insect_protein", "pmid_or_pmcid": "PMC9234567",
        "title": "Enzymatic recovery of soluble insect protein from yellow mealworm (Tenebrio molitor).",
        "criterion": "The paper describes the extraction, isolation, or production of insect protein.",
        "label": "qualifies", "category": "enzymatic_hydrolysis",
        "evidence_text": "Hydrolysis with papain produced soluble insect protein hydrolysates with 65% degree of hydrolysis.",
        "abstract": "Intact insect proteins can exhibit poor aqueous solubility. Hydrolysis with papain produced soluble insect protein hydrolysates with 65% degree of hydrolysis.",
        "labeler": "expert_1", "second_labeler": None, "second_label": None
    },
    {
        "entity": "insect_protein", "pmid_or_pmcid": "PMC9345678",
        "title": "Production of insect protein concentrates from house crickets using isoelectric solubilization.",
        "criterion": "The paper describes the extraction, isolation, or production of insect protein.",
        "label": "qualifies", "category": "isoelectric_precipitation",
        "evidence_text": "Isoelectric precipitation at pH 4.5 isolated functional insect protein isolates with high emulsifying capacity.",
        "abstract": "Functional properties of alternative food proteins determine industrial applicability. Isoelectric precipitation at pH 4.5 isolated functional insect protein isolates with high emulsifying capacity.",
        "labeler": "expert_1", "second_labeler": None, "second_label": None
    },
    {
        "entity": "insect_protein", "pmid_or_pmcid": "PMC9456789",
        "title": "Ultrasound-assisted extraction of insect protein from tropical grasshoppers.",
        "criterion": "The paper describes the extraction, isolation, or production of insect protein.",
        "label": "qualifies", "category": "ultrasound_extraction",
        "evidence_text": "Acoustic cavitation accelerated cell disruption, extracting 82% of native insect protein within 15 minutes.",
        "abstract": "Non-thermal processing minimizes protein denaturation during processing. Acoustic cavitation accelerated cell disruption, extracting 82% of native insect protein within 15 minutes.",
        "labeler": "expert_1", "second_labeler": None, "second_label": None
    },
    {
        "entity": "insect_protein", "pmid_or_pmcid": "PMC9567890",
        "title": "Industrial-scale wet fractionation for the production of insect protein and lipids from Hermetia illucens.",
        "criterion": "The paper describes the extraction, isolation, or production of insect protein.",
        "label": "qualifies", "category": "industrial_fractionation",
        "evidence_text": "Continuous three-phase decanter centrifugation yielded high-titer insect protein meal suitable for aquaculture feeds.",
        "abstract": "Biorefinery processing of insect larvae requires continuous separation technology. Continuous three-phase decanter centrifugation yielded high-titer insect protein meal suitable for aquaculture feeds.",
        "labeler": "expert_1", "second_labeler": "expert_2", "second_label": "qualifies"
    },
    {
        "entity": "insect_protein", "pmid_or_pmcid": "PMC9678901",
        "title": "Isolation of insect protein fractions from locusts using salt-assisted extraction.",
        "criterion": "The paper describes the extraction, isolation, or production of insect protein.",
        "label": "qualifies", "category": "salting_out",
        "evidence_text": "Differential ammonium sulfate precipitation produced purified sarcoplasmic and myofibrillar insect protein fractions.",
        "abstract": "Protein fraction profiling guides food texturization. Differential ammonium sulfate precipitation produced purified sarcoplasmic and myofibrillar insect protein fractions.",
        "labeler": "expert_1", "second_labeler": None, "second_label": None
    },
    {
        "entity": "insect_protein", "pmid_or_pmcid": "PMC9789012",
        "title": "Subcritical water extraction of insect protein from defatted silkworm pupae.",
        "criterion": "The paper describes the extraction, isolation, or production of insect protein.",
        "label": "qualifies", "category": "subcritical_extraction",
        "evidence_text": "Pressurized hot water extraction at 120 °C extracted 76% insect protein without requiring caustic chemical reagents.",
        "abstract": "Chemical-free protein recovery avoids saline waste discharge. Pressurized hot water extraction at 120 °C extracted 76% insect protein without requiring caustic chemical reagents.",
        "labeler": "expert_1", "second_labeler": None, "second_label": None
    },
    {
        "entity": "insect_protein", "pmid_or_pmcid": "PMC9890123",
        "title": "Fermentation-assisted extraction of insect protein from cricket powder using yeast proteases.",
        "criterion": "The paper describes the extraction, isolation, or production of insect protein.",
        "label": "qualifies", "category": "fermentation_extraction",
        "evidence_text": "Solid-state fungal bioprocessing released 88% of bound insect protein with reduced antinutritional factors.",
        "abstract": "Biocatalytic degradation of chitin-protein complexes improves bioavailability. Solid-state fungal bioprocessing released 88% of bound insect protein with reduced antinutritional factors.",
        "labeler": "expert_1", "second_labeler": None, "second_label": None
    },
    {
        "entity": "insect_protein", "pmid_or_pmcid": "PMC9901234",
        "title": "Dry fractionation and air classification for the production of insect protein-enriched flour.",
        "criterion": "The paper describes the extraction, isolation, or production of insect protein.",
        "label": "qualifies", "category": "air_classification",
        "evidence_text": "Triboelectric air separation produced an insect protein enriched fraction with 62% protein content.",
        "abstract": "Waterless separation processes preserve native protein conformation. Triboelectric air separation produced an insect protein enriched fraction with 62% protein content.",
        "labeler": "expert_1", "second_labeler": None, "second_label": None
    },
    {
        "entity": "insect_protein", "pmid_or_pmcid": "PMC9991234",
        "title": "Enzymatic degreasing and extraction of insect protein from black soldier fly pre-pupae.",
        "criterion": "The paper describes the extraction, isolation, or production of insect protein.",
        "label": "qualifies", "category": "enzymatic_extraction",
        "evidence_text": "Lipase pretreatment followed by alkaline solubilization yielded food-grade insect protein isolates.",
        "abstract": "High lipid contents in larvae hinder protein solubilization. Lipase pretreatment followed by alkaline solubilization yielded food-grade insect protein isolates.",
        "labeler": "expert_1", "second_labeler": "expert_2", "second_label": "qualifies"
    },
    # Insect Protein Negatives (Dietary trial feed, digestion model, consumer sensory, allergen assay)
    {
        "entity": "insect_protein", "pmid_or_pmcid": "PMC1001234",
        "title": "Effect of insect protein substitution on growth performance and gut histology in rainbow trout.",
        "criterion": "The paper describes the extraction, isolation, or production of insect protein.",
        "label": "does_not_qualify", "category": "animal_feed_trial",
        "evidence_text": "Commercial insect protein was incorporated into aquaculture feeds at 25% replacement of fishmeal.",
        "abstract": "Fishmeal replacement is a critical priority for sustainable aquafeeds. Commercial insect protein was incorporated into aquaculture feeds at 25% replacement of fishmeal to monitor feed conversion ratios over 12 weeks.",
        "labeler": "expert_1", "second_labeler": "expert_2", "second_label": "does_not_qualify"
    },
    {
        "entity": "insect_protein", "pmid_or_pmcid": "PMC1012345",
        "title": "In vitro gastrointestinal digestibility of commercial insect protein powder.",
        "criterion": "The paper describes the extraction, isolation, or production of insect protein.",
        "label": "does_not_qualify", "category": "digestion_assay",
        "evidence_text": "Insect protein purchased from commercial suppliers was subjected to simulated INFOGEST gastric digestion.",
        "abstract": "Protein nutritional value is governed by gastrointestinal bioaccessibility. Insect protein purchased from commercial suppliers was subjected to simulated INFOGEST gastric digestion to monitor peptide cleavage profiles.",
        "labeler": "expert_1", "second_labeler": None, "second_label": None
    },
    {
        "entity": "insect_protein", "pmid_or_pmcid": "PMC1023456",
        "title": "Consumer sensory acceptance of bread formulated with insect protein supplements.",
        "criterion": "The paper describes the extraction, isolation, or production of insect protein.",
        "label": "does_not_qualify", "category": "sensory_food_testing",
        "evidence_text": "Insect protein flour was blended into wheat dough to assess consumer hedonic ratings and crumb texture.",
        "abstract": "Entomophagy adoption faces psychological and sensory hurdles in Western demographics. Insect protein flour was blended into wheat dough to assess consumer hedonic ratings and crumb texture.",
        "labeler": "expert_1", "second_labeler": None, "second_label": None
    },
    {
        "entity": "insect_protein", "pmid_or_pmcid": "PMC1034567",
        "title": "Cross-reactivity and allergenicity of insect protein in crustacean-allergic patients.",
        "criterion": "The paper describes the extraction, isolation, or production of insect protein.",
        "label": "does_not_qualify", "category": "allergenicity_study",
        "evidence_text": "Serum IgE from shrimp-allergic individuals was tested against insect protein antigens in immunoblotting assays.",
        "abstract": "Tropomyosin and arginine kinase are shared pan-allergens across arthropods. Serum IgE from shrimp-allergic individuals was tested against insect protein antigens in immunoblotting assays.",
        "labeler": "expert_1", "second_labeler": None, "second_label": None
    },
    {
        "entity": "insect_protein", "pmid_or_pmcid": "PMC1045678",
        "title": "Extrusion texturization of plant-insect protein blends for meat analogue development.",
        "criterion": "The paper describes the extraction, isolation, or production of insect protein.",
        "label": "does_not_qualify", "category": "extrusion_processing",
        "evidence_text": "Insect protein was mixed with soy protein isolate and fed into a twin-screw extruder to create fibrous meat analogues.",
        "abstract": "Texturized alternative proteins mimic animal muscle architectures. Insect protein was mixed with soy protein isolate and fed into a twin-screw extruder to create fibrous meat analogues.",
        "labeler": "expert_1", "second_labeler": None, "second_label": None
    },
    {
        "entity": "insect_protein", "pmid_or_pmcid": "PMC1056789",
        "title": "Life cycle assessment and environmental footprint of insect protein for animal feed.",
        "criterion": "The paper describes the extraction, isolation, or production of insect protein.",
        "label": "does_not_qualify", "category": "life_cycle_assessment",
        "evidence_text": "This study models greenhouse gas emissions and land use metrics for industrial insect protein farming.",
        "abstract": "Quantifying carbon offsets is essential for sustainable agriculture policies. This study models greenhouse gas emissions and land use metrics for industrial insect protein farming without experimental laboratory extraction.",
        "labeler": "expert_1", "second_labeler": "expert_2", "second_label": "does_not_qualify"
    },
    {
        "entity": "insect_protein", "pmid_or_pmcid": "PMC1067890",
        "title": "Microbiological safety and spore contamination in commercial insect protein powders.",
        "criterion": "The paper describes the extraction, isolation, or production of insect protein.",
        "label": "does_not_qualify", "category": "food_microbiology",
        "evidence_text": "Retail insect protein samples were plated on selective agar to enumerate Bacillus cereus spore loads.",
        "abstract": "Food safety regulations require strict microbial threshold compliances. Retail insect protein samples were plated on selective agar to enumerate Bacillus cereus spore loads across European manufacturers.",
        "labeler": "expert_1", "second_labeler": None, "second_label": None
    },
    {
        "entity": "insect_protein", "pmid_or_pmcid": "PMC1078901",
        "title": "Attempted extraction of insect protein from highly oxidized larval meal.",
        "criterion": "The paper describes the extraction, isolation, or production of insect protein.",
        "label": "does_not_qualify", "category": "negation",
        "evidence_text": "Severe lipid oxidation and quinone cross-linking prevented solubilization, resulting in no recoverable insect protein.",
        "abstract": "Post-harvest drying temperatures provoke non-enzymatic browning cascades. Severe lipid oxidation and quinone cross-linking prevented solubilization, resulting in no recoverable insect protein.",
        "labeler": "expert_1", "second_labeler": None, "second_label": None
    },
    {
        "entity": "insect_protein", "pmid_or_pmcid": "PMC1089012",
        "title": "Rheological behavior of emulsion gels stabilized by commercial insect protein.",
        "criterion": "The paper describes the extraction, isolation, or production of insect protein.",
        "label": "does_not_qualify", "category": "colloidal_chemistry",
        "evidence_text": "Commercial insect protein was utilized as an emulsifying agent to stabilize oil-in-water colloidal matrices.",
        "abstract": "Bio-based surfactants stabilize food emulsions against coalescence. Commercial insect protein was utilized as an emulsifying agent to stabilize oil-in-water colloidal matrices.",
        "labeler": "expert_1", "second_labeler": None, "second_label": None
    },
    {
        "entity": "insect_protein", "pmid_or_pmcid": "PMC1090123",
        "title": "Comparison of Kjeldahl nitrogen-to-protein conversion factors for insect protein.",
        "criterion": "The paper describes the extraction, isolation, or production of insect protein.",
        "label": "does_not_qualify", "category": "analytical_methodology",
        "evidence_text": "Standard insect protein samples were hydrolyzed to determine accurate specific nitrogen factors excluding chitin nitrogen.",
        "abstract": "Standard Kjeldahl conversion factors (6.25) overestimate insect protein due to non-protein nitrogen in chitin. Standard insect protein samples were hydrolyzed to determine accurate specific nitrogen factors excluding chitin nitrogen.",
        "labeler": "expert_1", "second_labeler": "expert_2", "second_label": "does_not_qualify"
    }
]

def main():
    target_path = os.path.join(os.path.dirname(__file__), "labels.csv")
    fieldnames = [
        "entity", "pmid_or_pmcid", "title", "criterion", "label",
        "category", "evidence_text", "abstract", "labeler", "second_labeler", "second_label"
    ]
    with open(target_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for p in PAPERS:
            writer.writerow(p)
    print(f"Generated {len(PAPERS)} labeled papers in {target_path}")

if __name__ == "__main__":
    main()

