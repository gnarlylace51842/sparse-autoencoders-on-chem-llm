# Results summary

Layer: **6**, model: `seyonec/ChemBERTa-zinc-base-v1`, corpus: `sagawa/ZINC-canonicalized` × 50000, SAE expansion: **4×**, L1: **0.001**.

## Headline

| Variant | features | labeled | frac labeled | unique concepts | median lift | max lift |
|---------|----------|---------|--------------|-----------------|-------------|----------|
| SAE | 200 | 200 | 1.00 | 28 | 1.57 | 24.93 |
| shuffled control | 150 | 150 | 1.00 | 6 | 1.02 | 1.81 |
| raw neurons | 200 | 200 | 1.00 | 22 | 1.36 | 22.43 |

## Best feature per chemical concept

| Concept | # features | best feature | precision | lift | top SMILES (truncated) |
|---------|------------|--------------|-----------|------|------------------------|
| alkene | 7 | 1699 | 1.00 | 24.93 | `CC(C)C[C@@H](CNC(=O)C/C=C/c1ccc(F)cc1)[NH+](C)C | COCCN1CC[NH+](C/C=C/c2ccccc2)C` |
| trifluoromethyl | 5 | 1710 | 1.00 | 24.68 | `C[C@]([NH3+])(C(N)=O)c1cccc(C(F)(F)F)c1 | O=S(=O)([N-]c1cccc2[nH]ncc12)c1cccc(C(` |
| imidazole | 1 | 2150 | 1.00 | 23.70 | `CCC[C@H](CCCc1[nH+]ccn1CCC)[NH2+]CC | CCCN[C@H](c1[nH+]ccn1CC)[C@@](C)(CC)[NH+]1` |
| bromine | 1 | 788 | 1.00 | 22.89 | `Cc1ccc(-c2noc(COc3cc(C)c(Br)c(C)c3)n2)cc1 | CCC[C@H]1C[C@@]1(O)Cc1cccc(Br)c1 | C` |
| cyclopentane | 2 | 622 | 0.75 | 17.33 | `CC[NH2+][C@@]1(C(N)=O)CCC[C@H]1CC[NH+](C)CC1CC1 | C[NH2+][C@]1(C(N)=O)CCC[C@H]1C` |
| cyclohexane | 1 | 2408 | 1.00 | 17.24 | `N#CCCN[C@H]1CCCC[C@H]1[C@@H]1CCC[NH2+]1 | C[C@H]1CCCC[C@@H]1OCCNC(=O)[C@@H]1CCCN` |
| sulfide | 2 | 1 | 1.00 | 14.10 | `COc1ccccc1NC(=O)CSc1nnc(CNc2ccc(C)c(Cl)c2)n1C[C@@H]1CCCO1 | COc1ccccc1NC(=O)CSc1` |
| sulfonamide | 6 | 1894 | 1.00 | 13.61 | `Cc1ccc(C)c(NC(=O)c2ccc(Cl)c(S(=O)(=O)N3CCCCC3)c2)c1 | CCOC(=O)c1cccc(NC(=O)c2ccc` |
| ester | 1 | 559 | 1.00 | 12.65 | `C=CCCC[NH+](C)CC(=O)c1c(C)[nH]c(C(=O)OCC)c1C | CCCCCNc1nc(C)c(C(=O)OCC)s1 | CCOC` |
| thiophene | 1 | 1247 | 0.80 | 10.58 | `CCCNCc1ccc(S(=O)(=O)N(CC)C[C@@H](C)CC)s1 | Cc1ccc(C(=O)NCc2cn[nH]c2N)s1 | Cc1ccc` |
| sulfonyl | 8 | 2173 | 1.00 | 10.49 | `CCCC1CCC(NC(=O)CCNS(=O)(=O)c2cccs2)CC1 | CC[C@@H](C)N(C)C(=O)CCNS(=O)(=O)c1ccccc` |
| alcohol | 3 | 554 | 1.00 | 10.26 | `CCOC(=O)c1cccc(OC[C@@H](O)Cn2cc[nH+]c2)c1 | CCCn1cc[nH+]c1C[C@@H](O)c1cccc(C(F)(` |
| chlorine | 3 | 305 | 1.00 | 7.70 | `CC(C)[NH2+]C[C@H](Cc1cccc(Cl)c1)c1cccc(Cl)c1 | CS(=O)(=O)NCCCNC(=O)C1(c2cccc(Cl)` |
| pyridine | 6 | 2725 | 1.00 | 5.95 | `CCCOc1ccc(CNC(=O)NC[C@@H](O)C(CC)CC)cn1 | CSc1ccc(C(=O)N[C@H](CO)c2ccccc2F)cn1 |` |
| ring_S | 8 | 1461 | 0.95 | 5.79 | `CCCNC(=O)c1sc2cccc(F)c2c1N | CCOCCn1/c(=N/C(=O)c2ccc(F)cc2)sc2cccc(OCC)c21 | CCC` |
| fluorine | 2 | 2399 | 1.00 | 5.37 | `C=CCNS(=O)(=O)c1cc(C(=O)N(C)c2cccc(C#N)c2)ccc1F | Cc1ccc(C(=O)N2CC[C@@H](C(=O)[O` |
| amine_tertiary | 2 | 2812 | 0.75 | 5.21 | `C[C@@H]1CCCCN1c1cc([C@@H](C)N)cc[nH+]1 | NC(=O)NC[C@H]1CCCCN1Cc1cc[nH]c1 | CC[C@` |
| ring_O | 9 | 2355 | 1.00 | 3.94 | `O=C(Nc1cnn(C[C@@H]2CCCO2)c1)c1cccnc1OCC(F)F | O=C(Nc1cnn(C[C@@H]2CCCO2)c1)N1CC[C` |
| halogen_any | 4 | 492 | 1.00 | 3.03 | `Cc1ccc(Nc2[nH+]cnc(Nc3ccc(F)cc3)c2N)cc1Cl | Cc1ccc(Nc2[nH+]cnc(NCc3ccc(Cl)cc3)c2` |
| aniline | 11 | 2056 | 1.00 | 2.72 | `CC(C)(C)C(=O)NCC(=O)N1CCN(c2ncnc3sccc23)CC1 | Cn1ccnc1C[C@@H]1CCCN(c2cnccc2C#N)C` |
| ether | 11 | 1003 | 1.00 | 2.13 | `COc1ccc(OC)c(C(=O)COC(=O)COc2ccccc2F)c1 | COc1ccc(OC)c(C(=O)N2C[C@@H](C)Oc3ccc(C` |
| amide | 21 | 389 | 1.00 | 1.64 | `CN(C(=O)Cc1ccc[nH]1)c1cccc(C#N)c1 | CN(C(=O)COCC(F)(F)F)c1ccccc1 | CCN(C(=O)c1cc` |
| aliphatic_ring | 31 | 1289 | 1.00 | 1.57 | `CCCN[C@@]1(C#N)CCC[C@@H]([NH+]2CCC[C@H](CC)CC2)C1 | NS(=O)(=O)c1ccc(CNC(=O)[C@@H` |
| carbonyl_any | 9 | 1784 | 1.00 | 1.46 | `COCCNC(=O)c1ccc(NCC(=O)N(C)C2CCCCC2)cc1 | CCNC(=O)c1ccc(CNS(=O)(=O)c2cn(C)cn2)cc` |
| benzene | 11 | 592 | 1.00 | 1.36 | `CN1CC[NH+](Cc2cc(Cc3c[nH]c4cc5ccccc5cc4c3=O)ccc2O)CC1 | Cc1ccc(C)c2sc(OC3CCN(C(=` |
| ring_N | 12 | 1944 | 1.00 | 1.27 | `COC(=O)C[NH+]1CC[C@@H](C)[C@@H](n2ccnc2)C1 | O=C(COc1ccccc1CN1CCC[C@@H](n2cccn2)` |
| aromatic_ring_any | 15 | 124 | 1.00 | 1.08 | `C=CC/[NH+]=c1\cc(C)oc(O)c1C(=O)CCC | Nc1cc(Cl)c(S(=O)(=O)/N=c2\[nH]ccs2)c(Cl)c1 ` |
| chiral_center | 7 | 848 | 1.00 | 1.00 | `COc1ccc([C@@H](C)[C@@H](C)[NH2+]C2CC2)cc1F | COc1cccc(F)c1[C@@H](C)[NH2+]C[C@@H]` |

## Layer comparison

|   layer |   n_features_analyzed |   n_labeled |   frac_labeled |   median_precision |   median_lift |   unique_concepts |
|--------:|----------------------:|------------:|---------------:|-------------------:|--------------:|------------------:|
|       2 |                   200 |         200 |              1 |                  1 |       1.49526 |                26 |
|       4 |                   200 |         200 |              1 |                  1 |       1.59622 |                28 |
|       6 |                   200 |         200 |              1 |                  1 |       1.57396 |                28 |

## Specificity tests

| concept           | pos_name         | neg_name        |   feature_id |   pos_activation |   neg_activation |     delta |
|:------------------|:-----------------|:----------------|-------------:|-----------------:|-----------------:|----------:|
| aromatic_ring_any | benzene          | cyclohexane     |         2049 |         0        |          0       |  0        |
| aromatic_ring_any | naphthalene      | decalin         |         2049 |         0        |          0       |  0        |
| pyridine          | pyridine         | piperidine      |         2725 |         0        |          0       |  0        |
| phenol            | phenol           | cyclohexanol    |         1189 |         1.21014  |          6.05832 | -4.84819  |
| carboxylic_acid   | acetic_acid      | ethanol         |         2404 |         7.96127  |          0       |  7.96127  |
| amine_primary     | ethylamine       | propane         |         2630 |         0.338335 |          1.03976 | -0.701429 |
| amide             | acetamide        | acetic_acid     |          690 |         0        |          0       |  0        |
| ester             | methyl_acetate   | acetic_acid     |          559 |         9.50728  |          4.74365 |  4.76363  |
| halogen_any       | chlorobenzene    | benzene         |          492 |         0        |          0       |  0        |
| trifluoromethyl   | trifluorotoluene | toluene         |         2872 |         8.70792  |          0       |  8.70792  |
| nitro             | nitrobenzene     | benzene         |         1028 |         9.52082  |          0       |  9.52082  |
| nitrile           | benzonitrile     | benzene         |         1004 |         6.49949  |          0       |  6.49949  |
| chiral_center     | L_alanine        | alanine_achiral |         1600 |         0.199827 |          0.79399 | -0.594163 |
| chiral_R_vs_S     | D_alanine        | L_alanine       |         2486 |         0        |          8.83301 | -8.83301  |
| ketone            | acetone          | isopropanol     |         2588 |         4.82465  |          0       |  4.82465  |
| alkene            | propene          | propane         |         1741 |         0        |          0       |  0        |
| alkyne            | propyne          | propane         |         2404 |         6.61203  |          0       |  6.61203  |
| furan             | furan            | tetrahydrofuran |         1142 |         6.65284  |          2.2055  |  4.44734  |
| thiophene         | thiophene        | thiolane        |         1247 |         0.433175 |          0       |  0.433175 |

## Figures

- `figures/fig_concept_by_layer.png`
- `figures/fig_feature_grid_layer06.png`
- `figures/fig_layer_comparison.png`
- `figures/fig_lift_layer06.png`
- `figures/fig_neurons_vs_sae_layer06.png`
- `figures/fig_per_concept_layer06.png`
- `figures/fig_specificity_layer06.png`
- `figures/fig_training_layer06.png`
