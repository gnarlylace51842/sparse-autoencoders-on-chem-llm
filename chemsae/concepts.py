"""Curated SMARTS dictionary for named chemical concepts.

Each entry compiles to an RDKit Mol pattern. Matching is done with
`mol.HasSubstructMatch(pattern)`. The list intentionally mixes broad concepts
(any aromatic ring, any halogen) with specific ones (benzene, pyridine) so the
analysis pipeline can both find features and characterize their selectivity.
"""

from __future__ import annotations

from rdkit import Chem


SMARTS: dict[str, str] = {
    # Aromaticity / ring systems
    "aromatic_ring_any": "a",
    "benzene": "c1ccccc1",
    "pyridine": "c1ccncc1",
    "pyrrole": "c1cc[nH]c1",
    "furan": "c1ccoc1",
    "thiophene": "c1ccsc1",
    "imidazole": "c1ncc[nH]1",
    "fused_aromatic": "c1ccc2ccccc2c1",
    "aliphatic_ring": "[R;!a]",
    "cyclohexane": "C1CCCCC1",
    "cyclopentane": "C1CCCC1",

    # Heteroatoms in rings
    "ring_N": "[#7;R]",
    "ring_O": "[#8;R]",
    "ring_S": "[#16;R]",

    # Functional groups (oxygen)
    "hydroxyl": "[OX2H1]",
    "alcohol": "[CX4][OX2H1]",
    "phenol": "c[OX2H1]",
    "ether": "[OD2]([#6])[#6]",
    "carbonyl_any": "[CX3]=[OX1]",
    "aldehyde": "[CX3H1](=O)[#6]",
    "ketone": "[#6][CX3](=O)[#6]",
    "carboxylic_acid": "C(=O)[OX2H1]",
    "ester": "[#6][CX3](=O)[OX2][#6]",

    # Functional groups (nitrogen)
    "amine_primary": "[NX3H2][CX4]",
    "amine_secondary": "[NX3H1]([CX4])[CX4]",
    "amine_tertiary": "[NX3H0]([CX4])([CX4])[CX4]",
    "aniline": "c[NX3]",
    "amide": "[NX3][CX3](=O)",
    "nitrile": "[NX1]#[CX2]",
    "nitro": "[NX3](=O)=O",

    # Halogens
    "halogen_any": "[F,Cl,Br,I]",
    "fluorine": "[F]",
    "chlorine": "[Cl]",
    "bromine": "[Br]",
    "iodine": "[I]",
    "trifluoromethyl": "C(F)(F)F",

    # Sulfur / phosphorus
    "sulfide": "[#6][SX2][#6]",
    "sulfonyl": "[SX4](=O)(=O)",
    "sulfonamide": "[SX4](=O)(=O)[NX3]",
    "phosphate": "[PX4](=O)([OX2])([OX2])[OX2]",

    # Unsaturation
    "alkene": "[CX3]=[CX3]",
    "alkyne": "[CX2]#[CX2]",
    "conjugated_diene": "C=CC=C",

    # Stereochemistry
    "chiral_center": "[C@,C@@,C@H,C@@H]",
}


def compiled_smarts() -> dict[str, Chem.Mol]:
    """Compile SMARTS to RDKit Mol patterns."""
    out: dict[str, Chem.Mol] = {}
    for name, smarts in SMARTS.items():
        m = Chem.MolFromSmarts(smarts)
        if m is None:
            raise ValueError(f"invalid SMARTS for {name}: {smarts}")
        out[name] = m
    return out


def match_concepts(mol: Chem.Mol, patterns: dict[str, Chem.Mol]) -> dict[str, bool]:
    return {name: mol.HasSubstructMatch(p) for name, p in patterns.items()}
