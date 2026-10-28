"""Canonical species labels.

Every source's naming is collapsed onto a single vocabulary here. A species spelled two ways would
split the substitution graph into disconnected halves, so unmapped labels raise.
"""

from __future__ import annotations

# --- A site: organic cations (HOIP) -------------------------------------------------
# name -> (canonical label, chemical formula as written by the source)
ORGANIC_CATIONS: dict[str, tuple[str, str]] = {
    "Ammonium":            ("NH4", "NH4"),
    "Methylammonium":      ("MA", "CH3NH3"),
    "Dimethylammonium":    ("DMA", "(CH3)2NH2"),
    "Trimethylammonium":   ("TMA", "(CH3)3NH"),
    "Tetramethylammonium": ("TetMA", "(CH3)4N"),
    "Ethylammonium":       ("EA", "CH3CH2NH3"),
    "Propylammonium":      ("PA", "CH3CH2CH2NH3"),
    "Isopropylammonium":   ("iPA", "(CH3)2CHNH3"),
    "Butylammonium":       ("BA", "CH3CH2CH2CH2NH3"),
    "Hydroxylammonium":    ("HxA", "OHNH3"),
    "Formamidinium":       ("FA", "NH2CHNH2"),
    "Acetamidinium":       ("AcA", "NH2C(CH3)NH2"),
    "Hydrazinium":         ("HyA", "NH2NH3"),
    "Guanidinium":         ("GUA", "C(NH2)3"),
    "Azetidinium":         ("AZE", "(CH2)3NH2"),
    "Imidazolium":         ("IMI", "C3N2H5"),
}

# --- B site: group-IV cations --------------------------------------------------------
B_CATIONS: dict[str, str] = {
    "Germanium": "Ge", "Tin": "Sn", "Lead": "Pb",
    "Ge": "Ge", "Sn": "Sn", "Pb": "Pb",
}

# --- X site: halides -----------------------------------------------------------------
X_ANIONS: dict[str, str] = {
    "Fluoride": "F", "Chloride": "Cl", "Bromide": "Br", "Iodide": "I",
    "F": "F", "Cl": "Cl", "Br": "Br", "I": "I",
    "O": "O", "N": "N", "S": "S",  # Castelli also has O/N/S-containing anion combinations
}


class UnknownSpecies(KeyError):
    pass


def canon_A(name: str) -> tuple[str, str]:
    """-> (label, kind).  kind is 'molecule' for organic cations, 'element' otherwise."""
    name = name.strip()
    if name in ORGANIC_CATIONS:
        return ORGANIC_CATIONS[name][0], "molecule"
    if name.isalpha() and len(name) <= 2:  # elemental A site (Castelli)
        return name, "element"
    raise UnknownSpecies(f"unmapped A-site species: {name!r}")


def canon_B(name: str) -> tuple[str, str]:
    name = name.strip()
    if name in B_CATIONS:
        return B_CATIONS[name], "element"
    if name.isalpha() and len(name) <= 2:
        return name, "element"
    raise UnknownSpecies(f"unmapped B-site species: {name!r}")


def canon_X(name: str) -> tuple[str, str]:
    """X may be a single halide or a mixed anion string like 'O2N' (Castelli)."""
    name = name.strip()
    if name in X_ANIONS:
        return X_ANIONS[name], "element"
    if any(ch.isdigit() for ch in name) or len(name) > 2:
        return name, "molecule"   # composite anion sublattice, treated as one X level
    raise UnknownSpecies(f"unmapped X-site species: {name!r}")


def organic_formula(label: str) -> str | None:
    for name, (lab, formula) in ORGANIC_CATIONS.items():
        if lab == label:
            return formula
    return None
