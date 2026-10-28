"""Canonical sample schema: the data dictionary in executable form.

Every downstream module (substitution graph, splits, models, reports) reads this schema. Ingest
adapters map a source dataset onto it, and a source that cannot fill a required column raises.

A sample is the tuple z = (a, b, x, s): the A-, B- and X-site species and a structural context s
(`prototype_id`). Two samples form a substitution pair when they agree on every coordinate but the
substituted one and share a structural context; see `prfield.graph.typed_graph`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

SITES = ("A", "B", "X")

# ---------------------------------------------------------------------------
# Column contract
# ---------------------------------------------------------------------------

# Identity ------------------------------------------------------------------
ID_COLUMNS = {
    "sample_id": "string",      # globally unique: f"{source}:{native_id}"
    "source": "string",         # hoip | castelli | leadfree1117
    "native_id": "string",      # identifier inside the source dataset
    "formula": "string",        # reduced formula as reported by the source
}

# Typed composition ---------------------------------------------------------
# Species labels are *canonical* strings from prfield.ingest.canonical, not raw source
# strings: "MA" and "methylammonium" must collapse to one label or edges silently split.
SITE_COLUMNS = {
    "site_A": "string",
    "site_B": "string",
    "site_X": "string",
    "site_A_kind": "string",    # element | molecule
    "site_B_kind": "string",
    "site_X_kind": "string",
}

# Structural context s ------------------------------------------------------
# `prototype_id` is the equivalence class within which substitutions are considered
# comparable.  For Castelli every entry is a relaxed cubic ABX3, so the prototype is
# constant; for HOIP it must be derived from the structure (space group + connectivity),
# NOT assumed.  `prototype_source` records how it was assigned so that audit A1 can
# distinguish "asserted by the dataset" from "derived by us".
CONTEXT_COLUMNS = {
    "prototype_id": "string",
    "prototype_source": "string",      # source_declared | derived_spacegroup | derived_connectivity | unknown
    "spacegroup_symbol": "string",
    "spacegroup_number": "Int64",
    "n_formula_units": "Int64",
}

# Relaxation metadata -------------------------------------------------------
# The kill-switch columns for audit A1.  If a source provides no way to populate these,
# `relaxation_known` is False and every edge built from it is provisional.
RELAXATION_COLUMNS = {
    "relaxation_known": "boolean",
    "volume_per_fu": "float64",          # A^3 per formula unit, relaxed
    "volume_change_frac": "float64",     # (V_relaxed - V_init) / V_init, if init known
    "rmsd_init_final": "float64",        # A, matched init->final displacement
    "max_disp_init_final": "float64",    # A
    "b_x_coordination": "Int64",         # B-site coordination number, relaxed
    "bx_bond_mean": "float64",           # A, mean B-X bond length
    "octahedral_distortion": "float64",  # dimensionless, e.g. Baur delta index
    "tilt_angle_mean": "float64",        # degrees, B-X-B deviation from 180
}

# Properties ----------------------------------------------------------------
# Stored long-form-friendly but kept wide for convenience.  Every property carries the
# functional it came from: differences across functionals are not comparable and audit
# A1/A2 must be able to refuse to build such an edge.
PROPERTY_COLUMNS = {
    "band_gap": "float64",
    "band_gap_kind": "string",         # direct | indirect | unspecified
    "band_gap_functional": "string",   # PBE | GLLB-SC | HSE06 | ...
    # A second functional for the same property, when the source provides one.  Not a
    # nuisance column: Delta_HSE vs Delta_GGA is the cheapest available test of whether a
    # recovered response field is an artifact of the functional.
    "band_gap_secondary": "float64",
    "band_gap_secondary_functional": "string",
    "relative_energy": "float64",   # eV per atom, within-composition reference
    "relative_energy_alt": "float64",  # second reference convention, if the source has one
    "heat_of_formation": "float64", # eV per atom
    "atomization_energy": "float64",# eV per atom
    "dielectric_constant": "float64",     # total
    "dielectric_electronic": "float64",
    "dielectric_ionic": "float64",
    "refractive_index": "float64",
    "density": "float64",           # g/cm^3
    "magnetic_moment": "float64",   # Bohr magneton per reported unit cell; magnitude
    "energy_functional": "string",
}

# Provenance ----------------------------------------------------------------
PROVENANCE_COLUMNS = {
    "dft_code": "string",
    "pseudopotentials": "string",
    "structure_blob": "string",        # serialized structure (CIF text or JSON), optional
    "ingest_version": "string",
}

SCHEMA: dict[str, str] = {
    **ID_COLUMNS,
    **SITE_COLUMNS,
    **CONTEXT_COLUMNS,
    **RELAXATION_COLUMNS,
    **PROPERTY_COLUMNS,
    **PROVENANCE_COLUMNS,
}

REQUIRED_COLUMNS = (
    "sample_id",
    "source",
    "native_id",
    "site_A",
    "site_B",
    "site_X",
    "prototype_id",
    "prototype_source",
    "relaxation_known",
)

PROPERTY_NAMES = (
    "band_gap",
    "band_gap_secondary",
    "relative_energy",
    "heat_of_formation",
    "atomization_energy",
    "dielectric_constant",
    "dielectric_electronic",
    "dielectric_ionic",
    "refractive_index",
    "magnetic_moment",
)

# The functional column that governs each property.  Used to refuse cross-functional edges.
PROPERTY_FUNCTIONAL = {
    "band_gap": "band_gap_functional",
    "band_gap_secondary": "band_gap_secondary_functional",
    "relative_energy": "energy_functional",
    "heat_of_formation": "energy_functional",
    "atomization_energy": "energy_functional",
    "dielectric_constant": "energy_functional",
    "dielectric_electronic": "energy_functional",
    "dielectric_ionic": "energy_functional",
    "refractive_index": "energy_functional",
    "magnetic_moment": "energy_functional",
}


@dataclass
class IngestReport:
    """What an adapter could and could not fill.  Printed into the data card."""

    source: str
    n_rows: int
    missing_columns: list[str] = field(default_factory=list)
    relaxation_available: bool = False
    prototype_strategy: str = "unknown"
    notes: list[str] = field(default_factory=list)
    dropped: dict[str, int] = field(default_factory=dict)

    def render(self) -> str:
        lines = [f"## ingest report — {self.source}", f"- rows: {self.n_rows}"]
        lines.append(f"- prototype assignment: {self.prototype_strategy}")
        lines.append(f"- relaxation metadata available: {self.relaxation_available}")
        if self.missing_columns:
            lines.append(f"- unfilled columns: {', '.join(sorted(self.missing_columns))}")
        for reason, n in sorted(self.dropped.items()):
            lines.append(f"- dropped {n} rows: {reason}")
        for note in self.notes:
            lines.append(f"- note: {note}")
        return "\n".join(lines)


def empty_frame() -> pd.DataFrame:
    return pd.DataFrame({col: pd.Series(dtype=dt) for col, dt in SCHEMA.items()})


def conform(df: pd.DataFrame, source: str) -> pd.DataFrame:
    """Add missing schema columns as nulls, coerce dtypes, order columns, sanity-check ids."""
    out = df.copy()
    for col, dtype in SCHEMA.items():
        if col not in out.columns:
            na = np.nan if dtype.startswith("float") else pd.NA
            out[col] = pd.Series([na] * len(out), index=out.index, dtype=dtype)
        else:
            try:
                out[col] = out[col].astype(dtype)
            except (TypeError, ValueError) as exc:  # keep the raw values, flag loudly
                raise TypeError(f"[{source}] column {col!r} cannot be cast to {dtype}: {exc}") from exc
    out = out[list(SCHEMA)]
    validate(out, source)
    return out


def validate(df: pd.DataFrame, source: str = "?") -> None:
    missing = [c for c in REQUIRED_COLUMNS if df[c].isna().any()]
    if missing:
        raise ValueError(f"[{source}] required columns contain nulls: {missing}")
    if df["sample_id"].duplicated().any():
        dupes = df.loc[df["sample_id"].duplicated(), "sample_id"].head(5).tolist()
        raise ValueError(f"[{source}] duplicate sample_id, e.g. {dupes}")
    if not df[[f"site_{s}" for s in SITES]].map(lambda v: isinstance(v, str) and v != "").all().all():
        raise ValueError(f"[{source}] site labels must be non-empty strings")


def available_properties(df: pd.DataFrame) -> list[str]:
    return [p for p in PROPERTY_NAMES if p in df.columns and df[p].notna().any()]


def composition_key(row: "pd.Series | dict", include_prototype: bool = True) -> tuple:
    key = (row["site_A"], row["site_B"], row["site_X"])
    return key + (row["prototype_id"],) if include_prototype else key


def site_of(col: str) -> Optional[str]:
    for s in SITES:
        if col == f"site_{s}":
            return s
    return None
