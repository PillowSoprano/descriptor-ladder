"""The Emery-Wolverton ABO3 grid (Sci. Data 4, 170153, 2017) in the schema of the Castelli frame.

73 A-site by 73 B-site elements, one DFT calculation each (the lowest-energy of four distortions),
so every composition is O3 and the anion site has a single level. Formation energy is per atom and
volume is per formula unit (five atoms), as for the Castelli grid.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SOURCE = Path("data/raw/emery_wolverton/abo3_perovskites.csv")


def load_emery_grid() -> pd.DataFrame:
    raw = pd.read_csv(SOURCE)
    energy = pd.to_numeric(raw["Formation energy [eV/atom]"], errors="coerce")
    volume = pd.to_numeric(raw["Volume per atom [A^3/atom]"], errors="coerce") * 5.0
    return pd.DataFrame({
        "sample_id": "ew-" + raw.A + "-" + raw.B,
        "formula": raw["Chemical formula"],
        "site_A": raw.A.astype(str), "site_B": raw.B.astype(str), "site_X": "O3",
        "site_A_kind": "element", "site_B_kind": "element", "site_X_kind": "anion",
        "prototype_id": "ABO3-lowest-distortion", "energy_functional": "OQMD",
        "relaxation_known": True, "volume_change_frac": np.nan, "rmsd_init_final": np.nan,
        "b_x_coordination": np.nan, "octahedral_distortion": np.nan,
        "heat_of_formation": energy, "volume_per_fu": volume, "magnetic_moment": np.nan,
    })
