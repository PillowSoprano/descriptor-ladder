"""Fetch the ABX3 perovskites of the Materials Project with their structures.

Materials Project labels each structure with a mineral type (robocrystallographer). The script
takes every entry labelled as a cubic or orthorhombic perovskite, keeps those whose anonymous
formula is ABC3, and stores their identifiers, formulas, flags and structures together with the
database version. Requires an API key in the environment variable MP_API_KEY.

Writes data/raw/mp_perovskites/mp_abx3_perovskites.json.gz and provenance.json.

Usage:  python3 scripts/download_mp_perovskites.py
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

from mp_api.client import MPRester

OUT = Path("data/raw/mp_perovskites")
FIELDS = ["material_id", "formula_pretty", "formula_anonymous", "theoretical",
          "energy_above_hull", "symmetry", "structure"]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with MPRester(mute_progress_bars=True) as mpr:
        labelled = mpr.materials.robocrys.search(keywords=["perovskite"])
        mineral = {str(d.material_id): str(d.condensed_structure.mineral.type) for d in labelled}
        docs = mpr.materials.summary.search(material_ids=sorted(mineral), fields=FIELDS)
        version = mpr.get_database_version()
    rows = []
    for d in docs:
        if d.formula_anonymous != "ABC3":
            continue
        mid = str(d.material_id)
        rows.append({"material_id": mid, "formula": d.formula_pretty, "mineral": mineral[mid],
                     "theoretical": bool(d.theoretical), "e_above_hull": d.energy_above_hull,
                     "spacegroup": d.symmetry.symbol, "structure": d.structure.as_dict()})
    rows.sort(key=lambda r: r["material_id"])
    path = OUT / "mp_abx3_perovskites.json.gz"
    with gzip.open(path, "wt") as fh:
        json.dump(rows, fh)
    prov = {"source": "Materials Project, https://materialsproject.org",
            "licence": "CC BY 4.0",
            "database_version": version,
            "selection": "robocrystallographer mineral type containing 'perovskite', anonymous formula ABC3",
            "labelled_entries": len(mineral), "abc3_entries": len(rows)}
    (OUT / "provenance.json").write_text(json.dumps(prov, indent=2) + "\n")
    print(json.dumps(prov, indent=2))


if __name__ == "__main__":
    main()
