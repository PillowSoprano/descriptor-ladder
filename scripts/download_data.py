"""Fetch and verify the raw datasets.

Castelli is fetched from figshare.  HOIP is not: Dryad serves the archive behind an
anti-scraping challenge, so this script only verifies a manually downloaded copy and
tells you where to get it.  Automating around a bot check is out of scope on purpose.

Usage:  PYTHONPATH=src python3 scripts/download_data.py
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

CASTELLI = {
    "url": "https://ndownloader.figshare.com/files/13284200",
    "path": Path("data/raw/castelli/castelli_perovskites.csv"),
    "size": 30_032_629,
    "cite": "Castelli et al., Energy Environ. Sci. 5, 9034 (2012); figshare 10.6084/m9.figshare.7215323",
}
CASTELLI_JSON = {
    "url": "https://ndownloader.figshare.com/files/13284197",
    "path": Path("data/raw/castelli/castelli_perovskites.json.gz"),
    "size": 4_947_688,
    "cite": CASTELLI["cite"],
}
GB1 = {
    "url": "https://raw.githubusercontent.com/gitter-lab/nn4dms/master/data/gb1/gb1.tsv",
    "path": Path("data/raw/gb1/gb1.tsv"),
    "size": 22_631_969,
    "cite": "Olson et al., Curr. Biol. 24, 2643-2651 (2014); gitter-lab/nn4dms",
}
CMR = {
    "url": "https://wiki.fysik.dtu.dk/cmr-files/cubic_perovskites.db",
    "path": Path("data/raw/cmr/cubic_perovskites.db"),
    "size": 47_226_880,
    "cite": "Castelli et al., Energy Environ. Sci. 5, 9034 (2012); Computational Materials Repository, cmr.fysik.dtu.dk/cubic_perovskites",
}
HOIP = {
    "path": Path("data/raw/hoip/HOIP_cif.tar"),
    "size": 5_030_912,
    "source": "https://datadryad.org/dataset/doi:10.5061/dryad.gq3rg  (file HOIP_cif.tar)",
    "cite": "Kim, Huan, Krishnan & Ramprasad, Sci. Data 4, 170057 (2017)",
}


def fetch(spec: dict) -> None:
    path: Path = spec["path"]
    if path.exists() and path.stat().st_size == spec["size"]:
        print(f"ok      {path} ({path.stat().st_size:,} B)")
        return
    if path.exists():
        print(f"replace {path} (unexpected file size)")
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"fetch   {path} <- {spec['url']}")
    urllib.request.urlretrieve(spec["url"], path)
    got = path.stat().st_size
    if got != spec["size"]:
        raise RuntimeError(f"{path}: expected {spec['size']:,} B, got {got:,} B")
    print(f"        {got:,} B")


def verify_hoip() -> bool:
    path: Path = HOIP["path"]
    if not path.exists():
        print(f"MISSING {path}")
        print("        Dryad serves this behind an anti-scraping challenge and cannot be")
        print("        scripted. Download it in a browser and place it at the path above:")
        print(f"        {HOIP['source']}")
        return False
    got = path.stat().st_size
    if got == HOIP["size"]:
        print(f"ok      {path} ({got:,} B)")
        return True
    print(f"WARNING {path} has {got:,} B; expected {HOIP['size']:,} B")
    return False


def main() -> int:
    fetch(CASTELLI)
    fetch(CASTELLI_JSON)
    fetch(GB1)
    fetch(CMR)
    ok = verify_hoip()
    print("\ncite:")
    print(f"  {HOIP['cite']}")
    print(f"  {CASTELLI['cite']}")
    print(f"  {GB1['cite']}")
    print(f"  {CMR['cite']}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
