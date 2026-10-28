# Descriptor ladder: code and data

This repository regenerates every table and supplementary figure of the manuscript "Which
compositions have been computed decides how chemical descriptors should enter a model" and its
Supporting Information.

## Layout

- `src/prfield/`: designs, descriptors, models and metrics.
- `scripts/`: one or more scripts per Supporting Information table; the file map in the last
  section of the Supporting Information names the script behind each table.
- `data/interim/`: the Castelli perovskite grid as used.
- `data/raw/emery_wolverton/`: the Emery-Wolverton ABO3 table, unchanged, with its source and licence.
- `data/raw/mp_perovskites/`: the ABX3 perovskite entries of the Materials Project, unchanged, with
  their database version, source and licence.
- `reports/`: every output file behind the tables, so results can be checked without rerunning.
  Intermediate curve files larger than 25 MB are left out; the commands of the table that uses
  them regenerate them first.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python scripts/download_data.py
```

`download_data.py` fetches the GB1 double-mutant scores and the raw Castelli files from their public
sources. The derived Castelli tables in `data/interim/` are included, so the materials analyses run
without the download.

## Reproducing a table

```bash
.venv/bin/python reproduce.py --list      # every table and the commands behind it
.venv/bin/python reproduce.py S4          # one table
.venv/bin/python reproduce.py --all       # every table
```

Runs that take longer than a few minutes are marked in `--list`.

## Data sources

- Castelli et al., Energy Environ. Sci. 5, 9034 (2012); figshare doi:10.6084/m9.figshare.7215323 (MIT).
- Emery and Wolverton, Sci. Data 4, 170153 (2017); figshare doi:10.6084/m9.figshare.5334142 (CC BY 4.0).
- Horton et al., Nat. Mater. 24, 1522 (2025); Materials Project, database version 2026.09.28 (CC BY 4.0).
- Olson et al., Curr. Biol. 24, 2643 (2014); GB1 double-mutant scan.
- Elemental properties from pymatgen.

## Licence

The code and the outputs in `reports/` are released under the MIT licence in `LICENSE`. The
Castelli table in `data/interim/` is derived from records released under the MIT licence, and the
Emery-Wolverton table and the Materials Project entries keep their CC BY 4.0 licence, as does the
composition list `reports/mp_perovskite_compositions.csv` derived from the latter; a `SOURCE.md`
beside each data folder states its source.
