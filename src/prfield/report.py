"""Report writing that does not destroy hand-written analysis.

Most reports here are half machine and half human: a script emits the tables, and the reading
of those tables is written afterwards by a person. Naively rewriting the file on a rerun
deletes the second half, which is the more expensive one.

`write_report` therefore writes the generated part and re-appends whatever followed the marker
in the previous version. The marker makes the split explicit in the file itself, so a reader
can always tell which half is which -- the alternative, a blanket claim that everything is
generated, was untrue and is exactly the sort of provenance error that survives into a
manuscript.
"""

from __future__ import annotations

from pathlib import Path

MARKER = "<!-- ANALYSIS: hand-written below; preserved across reruns -->"

HEADER = (
    "<!-- GENERATED: everything above the ANALYSIS marker is written by the script named in\n"
    "     this file's first paragraph and is overwritten on every rerun. -->"
)


def split_analysis(text: str) -> tuple[str, str]:
    """(generated part, hand-written part including the marker)."""
    idx = text.find(MARKER)
    if idx == -1:
        return text, ""
    return text[:idx], text[idx:]


def write_report(path: str | Path, generated: str) -> None:
    """Write `generated`, preserving any hand-written section from the existing file."""
    path = Path(path)
    preserved = ""
    if path.exists():
        _, preserved = split_analysis(path.read_text())
    body = generated.rstrip() + "\n"
    if not body.lstrip().startswith("<!--"):
        body = HEADER + "\n\n" + body
    if preserved:
        body = body + "\n\n" + preserved.lstrip()
    else:
        body = body + "\n\n" + MARKER + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body)


def has_analysis(path: str | Path) -> bool:
    p = Path(path)
    if not p.exists():
        return False
    _, tail = split_analysis(p.read_text())
    return len(tail.replace(MARKER, "").strip()) > 0
