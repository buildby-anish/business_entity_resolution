"""Stage 12: write the two PS output files, and a local self-check (the official validator is still the judge)."""
from __future__ import annotations

import csv
from pathlib import Path

MATCH_COLS = ["source1_entity_id", "matched_entity_ids"]
CAND_COLS = ["source1_entity_id", "candidate_entity_ids"]


def write_id_lists(path, s1_ids, mapping: dict, columns) -> None:
    """Tab-separated, ID lists comma-separated with no quoting, one row per S1 id, empty list -> empty cell."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t", quoting=csv.QUOTE_NONE, lineterminator="\n", escapechar="\\")
        w.writerow(columns)
        for s in s1_ids:
            w.writerow([s, ",".join(sorted(set(mapping.get(s, []))))])


def _read(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split("\t")
        for line in f:
            parts = line.rstrip("\n").split("\t")
            rows.append((parts[0], parts[1] if len(parts) > 1 else ""))
    return header, rows


def self_check(matching_path, candidate_path, s1_ids, s2_ids, s3_ids) -> list[str]:
    """Mirrors the PS rules. Returns a list of problems (empty list = looks fine)."""
    issues = []
    s1_ids, valid = set(s1_ids), set(s2_ids) | set(s3_ids)
    cands = {}
    for path, cols, is_match in ((candidate_path, CAND_COLS, False), (matching_path, MATCH_COLS, True)):
        header, rows = _read(path)
        name = Path(path).name
        if header != cols:
            issues.append(f"{name}: header {header} != {cols}")
        seen = [r[0] for r in rows]
        if len(seen) != len(set(seen)):
            issues.append(f"{name}: duplicate source1_entity_id rows")
        if set(seen) != s1_ids:
            issues.append(f"{name}: S1 ids differ from test_source1 (missing={len(s1_ids - set(seen))}, "
                          f"extra={len(set(seen) - s1_ids)})")
        for s1, lst in rows:
            ids = [x for x in lst.split(",") if x]
            if len(ids) != len(set(ids)):
                issues.append(f"{name}: duplicate ids inside list of {s1}")
            bad = [x for x in ids if x not in valid]
            if bad:
                issues.append(f"{name}: {s1} has ids that are not S2/S3 test ids, e.g. {bad[:3]}")
            if not is_match:
                cands[s1] = set(ids)
            elif not set(ids) <= cands.get(s1, set()):
                issues.append(f"{name}: {s1} has matches that are not in candidate_pairs.tsv")
    return issues
