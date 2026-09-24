"""Stage 2: text normalisation. Purely rule-based, nothing is learned from data here.

Layers (raw text is always kept):
  *_norm    light normalisation  (SAFE)
  name_core same as name_norm minus legal-suffix words (only used as an extra view)
  addr_nums / addr_post  numeric tokens pulled out of the address (never deleted from addr_norm)

Deliberately NOT done (over-normalisation merges different businesses):
  deleting numbers, expanding ambiguous abbreviations (dr, ct, ste, co), phonetic codes,
  sorting tokens, hand-made transliteration maps, removing landmark phrases.
"""
from __future__ import annotations

import re
import unicodedata

import pandas as pd

_NON_WORD = re.compile(r"[^\w\s]", re.UNICODE)
_WS = re.compile(r"\s+")

# Conservative, clearly unambiguous abbreviations only. Extend from what you SEE in train data.
NAME_ABBREV = {"pvt": "private", "ltd": "limited", "corp": "corporation"}
ADDR_ABBREV = {"rd": "road", "st": "street", "ave": "avenue", "blvd": "boulevard", "ln": "lane"}
LEGAL_SUFFIXES = {
    "private", "limited", "pvt", "ltd", "llc", "inc", "incorporated", "corp", "corporation",
    "company", "co", "llp", "plc", "gmbh", "sarl", "sas", "sa", "lp",
}


def _strip_latin_accents(s: str) -> str:
    """Remove accents from Latin letters only (é -> e). Non-Latin scripts are left untouched."""
    if s.isascii():
        return s
    out = []
    for ch in s:
        d = unicodedata.normalize("NFKD", ch)
        if d and ord(d[0]) < 0x250:
            out.append("".join(c for c in d if not unicodedata.combining(c)))
        else:
            out.append(ch)
    return "".join(out)


def _fold(s: str) -> str:
    s = unicodedata.normalize("NFKC", s or "").casefold()
    s = s.replace("&", " and ")
    s = _strip_latin_accents(s)
    s = _NON_WORD.sub(" ", s).replace("_", " ")
    return _WS.sub(" ", s).strip()


def normalize_name(s: str) -> str:
    return " ".join(NAME_ABBREV.get(t, t) for t in _fold(s).split())


def core_name(name_norm: str) -> str:
    toks = [t for t in name_norm.split() if t not in LEGAL_SUFFIXES]
    return " ".join(toks) or name_norm


def normalize_address(s: str) -> str:
    return " ".join(ADDR_ABBREV.get(t, t) for t in _fold(s).split())


def numeric_tokens(addr_norm: str) -> str:
    return " ".join(t for t in addr_norm.split() if any(ch.isdigit() for ch in t))


def postcode_tokens(addr_norm: str) -> str:
    # generic on purpose (4-6 digits): no country-specific hard-coding, works for unseen countries
    return " ".join(t for t in addr_norm.split() if t.isdigit() and 4 <= len(t) <= 6)


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["name_norm"] = out["business_name"].map(normalize_name)
    out["name_core"] = out["name_norm"].map(core_name)
    out["addr_norm"] = out["business_address"].map(normalize_address)
    out["addr_nums"] = out["addr_norm"].map(numeric_tokens)
    out["addr_post"] = out["addr_norm"].map(postcode_tokens)
    # country stays an open-set string label: casefold only, no mapping, no one-hot
    out["country_norm"] = out["country"].str.strip().str.casefold()
    return out
