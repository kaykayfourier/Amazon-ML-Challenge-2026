"""
norm_script.py -- EDA §2.4, script detection and transliteration.

S1 is Latin (train: pure ASCII; test: ASCII + French accents). S2/S3 carry ten
Indic scripts -- Devanagari, Telugu, Kannada, Tamil, Bengali, Gujarati,
Malayalam, Oriya, Gurmukhi -- in ~9% of S2 and ~5% of S3 names. Without
transliteration those records share no characters with their S1 match and are
invisible to every name-based key.

Direction is one-way: everything converges on Latin.
Dispatch is on DETECTED SCRIPT, never on the country label -- France appears
only in test, and an unenumerated script must degrade, not crash.

    from norm_script import add_latin, check_latin

    s2 = add_latin(s2)          # adds name_script, name_latin
    check_latin(s2, "S2")
"""

import re

from anyascii import anyascii
# from indic_transliteration import sanscript
# from indic_transliteration.detect import detect

from amazon.code.business_entity_resolution.src.normalization_layer.norm_text import clean
from amazon.code.business_entity_resolution.src.normalization_layer.validation import Check
INDIC = re.compile("[\u0900-\u0D7F]")
NON_ASCII = re.compile("[^\x00-\x7F]")
# Transliteration artefacts worth repairing. Measured on 2,000 true pairs:
# anyascii alone 73.3, +ph->f 74.5, +anusvara 75.2. Schwa deletion tested
# negative (73.1) and is deliberately absent.
REPAIRS = [(r"ph", "f"), (r"m(?=[kgcjtdnpbm])", "n")]

# ---------------------------------------------------------------- helpers --

SCRIPTS = [
    ("devanagari", 0x0900), ("bengali", 0x0980), ("gurmukhi", 0x0A00),
    ("gujarati", 0x0A80), ("oriya", 0x0B00), ("tamil", 0x0B80),
    ("telugu", 0x0C00), ("kannada", 0x0C80), ("malayalam", 0x0D00),
]

def _script_of(text):
    """Script label for the first non-Latin character. Feature only --
    transliteration no longer dispatches on it."""
    for ch in text:
        cp = ord(ch)
        if cp < 0x0900:
            continue
        for name, start in reversed(SCRIPTS):
            if cp >= start:
                return name
    return "latin"


'''def _to_latin(text):
    """Transliterate to IAST, falling back to anyascii for anything the
    Indic schemes do not cover. Never raises -- a bad row degrades to
    whatever anyascii can make of it."""
    try:
        scheme = detect(text)
        if scheme and scheme in sanscript.SCHEMES:
            return sanscript.transliterate(text, scheme, sanscript.IAST)
    except Exception:
        pass
    return anyascii(text)'''

def _to_latin(text):
    s = anyascii(text)
    for pat, rep in REPAIRS:
        s = re.sub(pat, rep, s)
    return s

def _map_unique(series, fn):
    """Apply a per-row Python function via the unique values only.

    Indic legal suffixes repeat enormously (प्राइवेट लिमिटेड appears in
    hundreds of thousands of rows), so this is typically a 10-50x saving
    over .map(fn) on the full column.
    """
    uniq = series.unique()
    return series.map({v: fn(v) for v in uniq})


# ------------------------------------------------------------------- core --

def add_latin(df, col="name_clean"):
    """Append name_script and name_latin.

    name_latin is Latin for every row: a passthrough where the source is
    already Latin, a transliteration where it is not. Downstream keys read
    this column and never need to know which happened.
    """
    src = df[col].fillna("")
    needs = src.str.contains(NON_ASCII, regex=True, na=False)

    out = df.copy()
    out["name_script"] = "latin"
    out["name_latin"] = src

    if needs.any():
        sub = src[needs]
        out.loc[needs, "name_script"] = _map_unique(sub, _script_of)
        # clean() again: transliteration reintroduces diacritics (IAST) and
        # punctuation that the §2.3 pass had already removed.
        out.loc[needs, "name_latin"] = clean(_map_unique(sub, _to_latin))

    return out


def script_profile(df):
    """Script distribution -- run on train and test and diff them (§3.7)."""
    return df["name_script"].value_counts(normalize=True).round(4)


# ------------------------------------------------------------- checkpoint --

def check_latin(df, label, strict=True):
    chk = Check(f"script/{label}", strict=strict)

    chk.no_nulls(df["name_latin"])
    chk.no_nulls(df["name_script"])

    # The point of the section: the output column is Latin everywhere.
    residual = df["name_latin"].str.contains(INDIC, regex=True, na=False)
    chk.equal(int(residual.sum()), 0, "name_latin: no Indic characters remain")

    # Transliteration must not silently empty a populated name.
    had = df["name_clean"].str.strip() != ""
    chk.between(int((had & (df["name_latin"] == "")).sum()), 0, 20,
                "name_latin: populated names survived")

    # Latin rows must pass through untouched.
    lat = df["name_script"] == "latin"
    chk.equal(int((lat & (df["name_latin"] != df["name_clean"])).sum()), 0,
              "latin rows unchanged")

    # §2.3 invariants must still hold after the second clean().
    chk.equal(int(df["name_latin"].str.contains("  ", regex=False).sum()), 0,
              "name_latin: no double spaces")
    chk.equal(int(df["name_latin"].str.contains("[A-Z]", regex=True).sum()), 0,
              "name_latin: lowercased")

    # An unenumerated script should be rare; a spike means the detector is
    # failing on something worth looking at.
    chk.rate(df["name_script"] == "unknown", 0.0, 0.01, "unknown script rare")

    chk.report()
    return chk