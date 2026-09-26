
"""
norm_missingness.py -- EDA §2.2, the first normalization checkpoint.

Adds flags describing what each record actually carries. Nothing is imputed
and nothing is dropped: a record with a valid name but no address is still
matchable, and the matcher needs to know which path to take.

Columns added
-------------
has_address     address is non-blank
addr_degenerate address is non-blank but contains no letters in any script
                (e.g. "425 587") -- numeric tokens still usable, prose is not
addr_usable     has_address & ~addr_degenerate
has_name        name is non-blank and not a sentinel ("NA", "null", ...)
name_short      name is <= 2 chars -- an initialism, NOT degenerate (EDA §1.4:
                98-99.9% of these have a true match). Routes to the
                address-blocks/name-confirms branch.

    from norm_missingness import add_flags, check_flags

    s2 = add_flags(s2)
    check_flags(s2, "S2")
"""

import pandas as pd

from src.normalization_layer.validation import Check
# EDA §1.2: the only sentinel observed is the literal "NA" (2 in S2, 13 in S3).
# The rest are here because they are the usual suspects and cost nothing.
SENTINELS = {"na", "n/a", "null", "none", "nil", "nan", "unknown",
             "not available", "xxx", "test", "-", "--", "."}

# Any letter in any script. Used to tell prose from bare digits/punctuation.
HAS_LETTER = r"[^\W\d_]"

SHORT_NAME_MAX = 2      # EDA §1.4: 3-letter names (Boa, Ace, Zio) are real


# ---------------------------------------------------------------- helpers --

def _clean(series):
    """Stripped view of a column. Not written back -- flags only."""
    return series.fillna("").astype(str).str.strip()


def _is_blank(stripped):
    return stripped == ""


def _is_sentinel(stripped):
    return stripped.str.lower().isin(SENTINELS)


def _has_letter(stripped):
    return stripped.str.contains(HAS_LETTER, regex=True, na=False)


# ------------------------------------------------------------------- core --

def add_flags(df, name_col="business_name", addr_col="business_address"):
    """Return a copy of df with the missingness flags appended.

    Vectorised throughout -- .apply() over 5.3M rows is not viable.
    """
    name = _clean(df[name_col])
    addr = _clean(df[addr_col])

    addr_blank = _is_blank(addr)
    addr_letters = _has_letter(addr)

    out = df.copy()
    out["has_address"] = ~addr_blank
    out["addr_degenerate"] = (~addr_blank) & (~addr_letters)
    out["addr_usable"] = out["has_address"] & ~out["addr_degenerate"]
    out["has_name"] = (~_is_blank(name)) & (~_is_sentinel(name))
    out["name_short"] = out["has_name"] & (name.str.len() <= SHORT_NAME_MAX)
    return out


def unmatchable(df):
    """Records with neither a usable name nor any address.

    EDA §1.2 puts this at 0 / 3 / 4 across S1 / S2 / S3 -- if it comes back
    large, something upstream is wrong.
    """
    return df[~df["has_name"] & ~df["has_address"]]


# ------------------------------------------------------------- checkpoint --

FLAGS = ["has_address", "addr_degenerate", "addr_usable", "has_name", "name_short"]


def check_flags(df, label, n_rows=None, strict=True):
    """Validate the missingness checkpoint against EDA expectations."""
    chk = Check(f"missingness/{label}", strict=strict)

    if n_rows is not None:
        chk.equal(len(df), n_rows, "row count preserved")

    for col in FLAGS:
        chk.ok(col in df.columns, f"{col} present")
        if col in df.columns:
            chk.is_bool(df[col])
            chk.no_nulls(df[col])

    # Raw columns must survive untouched -- normalization is additive.
    for col in ("entity_id", "business_name", "business_address", "country"):
        chk.no_nulls(df[col], f"{col}: raw column has no nulls")

    # The failure mode that manufactures false merges: blanks that became
    # the string "nan" and then fuzzy-match each other.
    chk.absent(df["business_address"], {"nan", "none", "null"},
               "address: no coerced-null strings")

    # Internal consistency.
    chk.equal(int((df["addr_usable"] & ~df["has_address"]).sum()), 0,
              "addr_usable implies has_address")
    chk.equal(int((df["name_short"] & ~df["has_name"]).sum()), 0,
              "name_short implies has_name")

    # EDA §1.2: ~3% blank addresses in S2/S3, ~0 in S1.
    chk.rate(~df["has_address"], 0.0, 0.06, "blank-address rate plausible")

    # EDA §1.2: unmatchable records are a handful, not a population.
    n_bad = len(unmatchable(df))
    chk.between(n_bad, 0, 100, "unmatchable records are negligible")

    chk.report()
    return chk


def summary(df, label):
    """One-line profile for the notebook."""
    n = len(df)
    return pd.Series({
        "rows": n,
        "blank_address": int((~df["has_address"]).sum()),
        "blank_address_pct": round(float((~df["has_address"]).mean()) * 100, 2),
        "addr_degenerate": int(df["addr_degenerate"].sum()),
        "no_name": int((~df["has_name"]).sum()),
        "name_short": int(df["name_short"].sum()),
        "unmatchable": len(unmatchable(df)),
    }, name=label)