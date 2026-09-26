"""
norm_keys.py -- EDA §2.5 and §2.6: name and address decomposition.

Produces the columns blocking consumes. Everything is a SET, not an ordered
field: EDA §1.5 established that address components sit at no fixed index, so
positional parsing is unsafe.

Columns added
-------------
name_core        name_latin minus legal suffixes (business-type words KEPT --
                 stripping them collapses "shree medical center" to "shree")
name_tokens      sorted token set of name_core, order-invariant
suffix_set       the legal suffixes that were removed, as a comparison feature
initialism       first letters of name_core tokens -- the key for the ≤2-char
                 slice (EDA §1.4), matched against short S2/S3 names
addr_numbers     ALL numeric tokens, any position, compounds preserved
addr_tokens      alphabetic address tokens, stopworded
addr_locality    address segments matching the corpus locality vocabulary

    from norm_keys import add_keys, check_keys, build_locality_vocab

    vocab = build_locality_vocab([s1, s2, s3, st1, st2, st3])
    s2 = add_keys(s2, vocab)
    check_keys(s2, "S2")
"""

import re
from collections import Counter

import pandas as pd

from src.normalization_layer.validation import Check

# Legal forms only -- these come OUT of the name into suffix_set.
# Derived empirically from trailing-token counts over train+test, name_latin
# (so transliterated Indic forms and French forms are included).
LEGAL = {
    # english
    "limited", "ltd", "llc", "inc", "incorporated", "llp", "lp", "corp",
    "corporation", "co", "company", "plc", "pc", "pllc", "pvt", "private",
    # french
    "sarl", "sas", "sasu", "sa", "eurl", "sci", "scp", "privee", "prive",
    # transliterated indic
    "elelpi", "li", "pra",
    # injected typo variants
    "limitet", "limirrd", "limted", "limtied", "limied", "limlted",
    "privat", "privte", "privated", "privite",
}

# Business-type words are deliberately NOT here -- EDA §1.7.

CONNECTORS = {"and", "the", "of", "de", "du", "la", "le", "les", "des", "et"}

ADDR_STOP = {
    "near", "opp", "opposite", "behind", "beside", "adjacent", "front",
    "no", "nos", "floor", "flat", "plot", "shop", "block", "building",
    "road", "rd", "street", "st", "ave", "avenue", "lane", "ln", "drive",
    "dr", "court", "ct", "circle", "way", "rue", "city", "cty", "co",
    "unit", "suite", "ste", "po", "box", "sector", "phase", "colony",
} | CONNECTORS

NUM = re.compile(r"\d[\w/_-]*")          # 229, 118_103, 8_2_644, 54th
WORD = re.compile(r"[^\W\d_]+")


# ---------------------------------------------------------------- helpers --

def _tokens(series):
    return series.fillna("").str.split()


def _join_sorted(lists):
    return lists.map(lambda t: " ".join(sorted(set(t))))


# --------------------------------------------------------- §2.5  the name --

def add_name_keys(df, col="name_latin"):
    toks = _tokens(df[col])

    core = toks.map(lambda t: [w for w in t if w not in LEGAL])
    suffix = toks.map(lambda t: sorted({w for w in t if w in LEGAL}))
    sig = core.map(lambda t: [w for w in t if w not in CONNECTORS and len(w) > 1])

    out = df.copy()
    out["name_core"] = core.map(" ".join)
    out["name_tokens"] = _join_sorted(core)
    out["suffix_set"] = suffix.map(" ".join)
    out["initialism"] = sig.map(lambda t: "".join(w[0] for w in t))
    return out


# ------------------------------------------------------ §2.6  the address --

def build_locality_vocab(frames, col="addr_clean", top=3000, min_count=200):
    """Locality vocabulary from ALL comma-positions, learned from the corpus.

    EDA §1.8 + §1.5: cities and states appear at any index, so the trailing
    segment is not a reliable source. Learned rather than hardcoded so France
    is covered without a code change.

    NOTE: addr_clean has already had commas stripped by §2.3, so this counts
    whole-token frequency. Pass the RAW address column to get true segments.
    """
    c = Counter()
    for df in frames:
        for part in df[col].fillna(""):
            c.update(WORD.findall(part))
    return {w for w, n in c.most_common(top) if n >= min_count} - ADDR_STOP


def add_addr_keys(df, vocab, col="addr_clean"):
    s = df[col].fillna("")

    out = df.copy()
    out["addr_numbers"] = s.map(lambda a: " ".join(sorted(set(NUM.findall(a)))))
    words = s.map(WORD.findall)
    out["addr_tokens"] = _join_sorted(
        words.map(lambda t: [w for w in t if w not in ADDR_STOP and len(w) > 2]))
    out["addr_locality"] = _join_sorted(
        words.map(lambda t: [w for w in t if w in vocab]))
    return out


# ------------------------------------------------------------------- core --

def add_keys(df, vocab):
    return add_addr_keys(add_name_keys(df), vocab)


# ------------------------------------------------------------- checkpoint --

KEY_COLS = ["name_core", "name_tokens", "suffix_set", "initialism",
            "addr_numbers", "addr_tokens", "addr_locality"]


def check_keys(df, label, strict=True):
    chk = Check(f"keys/{label}", strict=strict)

    for c in KEY_COLS:
        chk.ok(c in df.columns, f"{c} present")
        chk.no_nulls(df[c])

    # Stripping suffixes must not erase the whole name.
    had = df["name_latin"].str.strip() != ""
    chk.rate((had & (df["name_core"] == "")), 0.0, 0.01,
             "name_core: suffix strip rarely empties the name")

    # Initialism is the ≤2-char slice's only name signal (EDA §1.4).
    chk.rate(df["initialism"] != "", 0.95, 1.0, "initialism populated")

    # Numeric tokens are the recovery mechanism for the hard 12.5% (§1.12).
    has_digit = df["addr_clean"].str.contains(r"\d", regex=True, na=False)
    chk.equal(int((has_digit & (df["addr_numbers"] == "")).sum()), 0,
              "addr_numbers: no digits lost")

    # Compound numbers must survive as single tokens (614 1_2, 118_103).
    chk.rate(df["addr_numbers"].str.contains("_", na=False), 0.0, 0.5,
             "compound numbers present but not universal")

    # Token sets must be sorted and deduplicated.
    samp = df["name_tokens"].head(1000)
    chk.equal(int(sum(t.split() != sorted(set(t.split())) for t in samp)), 0,
              "name_tokens: sorted and unique")

    chk.report()
    return chk


def profile(df, label):
    return pd.Series({
        "empty_name_core": int((df["name_core"] == "").sum()),
        "has_suffix": int((df["suffix_set"] != "").sum()),
        "mean_name_tokens": round(df["name_tokens"].str.split().str.len().mean(), 2),
        "empty_addr_numbers": int((df["addr_numbers"] == "").sum()),
        "empty_addr_locality": int((df["addr_locality"] == "").sum()),
        "mean_addr_tokens": round(df["addr_tokens"].str.split().str.len().mean(), 2),
    }, name=label)