"""
normalize_pipeline.py -- end-to-end normalization, raw TSV to blocking keys.

    python -m amazon.code.business_entity_resolution.src.normalize_pipeline

    # or from a notebook
    from amazon.code.business_entity_resolution.src.normalize_pipeline import run
    run()
    run(splits=("train",), refresh=False)     # skip finished stages

Two phases, because the locality vocabulary is corpus-wide and cannot be built
until every frame has been cleaned:

    phase 1   flags -> clean -> transliterate      -> *_norm.parquet
    vocab     frequency count over all six frames  -> locality_vocab.json
    phase 2   name keys + address keys             -> *_keys.parquet

Frames are read and written one at a time. Holding all six in memory is ~21M
rows and unnecessary.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import polars as pl

from amazon.code.business_entity_resolution.src.dataio import CACHE, load_source
from amazon.code.business_entity_resolution.src.normalization_layer.norm_keys import (
    add_keys, build_locality_vocab, check_keys, profile,
)
from amazon.code.business_entity_resolution.src.normalization_layer.norm_missingness import (
    add_flags, check_flags,
)
from amazon.code.business_entity_resolution.src.normalization_layer.norm_script import (
    add_latin, check_latin, script_profile,
)
from amazon.code.business_entity_resolution.src.normalization_layer.norm_text import (
    add_clean, check_clean,
)

SPLITS = ("train", "test")
SOURCES = (1, 2, 3)
VOCAB_PATH = CACHE / "locality_vocab.json"
MIN_LOCALITY_LEN = 4          # drops 2-3 char noise: 'a', 'ab', 'ac', 'ag'


# ----------------------------------------------------------------- naming --

def _norm_path(split, n):
    return CACHE / f"{split}_source{n}_norm.parquet"


def _keys_path(split, n):
    return CACHE / f"{split}_source{n}_keys.parquet"


def _write(df, path):
    """Polars handles object columns cleanly where pandas to_parquet can trip."""
    pl.from_pandas(df).write_parquet(path, compression="zstd")


def _frames(paths):
    """Yield frames one at a time -- never all six in memory at once."""
    for p in paths:
        yield pd.read_parquet(p)


# --------------------------------------------------------------- phase  1 --

def normalize_one(split, n, strict=True):
    """§2.2 flags -> §2.3 clean -> §2.4 transliterate, with a checkpoint each."""
    label = f"{split}_source{n}"
    df = load_source(split, n)

    df = add_flags(df)
    check_flags(df, label, n_rows=len(df), strict=strict)

    df = add_clean(df)
    check_clean(df, label, strict=strict)

    df = add_latin(df)
    check_latin(df, label, strict=strict)

    _write(df, _norm_path(split, n))
    return df


def phase_normalize(splits=SPLITS, refresh=False, strict=True):
    for split in splits:
        for n in SOURCES:
            if _norm_path(split, n).exists() and not refresh:
                print(f"  {split}_source{n}: cached")
                continue
            df = normalize_one(split, n, strict)
            print(f"  {split}_source{n}: {len(df):,} rows")
            print(f"    scripts {script_profile(df).head(3).to_dict()}")


# ------------------------------------------------------------------ vocab --

def phase_vocab(splits=SPLITS, refresh=False):
    """Locality vocabulary, learned from the corpus rather than hardcoded.

    Corpus-derived so France (test only) is covered without a code change.
    """
    if VOCAB_PATH.exists() and not refresh:
        vocab = set(json.loads(VOCAB_PATH.read_text()))
        print(f"  vocab: cached, {len(vocab):,} terms")
        return vocab

    paths = [_norm_path(s, n) for s in splits for n in SOURCES]
    vocab = build_locality_vocab(_frames(paths))
    vocab = {w for w in vocab if len(w) >= MIN_LOCALITY_LEN}

    VOCAB_PATH.write_text(json.dumps(sorted(vocab)))
    print(f"  vocab: {len(vocab):,} terms -> {VOCAB_PATH.name}")
    return vocab


# --------------------------------------------------------------- phase  2 --

def keys_one(split, n, vocab, strict=True):
    """§2.5 name decomposition + §2.6 address decomposition."""
    label = f"{split}_source{n}"
    df = pd.read_parquet(_norm_path(split, n))
    df = add_keys(df, vocab)
    check_keys(df, label, strict=strict)
    _write(df, _keys_path(split, n))
    return df


def phase_keys(vocab, splits=SPLITS, refresh=False, strict=True):
    rows = []
    for split in splits:
        for n in SOURCES:
            label = f"{split}_source{n}"
            if _keys_path(split, n).exists() and not refresh:
                print(f"  {label}: cached")
                continue
            df = keys_one(split, n, vocab, strict)
            rows.append(profile(df, label))
            print(f"  {label}: {len(df):,} rows")
    return pd.concat(rows, axis=1) if rows else None


# -------------------------------------------------------------------- run --

def run(splits=SPLITS, refresh=False, strict=True):
    """Full pipeline. Set refresh=True to rebuild stages that are cached."""
    print("phase 1 -- flags, clean, transliterate")
    phase_normalize(splits, refresh, strict)

    print("\nphase 2 -- locality vocabulary")
    vocab = phase_vocab(splits, refresh)

    print("\nphase 3 -- blocking keys")
    summary = phase_keys(vocab, splits, refresh, strict)

    print("\ndone. handoff files:")
    for split in splits:
        for n in SOURCES:
            print(f"  {_keys_path(split, n)}")
    print(f"  {VOCAB_PATH}")

    if summary is not None:
        print()
        print(summary.to_string())
    return summary


if __name__ == "__main__":
    run()