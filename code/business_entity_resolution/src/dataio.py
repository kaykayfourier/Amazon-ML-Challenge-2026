"""
dataio.py -- strict loading and parquet caching.

Reads with polars (fast over 5.3M rows), returns pandas (what normalization
and the EDA notebooks use).

Read semantics: nothing is coerced to null. An empty cell stays an empty
string, a business named "NA" stays the string "NA".

    from dataio import load, load_gt, load_pairs

    s1, s2, s3 = load("train")      # pandas, cached after first call
    gt         = load_gt("train")   # one row per S1 entity, singletons kept
    pairs      = load_pairs("train")# one row per (s1_id, matched_id)
"""

from pathlib import Path

import polars as pl
ROOT = Path(__file__).resolve().parents[3]   # src -> business_entity_resolution -> code -> amazon
DATA = ROOT / "dataset"
CACHE = ROOT / "cache"
COLUMNS = ["entity_id", "business_name", "business_address", "country"]
GT_COLUMNS = ["source1_entity_id", "matched_entity_ids"]


def _read_tsv(path, columns):
    """No dtype inference, no null coercion, no quote handling."""
    return pl.read_csv(
        path,
        separator="\t",
        quote_char=None,
        encoding="utf8",
        null_values=[],                      # "NA"/"null" stay literal
        missing_utf8_is_empty_string=True,   # empty cell -> "" not null
        schema_overrides={c: pl.String for c in columns},
        infer_schema_length=0,
    )


'''def _cached(name, build):
    """Read the parquet if it exists, otherwise build and write it.
    Returns pandas."""
    path = CACHE / f"{name}.parquet"
    if not path.exists():
        CACHE.mkdir(exist_ok=True)
        build().write_parquet(path, compression="zstd")
    return pl.read_parquet(path).to_pandas()'''

def _cached(name, build):
    path = CACHE / f"{name}.parquet"
    if not path.exists():
        CACHE.mkdir(exist_ok=True)
        build().write_parquet(path, compression="zstd")
    return pl.read_parquet(path).to_pandas().astype(object)

def load_source(split, n):
    """One source file, validated on first read, cached, returned as pandas."""
    def build():
        df = _read_tsv(DATA / split / f"{split}_source{n}.tsv", COLUMNS)
        assert df.columns == COLUMNS, f"S{n}: unexpected columns {df.columns}"
        assert df["entity_id"].n_unique() == df.height, f"S{n}: duplicate entity_id"
        return df
    return _cached(f"{split}_source{n}", build)


def load(split):
    """All three sources for a split."""
    return tuple(load_source(split, n) for n in (1, 2, 3))


def load_gt(split="train"):
    """Ground truth as-is -- one row per S1 entity, empty lists preserved.

    Use this (not load_pairs) whenever singletons matter: they are 5.6% of
    entities and score 1.0 when correctly predicted empty.
    """
    def build():
        df = _read_tsv(DATA / split / f"{split}_ground_truth.tsv", GT_COLUMNS)
        assert df.columns == GT_COLUMNS, f"gt: unexpected columns {df.columns}"
        return df
    return _cached(f"{split}_ground_truth", build)


def load_pairs(split="train"):
    """Ground truth exploded to one row per (source1_entity_id, mid).
    Entities with no matches are dropped."""
    def build():
        return (
            _read_tsv(DATA / split / f"{split}_ground_truth.tsv", GT_COLUMNS)
            .with_columns(pl.col("matched_entity_ids").str.split(","))
            .explode("matched_entity_ids")
            .with_columns(pl.col("matched_entity_ids").str.strip_chars().alias("mid"))
            .filter(pl.col("mid") != "")
            .select("source1_entity_id", "mid")
        )
    return _cached(f"{split}_pairs", build)


if __name__ == "__main__":
    for split in ("train", "test"):
        if not (DATA / split).exists():
            continue
        print(f"\n=== {split} ===")
        for n in (1, 2, 3):
            df = load_source(split, n)
            blank = (df["business_address"].str.strip() == "").sum()
            print(f"  S{n}: {len(df):>9,} rows   blank addr {blank:>8,} "
                  f"({blank / len(df):.2%})   {sorted(df['country'].unique())}")
        if (DATA / split / f"{split}_ground_truth.tsv").exists():
            print(f"  gt: {len(load_gt(split)):,} entities, "
                  f"{len(load_pairs(split)):,} pairs")