# Normalization Layer — Handoff

Six output files in `cache/`, one per source per split:

```
train_source1_keys.parquet   test_source1_keys.parquet
train_source2_keys.parquet   test_source2_keys.parquet
train_source3_keys.parquet   test_source3_keys.parquet
```

Plus `cache/locality_vocab.json` — the 2,516-term locality vocabulary, needed to reproduce `addr_locality`.

All six are checkpoint-validated. Raw columns are preserved untouched; normalization is additive throughout.

---

## Pipeline order

```python
from dataio           import load
from norm_missingness import add_flags
from norm_text        import add_clean
from norm_script      import add_latin
from norm_keys        import build_locality_vocab, add_keys

s1, s2, s3 = load("train")                    # §2.1 strict read + cache
df = add_flags(df)                            # §2.2 missingness flags
df = add_clean(df)                            # §2.3 character cleaning
df = add_latin(df)                            # §2.4 transliteration
vocab = build_locality_vocab([...all six...]) # corpus-derived, len>=4 filter
df = add_keys(df, vocab)                      # §2.5 name + §2.6 address keys
```

Each stage has a matching checkpoint (`check_flags`, `check_clean`, `check_latin`,
`check_keys`) built on the shared `Check` harness in `validation.py`. They collect
all issues rather than raising on the first, and are strict by default.

---

## What each stage does

**§2.1 Strict read** (`dataio.py`) — Polars read, no null coercion, no dtype
inference, no quote handling. An empty cell stays `""`; a business named "NA"
stays `"NA"`. A default `read_csv` conflates the two and fabricates ~340k
phantom nulls.

**§2.2 Missingness** (`norm_missingness.py`) — Flags only. Nothing imputed,
nothing dropped. A record with a name but no address is still matchable.

**§2.3 Character cleaning** (`norm_text.py`) — Latin-only accent fold (Indic
vowel signs are combining marks and must be protected), lowercase, `&`→`and`,
URL/domain removal, acronym periods resolved *before* tokenizing, compound
numbers preserved (`8-2-644/1/20` → `8_2_644_1_20`), address sentinels (`null`,
131k occurrences) stripped, punctuation→space with the Indic range exempted,
whitespace collapse, adjacent-token dedupe. Order is load-bearing.

**§2.4 Transliteration** (`norm_script.py`) — Nine Indic scripts (Devanagari,
Telugu, Kannada, Tamil, Bengali, Gujarati, Malayalam, Oriya, Gurmukhi) detected
by Unicode block. `anyascii` plus two repair rules (`ph`→`f`, anusvara→`n`),
measured at 75.2 mean similarity to true S1 matches vs 68.0 for IAST. Unique-value
caching makes it tractable. S1 is Latin, so conversion is strictly one-way.

**§2.5 / §2.6 Key generation** (`norm_keys.py`) — Legal suffixes split out,
business-type words retained, order-invariant token sets, position-independent
numeric extraction.

---

## Output columns

### Identity (unchanged from source)

| Column | Notes |
|---|---|
| `entity_id` | `S1-`/`S2-`/`S3-` prefix |
| `business_name` | raw, untouched |
| `business_address` | raw, untouched |
| `country` | **hard partition — zero cross-country true pairs in 7.6M** |

### Flags (§2.2)

| Column | Meaning |
|---|---|
| `has_address` | address non-blank. `False` → name-only scoring path (~3% of S2/S3) |
| `addr_degenerate` | non-blank but no letters in any script (87 rows — negligible) |
| `addr_usable` | `has_address & ~addr_degenerate` |
| `has_name` | non-blank and not a sentinel |
| `name_short` | name ≤2 chars → **initialism**, not a defect. 98–99.9% of these have a true match; route to the address-blocks/name-confirms branch |

### Cleaned text (§2.3, §2.4)

| Column | Meaning |
|---|---|
| `name_clean` | character-normalized name, original script preserved |
| `addr_clean` | character-normalized address |
| `name_script` | `latin` or one of nine Indic script labels — useful as a model feature (transliterated matches deserve less trust than native Latin ones) |
| `name_latin` | **Latin for every row.** Passthrough where already Latin, transliterated otherwise. Downstream keys read this and never branch on script |

### Blocking keys (§2.5, §2.6)

| Column | Meaning | Use |
|---|---|---|
| `name_core` | `name_latin` minus legal suffixes | display / features |
| `name_tokens` | sorted unique token set of `name_core` | `name_token` key |
| `suffix_set` | the legal forms removed (`ltd pvt`) | comparison feature, not a key |
| `initialism` | first letters of significant tokens (`sde`) | short-name branch: match against ≤2-char S2/S3 names |
| `addr_numbers` | all numeric tokens, **any position**, compounds intact | `addr_numeric` key |
| `addr_tokens` | stopworded alphabetic address tokens | `addr_token` key |
| `addr_locality` | tokens matching the corpus locality vocabulary | conjunctive component only |

Use `name_latin` for character-trigram keys; `name_tokens` for token-set keys.

---

## Constraints for the blocking stage

1. **Partition by country first.** Zero cross-country true pairs. Free, large reduction.
2. **`name_trigram ∪ addr_numeric` reaches 0.985 recall**; adding `addr_token` reaches 1.000.
3. **DF-cap common tokens or blocks explode.** `maharashtra` alone appears in ~192k S1 addresses — an uncapped `addr_token` key approaches a cross join. See `blocking_cost_analysis.md`.
4. **`addr_locality` is never a standalone key** — always conjunctive (with initialism or house number).
5. **Numeric keys must use set intersection**, not positional match. Address components sit at no fixed index.

---

## Required post-processing (not yet implemented)

**Duplicate expansion.** S2 and S3 contain injected duplicates — identical
`(name, address, country)` with distinct IDs: 25,060 groups in S2, 18,381 in S3.
Ground truth confirms every group has exactly one owner and zero unmatched
members, so collapsing to one representative before inference is safe and saves
~44k redundant comparisons.

**If you collapse, you must expand.** Every predicted match on a representative
means all member IDs of its group are matches. Omitting siblings loses recall for
free. This is §2.7 and is not yet built — currently the frames contain all
records uncollapsed, so blocking works as-is.

---

## Known limitations

- Locality vocabulary is token-level, not comma-segment-level, so multi-word
  localities (`hauts de france`) fragment. `build_locality_vocab` accepts a `col`
  parameter if you want to rebuild from raw segments.
- `ADDR_STOP` is English-only. Hindi address words (`nagar`, `gali`, `marg`,
  `vihar`, `puram`, `chowk`) are not stopworded and leak into `addr_tokens` and
  `addr_locality`.
- Injected typos hit transliterated suffixes too (`praivet`, `lintid`), which
  `LEGAL` does not catch, so a small number of Indic records keep suffix tokens
  in `name_core` and get a wrong `initialism`.
- Vocabulary noise: some personal names (`agarwal`, `ahmed`) and generic words
  (`acres`, `airport`) are in the locality set. Harmless in a conjunctive key.