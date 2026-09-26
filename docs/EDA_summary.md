# Business Entity Resolution — EDA Summary

Training data: S1 = 2,206,821 records (deduplicated reference), S2 and S3 ≈ 5.3M each.
All figures from a strict read (`dtype=str, na_filter=False, keep_default_na=False,
quoting=QUOTE_NONE`). Ground-truth-conditioned figures sampled at n=20,000 pairs.

---

# PART 1 — FINDINGS

## 1.1 File structure

- Headers correct in all three sources; raw line count reconciles with parsed row count; no line deviates from 3 tabs. No embedded tabs or newlines shifting columns.
- `entity_id` unique and well-formed (`^S[123]-\d+$`) in every file, prefix always consistent with the file.
- **S1 confirmed deduplicated** — zero content duplicates. The problem statement's claim holds.

## 1.2 Missingness

| Source | Empty addresses | Literal `NA` names |
|---|---|---|
| S2 | 168,967 (~3%) | 2 |
| S3 | 175,916 (~3%) | 13 |

- No literal `"NaN"` / `"null"` / `"None"` text anywhere. The "thousands of NaN addresses" seen initially were **empty strings coerced by a default `read_csv`** — a reader artefact, not a data property.
- `punct_only = 0` in all sources. Addresses have essentially no short values (one `DL` in S3).
- Records unmatchable on both fields: **0 / 3 / 4** across S1 / S2 / S3. Nothing needs excluding.

## 1.3 Injected duplicates in S2 and S3

Identical `(name, address, country)` with distinct `entity_id`:

| Source | Duplicate rows | Groups | Sizes |
|---|---|---|---|
| S2 | 50,933 | 25,060 | 24,270×2, 769×3, 19×4, 2×5 |
| S3 | 37,241 | 18,381 | predominantly pairs |

Ground-truth cross-check returns `owners=1, unmatched=0` for **every group in both sources** — all members claimed by the same S1 entity, none unlabelled. Zero singletons among ~88k duplicate rows indicates deliberate injection, not coincidental collision.

## 1.4 Short names are initialisms — and they match

- ≤2-char names: 704 in S2, 9,275 in S3, with **98% / 99.9% having a true S1 match**. Excluding them would have discarded ~10k matchable entities.
- 3-letter names (`Boa`, `Zio`, `Ace`) are legitimate and appear in S1 too → degeneracy threshold is **≤2 chars, not ≤3**.
- Patterns: token initials skipping suffixes/connectors/numerics (`AC` ← Aditya Consulting Private Limited, `SO` ← Sudduth **and** Olea, `WD` ← **864** Washington Drive Company); first+last letter of single-token names, Title-cased (`Zx` ← Zaix); junk prefixes (`#K`, `@c`); diacritics (`MÍ` ← Morales Innovative).
- Address fallback viable — only 0.04% of short-name S3 records lack an address.

## 1.5 Address component order is not fixed

The apparent "38% missing house number" was **permuted addresses**, not missing numbers — the regex only inspected position 0.

```
OH, Columbus, 5559 Orville Avenue          state, city, street
Charlotte, NC, 833 Reliance Street         city, state, street
West Bengal, Howrah, 229, Kolkata, …       state, district, number, city, street
```

India is 78% of the gap against a 40% corpus baseline (~2.3× over-represented) because Indian addresses genuinely lead with door/building identifiers — but the US 22% is pure reordering. Widening the house-number regex gained only 62.2% → 65.4%; **order, not the regex, was the limit**.

## 1.6 Script distribution

| Source | latin | devanagari | gujarati | other |
|---|---|---|---|---|
| S1 | **1.0000** | — | — | — |
| S2 | 0.9058 | 0.0535 | 0.0061 | 0.0346 |
| S3 | 0.9473 | 0.0299 | 0.0034 | 0.0194 |

**S1 is 100% Latin** → transliteration is strictly one-way with a fixed target. `other` (3.5% / 1.9%) is accented Latin plus possible additional scripts.

## 1.7 Name token vocabulary (empirical, from S1)

- **Legal suffixes:** `limited` (521,915), `llc` (355,736), `inc` (238,306), `ltd`, `llp`, `corp`, `pc`, `pllc`, `lp`, `co`, `corporation`, `company`.
- **Business-type words, NOT legal forms:** `group`, `center`, `associates`, `partners`, `clinic`, `care`, `trust`, `holdings`, `society`, `school`, `foundation`, `church`, `academy`, `institute`, `services`, `medicine`. Stripping these collapses "Shree Medical Center" → "Shree".
- Bare `c` (45,559) and `d` (5,017) are tokenization artefacts from `P.C.` and `D.B.A.`.
- Leading tokens (`pediatric`, `blue`, `shree`, `golden`) are ordinary name words — no action.

## 1.8 Locality vocabulary (self-building, all positions)

US state codes (`tx`, `ny`, `nc`, `oh`) and Indian states/cities (`maharashtra`, `delhi`, `mumbai`, `bangalore`) emerge together from all-position segment counts. Notes: `mumbai` and `mumbai city` appear separately; `in` (49,472) is ambiguous between Indiana and India.

## 1.9 Landmark markers

Present in **4.6%** of S1 addresses — lower than the Indian sample suggested. Forms include `Opp.Rta Office`, `Near E.N.T Hospital` (no space after punctuation).

## 1.10 Match structure — the decisive numbers

```
0 matches : 123,247      6 : 164,868
1         : 119,157      7 :  63,968
2         : 375,212      8 :  18,680
3         : 530,841      9 :   4,205
4         : 484,115     10 :     534
5         : 321,957     11 :      37
```

- **Singleton rate: 5.58%** — far below what the problem statement's emphasis implied.
- Mean matches: **3.461**. p95 = 6, max = 11.
- Score from predicting all-empty: **0.056**. The conservative floor is worthless.
- Source split: 80.5% of entities match **both** S2 and S3 (1,776,047); S3-only 164,498; S2-only 143,029.

## 1.11 Separability — the problem is easier than it looks

| Feature | True p5 | True p50 | Random p50 | Random p95 |
|---|---|---|---|---|
| `name_ratio` | 11 | 88 | 32 | 47 |
| `name_token_set` | 12 | 100 | 32 | 49 |
| `addr_ratio` | **59** | 94 | 34 | **45** |

`addr_ratio` alone nearly separates the classes — true p5 (59) sits **above** random p95 (45).

## 1.12 The hard 12.5% is one identifiable pattern

True pairs with `name_token_set < 60`: 2,502 / 20,000 = **12.5%**. Of those:

- short b-name: **1.0%** → the initialism slice is *not* the difficulty
- address missing: **0.3%** → nor is missingness
- **numeric address overlap: 83.3%** → addresses agree on numbers even when names diverge

The hard cases are genuine name divergence (DBA/trade names, transliteration, heavy abbreviation) with a matching address.

## 1.13 Blocking recall ceilings

| Key | Recall |
|---|---|
| `name_token` | 0.8472 |
| `name_trigram` | 0.9110 |
| `addr_numeric` | 0.7990 |
| `addr_token` | **0.9576** |
| `name_token ∪ addr_numeric` | 0.9741 |
| `name_trigram ∪ addr_numeric` | **0.9854** |
| all four | **1.0000** |

Unreachable by any key: **0.01%**.

## 1.14 Two free structural constraints

- **Cross-country true pairs: 0 / 7,638,365.** Country is a hard partition.
- **S2/S3 IDs claimed by >1 S1 entity: 0.** Matching is a clean partition — every S2/S3 record has at most one owner.

---

# PART 2 — NORMALIZATION STRATEGY

Every rule below traces to a numbered finding. Normalization is **additive** — raw columns are never overwritten.

## 2.1 Read layer

- Strict read everywhere: `dtype=str, na_filter=False, keep_default_na=False, quoting=QUOTE_NONE, encoding='utf-8-sig'`. *(§1.2 — a default read fabricates missingness and makes `"NA Enterprises"` indistinguishable from a blank cell.)*
- Cache normalized frames to parquet; the full pass costs ~3 min over 5.3M rows.

## 2.2 Missingness

- Carry explicit `has_address` boolean. Do **not** impute, do **not** drop — a record with a valid name is still matchable. *(§1.2)*
- Assert no normalized column ever contains the string `"nan"` — such values fuzzy-match each other and manufacture false merges.
- Fold bare-digit addresses (`425 587`, `541 117`) into `has_address = False`.

## 2.3 Character-level

- NFKD + accent strip. *(§1.4 `MÍ`; §1.6 `other` at 3.5%; required for France in test.)*
- Collapse whitespace runs — double spaces are real, NBSP is not (`str.contains('\xa0') == 0`; the `&nbsp;` in notebook output is a pandas HTML rendering artefact). *(§1.4)*
- Strip leading/trailing junk punctuation from names (`#`, `@`, trailing commas). *(§1.4)*
- Resolve periods **before** tokenizing, so `P.C.` and `D.B.A.` do not fragment into bare `c` / `d`. *(§1.7)*
- Normalize mangled ordinals: `2Nd`, `Ist`, `3Rd` → one form.
- Normalize doubled unit markers (`Unit UNIT 367`), care-of prefixes (`C/O`), colon separators and parentheticals (`Vill: Gondal(Mog)`), doubled hashes (`##3027`), trailing hyphens (`1510-`).

## 2.4 Script and transliteration

- Detect script by Unicode block, **dispatch on detected script — never on the `country` label** (France is in test and absent from train). *(§1.6)*
- Transliterate Indic → Latin, one-way, target form fixed by S1 being 100% Latin. Affects ~9% of S2 and ~5% of S3 names.
- Transliteration is lossy and inconsistent (`Ram`/`Raam`) → downstream comparison on that column must be character n-gram or phonetic, never exact.

## 2.5 Name decomposition

Two-tier token treatment *(§1.7)*:

- **Strip legal suffixes** (`limited`, `llc`, `inc`, `ltd`, `llp`, `corp`, `pc`, `pllc`, `lp`, `co`, `corporation`, `company`) into a separate `suffix_set` field — retained as a comparison feature, not discarded. `Summit Inc` vs `Summit LLC` differ meaningfully.
- **Keep business-type words** (`group`, `center`, `clinic`, `services`, `institute`, …) in the core name, down-weighted by IDF. Stripping them destroys precision.
- Emit `name_token_set` (sorted) so word-order transpositions collapse for free.
- Emit two derived keys per record: **token-initialism** (suffix/connector/numeric-aware) and **first+last-letter form**, both casefolded. *(§1.4)*

## 2.6 Address decomposition — order-invariant

**No positional parsing.** No component sits at a fixed index. *(§1.5)*

- Extract component **sets**, not ordered fields:
  - `addr_numbers` = `re.findall(r"\d+", addr)` over the **whole string** — recovers `229` mid-string and trailing `# 53/1`.
  - `addr_tokens` = all alphabetic tokens, stopworded.
  - `addr_locality` = any segment matching the locality vocabulary, at **any** position.
- Build the locality/state vocabulary from `value_counts` over **all** comma-segments, not trailing ones. *(§1.8)*
- Strip trailing `CITY` from locality tokens (`mumbai city` → `mumbai`). *(§1.8)*
- Resolve ambiguous `in` (Indiana vs India) by the record's `country` field. *(§1.8)*
- Expand street types bidirectionally (`ST`↔`Street`, `RD`↔`Road`) and map state code ↔ state name — both vocabularies **derived from the corpus**, not hardcoded. Fair-play clean and generalizes to France without a code change.
- Strip landmark markers (`near`, `opp`, `opposite`, `behind`, `beside`) — only 4.6% of records, low priority. *(§1.9)*

## 2.7 Duplicate collapse

- Content-hash S2 and S3 on normalized `(name, address, country)`; persist a `hash → [entity_ids]` expansion map. *(§1.3 — verified safe: every group has exactly one owner and zero unmatched members.)*
- Select one representative per group for blocking and inference.

---

# PART 3 — BLOCKING & MATCHING STRATEGY

## 3.1 Hard partition: country

Block **within country only**. Zero cross-country true pairs in 7.6M — a massive search-space reduction at zero recall cost. France partitions cleanly on its own. *(§1.14)*

## 3.2 Candidate generation

Blocking is effectively **solved** — union of keys reaches 1.000 recall with 0.01% unreachable. *(§1.13)*

- **Core:** `name_trigram ∪ addr_numeric` → 0.9854
- **Add** `addr_token` (0.9576 alone) to close the remaining gap
- `addr_numeric` must use position-independent extraction *(§1.5, §2.6)*

Effort belongs in **scoring and thresholding**, not exotic candidate generation. **Measure reduction ratio** — `addr_token` at 0.958 recall may generate very large blocks, and cost matters at 2.2M × 5.3M scale.

### Branch for short-name records *(§1.4)*

Key inverted — **address blocks, name confirms**:

| Key | p99 | max | coverage |
|---|---|---|---|
| `(init, locality)` | 23 | 1,167 | — |
| `(init, locality, house_no)` | 2 | 29 | 65% |
| `(init, locality, street_token)` on remainder | 4 | 55 | 96% of rest |

`(init, locality)` alone is unusable. Two-tier key gives **~98.6% combined coverage** at p99 ≤ 4.

### Branch for address-less records *(§1.2)*

~3% of S2/S3 cannot use any address key → name-only path, name-only scoring. A missing address must **not** produce a 0.0 address-similarity score that kills an otherwise good pair.

## 3.3 Posture: lean toward recall

This inverts the usual F₀.₅ instinct. *(§1.10)*

- Singleton rate is **5.58%**, not 30–40%. All-empty scores 0.056.
- Mean 3.461 matches per entity → missing one match drops per-entity recall by ~29%.
- The precision penalty only bites when wrong, and §1.11 shows errors will be rare.

**Be aggressive about emitting candidates.** Tune the threshold on a validation split with the macro-F₀.₅ formula, and expect the optimum to sit lower than instinct suggests.

## 3.4 Baseline before model

Given the separability in §1.11 — `addr_ratio` true p5 (59) above random p95 (45) — **measure a thresholded baseline first**:

```
score = 0.5 · name_token_set + 0.5 · addr_ratio
```

This may already be competitive. Do not build a model before this number is known.

## 3.5 Model: target the hard 12.5%

The model earns its place on name-divergent pairs with matching addresses. *(§1.12)*

Feature set:

- `name_token_set`, `name_ratio`, trigram Jaccard
- **`addr_numeric_overlap`** — first-class feature; 83.3% of hard positives have it
- `addr_ratio`, `addr_token_jaccard`
- `initialism_match` (`s1_initialism == s2_name`) — near-decisive for the short-name slice
- `has_address`, `suffix_set_match`, `script_mismatch`
- IDF-weighted rare-token overlap

Constraint: MIT/Apache-2.0 license, ≤8B parameters.

## 3.6 Post-processing

1. **Enforce single ownership** — no S2/S3 ID may be claimed by two S1 entities (verified: 0 in ground truth). On conflict, keep the higher-scoring claim. Free precision, no recall risk. *(§1.14)*
2. **Expand duplicate groups** — map each predicted representative back through `hash → [entity_ids]`. Omitting siblings loses recall for free. *(§1.3)*
3. **Format validation** — one row per S1 entity, empty lists for singletons, S2/S3 IDs only, no duplicates within a list, all IDs present in the test set. Run `utils/validate_submission.py` before every upload.

## 3.7 Test-set carry-over

- Run the identical normalization and profiling over test; diff the profiles against train.
- Verify the duplicate rate holds before relying on collapse/expand.
- Any pathology present in test but absent in train — France, almost certainly — is a gap in the cleaning layer. Script dispatch and corpus-derived vocabularies are what make this survivable.