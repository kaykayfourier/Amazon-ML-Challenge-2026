# Cost-Aware Blocking Strategy

**Business entity resolution | Simple decision guide based on your EDA and blocking-cost notes | 26 September 2026**

> **Recommendation:** use cheap, discriminative keys first, then a bounded search fallback. Never let a shared common token create a giant block. Keep a hard limit on pairs sent to ML, and measure the search work before that limit.

## 1. The problem in plain English

Blocking finds possible matches. A loose key such as the word **"road"** may keep a true match, but it also joins huge numbers of unrelated records. High recall by itself does not mean the method is affordable. The cost notes estimate that one common locality token alone could produce roughly **170 billion raw pair occurrences**. That estimate is illustrative, not a measured candidate count.

| Your measured signal | What it means for blocking |
|---|---|
| **2,206,821 Source-1 businesses; about 5.3M records per target source in the EDA notes** | Full all-pairs comparison is on the order of **23 trillion comparisons**. |
| **Mean 3.461 true matches per Source-1; maximum 11** | Keep multiple matches. Finding one strong match must not stop candidate generation. |
| **12.5% of sampled positive pairs had divergent names; 83.3% of those shared address numbers** | Always reserve a bounded address-led route for names that differ. |
| **About 3% missing target addresses; S1 Latin, some target names in Indic scripts** | Add name-only and transliteration paths, but run them within a budget. |

### The recommended starting budget

For the full run, start with **50 unique candidate records per Source-1 query after deduplication**: at most about **110.3 million candidate pairs** across **2,206,821 queries**. That is a ceiling, not a promise of runtime. For a **20,000-query pilot** it is at most **1 million pairs**. Train on the pilot before committing to a full run. The EDA files do not measure end-to-end throughput, so these are planning numbers only.

> **Important distinction:** "90% of true pairs found" is recall. "Millions of records emitted" is candidate cost. We need both on the same plot.

---

## 2. The actual blocking design

| Step | What to do | Why cost stays bounded |
|---|---|---|
| **0. Profile** | Measure frequency (DF) of each key in S2+S3, split by country; save the largest posting lists. | Counts are computed before generating pairs. |
| **1. Easy matches** | Exact normalized name + country; rare name token + locality; distinctive address number + rare street token. | Skip broad keys and oversized posting lists. Do not remove other possible matches. |
| **2. Main retrieval** | Name character TF-IDF and name BM25, then address channels. Include transliterated name search when useful. | Take small ranked lists per channel and merge by target ID. |
| **3. Targeted rescue** | For weak or incomplete queries, add address-led, initialism, no-address, and cross-script retrieval. | Spend extra work only where cheap channels fail; each path has its own cap. |
| **4. Final filter** | Merge, score cheaply, keep at most 50 unique IDs per query. Send these pairs to detailed features + LightGBM. | Absolute bound on expensive ML pairs. |

### A concrete pilot allocation *(tune from real measurements)*

| Channel | Starting top-K | Special condition |
|---|---:|---|
| Exact / rare composite keys | up to 15 | Reject or refine large postings; no all-token OR expansion. |
| Name TF-IDF + name BM25 | 15 each | Search relevant country shard; deduplicate overlaps. |
| Address TF-IDF + address BM25 | 10 each | Run when address exists; elevate for name divergence. |
| Transliteration / initialism rescue | up to 10 extra | Only on applicable scripts or very short names. |
| **Final unique candidates** | **50 total** | **Not the sum: merge and cap by query.** |

The channel counts are starting knobs, not measured optima. Because lists overlap, the union may contain fewer than their sum. Retain a query-specific reserve for address-led candidates so name channels cannot fill all 50 slots. If 50 loses too many true pairs, test **75 and 100 only on the hard slices** and price the extra cost.

> **Avoid this trap:** "one match found" does not mean the query is finished. The ground truth averages **3.461 matches per entity**. A cascade can reduce effort, but it must still look for additional matches.

---

## 3. Keep broad keys from exploding

### Frequency ceiling

Do not expand a key whose target posting list is above a measured limit. As a pilot, test a ceiling of **1,000 targets per key per country**. Such a key must be combined with a rare name token, house number, or locality; otherwise skip it. The exact ceiling is chosen by a cost-recall sweep, not treated as a universal rule.

### Rarest-token selection

Use at most **two or three distinctive tokens** from each field. `Delhi`, `road`, `street`, `limited`, `1`, and `2` must not create standalone blocks. Keep legal suffixes as features, but do not let them drive candidate generation.

### Order-invariant addresses

Extract all numbers and tokens from anywhere in the address. Do not assume the house number is first. Combine a number with a rare street/name token rather than using the number alone.

### Country

The EDA reports **zero cross-country positives** in its observed ground truth. Use country-specific indexes for known, valid countries, but send missing/unknown country queries to a bounded fallback. Recheck this assumption on held-out records and on the future test distribution; **France may occur at test time**.

### Cost example

| Final cap | Maximum full-run ML pairs | Approximate raw float32 feature matrix at 40 features* |
|---:|---:|---:|
| 20 | 44.1 million | 7.1 GB |
| 50 | 110.3 million | 17.7 GB |
| 100 | 220.7 million | 35.3 GB |

\*Illustrative arithmetic: `pairs × 40 × 4 bytes`, before IDs, Python objects, retrieval matrices, models, and temporary copies. Write pair features in batches to disk; do not allocate one giant pandas DataFrame. GPU does not fix an uncontrolled number of candidate pairs.

### Two budgets, not one

- **Retrieval budget:** number of posting entries scanned / similarities computed and wall-clock time. Top-50 output does not save you if finding that top-50 traverses a million-item block.
- **ML budget:** number of distinct candidate pairs after union and truncation. This controls feature generation and prediction cost.

### Where it fits the current repository

Your current baseline already has **character TF-IDF, BM25 and transliteration channels**. Treat the exact/rare composite keys and targeted rescue as a **proposed improvement**, not as implemented code. First run a small benchmark of the existing retriever at **K=20/50/100**; add a new key only if it improves recall per unit of search and ML cost.

---

## 4. How to decide if it is good enough

Use grouped validation by Source-1 entity. Build the target index without inserting labels, retrieve for validation queries, and only then compare candidate IDs to ground truth. Tune blocker settings on a development split; reserve a separate untouched holdout for the final report. Measure both **micro pair recall** and **macro query coverage**.

| Measure | Simple definition | Decision rule for pilot |
|---|---|---|
| **Pair completeness / micro recall** | True links retrieved / all true links | Seek **97% or better** first; improve only if affordable. |
| **Macro coverage** | Average per-query fraction of its true links retrieved | Watch multi-match entities, not just one hit per query. |
| **Candidate volume** | Total, mean, median, p95, p99, max unique pairs per query | Aim near **20-50 mean**; investigate **p99 over 500**. |
| **Retrieval work** | Posting entries touched, similarity calls, seconds per 1,000 queries, peak RAM | Set a Colab-hour/RAM budget before the full pass. |
| **Downstream score** | Calibration threshold, untouched holdout macro F0.5 | Do not select threshold on holdout. |

### A test matrix that directly answers the cost question

| Trial | Compare on the same query sample |
|---|---|
| **A** | Existing TF-IDF only at K=20, 50, 100 |
| **B** | Existing BM25 only at K=20, 50, 100 |
| **C** | Hybrid with transliteration at final cap 20, 50, 100 |
| **D** | C + rare composite keys + targeted rescue, only if measured benefit |

Price the marginal gain:

**(new true pairs found) / (extra unique pairs sent to ML)**

and also divide by extra retrieval seconds. Pick the smallest setting that meets the agreed recall and runtime limits. If none meet both, report the **Pareto tradeoff** honestly; do not claim an optimal key without a measured cost curve.

> **Go / no-go:** do a **20,000-query pilot against the full target catalog**. Stop the full run if projected CPU time, disk use or peak RAM exceeds your Colab budget, even when recall looks excellent.

---

## 5. Special cases and data integrity

- **Short names and initialisms:** only about **10,000 target records** have names of two characters or fewer in the supplied EDA. Search with initialism + locality + house number or rare street token, with a small posting cap. Do not drop them.
- **Missing address:** keep a name-only path; a blank address must not create an empty-string mega-block or act as hard negative evidence.
- **Cross-script names:** preserve Unicode and search an additional transliterated representation. A lossy transliteration is a retrieval hint, not proof of a match.
- **Duplicate target content:** grouping identical normalized target records could save work. The EDA reports same-owner training duplicates, but only collapse after verifying this on the current files and retain the ID expansion list for submission. Do not silently discard IDs.

### Important inconsistency in the supplied evidence

The attached EDA note says every Source-3 row was a valid four-column TSV and describes roughly **5.3M rows**. Your later Colab run found **two malformed physical rows** and counted **5,274,632 Source-3 records**, while an earlier report stated **5,285,603**. The current notebook also reported an **"Unknown target label"** error; the preliminary SQL count of **8,193 missing target IDs** may reflect either a partial building index or mismatched data.

These findings cannot be reconciled by a blocking algorithm. Before training or claiming final recall, verify the exact **file hashes**, **full row counts**, and that **every ground-truth target ID exists in S2/S3** using a completed index or independent ID scan.

### What not to claim

The **0.9854** value for name trigram union address numeric, and **1.0000** for four-key union, are **raw key-sharing recall on the EDA sample**. They are **not Recall@50**, not final capped blocker recall, not model F0.5, and not proof the method fits in memory. The **50-candidate cap** and **1,000-posting ceiling** in this report are proposals until measured on the current files.

### Suggested next actions

1. Reconcile the four current training files and ground-truth IDs.
2. Measure key frequencies and the most expensive postings without materializing pairs.
3. Run a **20,000-query full-catalog pilot** and save recall + runtime + p99.
4. Sweep final caps **20/50/100**.
5. Add rare-key or special-case rescue only for observed misses.
6. Once validated, train and evaluate on an untouched grouped holdout.

### Sources and scope

Project evidence: supplied **`EDA_summary(2).md`** and **`Blocking_Cost_Strategy(2).md`**; user-provided Colab counts and errors from **26 September 2026**.

General method context:
- Papadakis et al., *How to reduce the search space of Entity Resolution* (2022)
- Strojny and Beresewicz, *BlockingPy* (2025)

External papers support the need to compare blocking and nearest-neighbor workflows, not the numeric claims about this dataset.
