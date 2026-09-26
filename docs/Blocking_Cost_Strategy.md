# Blocking Cost: The Problem, the Options, and the Decision Procedure

## 1. The measurement gap

Section F of the EDA established blocking **recall** with hard numbers:

| Key | Recall |
|---|---|
| `name_token` | 0.8472 |
| `name_trigram` | 0.9110 |
| `addr_numeric` | 0.7990 |
| `addr_token` | 0.9576 |
| `name_trigram ∪ addr_numeric` | 0.9854 |
| all four | 1.0000 |

That measurement answered exactly one question: *given a true pair, do the two records share a key value?* It never asked the complementary question: *given a key value, how many records carry it?*

These two quantities are independent, and the second is where cost lives. A key can have perfect recall and still be computationally unusable — the recall number tells you nothing about how many candidate pairs the key actually emits.

## 2. Why this is dangerous at our scale

Consider `addr_token`, which reached 0.9576 by matching on any shared address token. The locality vocabulary derived in §1.8 of the EDA shows `maharashtra` appearing in 191,971 S1 addresses. Assuming rough proportionality in S2 and S3 (≈5.3M records each, so ~900k occurrences), that single token value generates:

```
191,971 × 900,000 ≈ 1.7 × 10¹¹ candidate pairs
```

From **one token value**. `delhi` (155,370 in S1), `mumbai` (81,293), `tx` (133,202), `ny` (102,380) each contribute comparable volume. The full key is effectively a cross join.

`addr_numeric` has the same pathology in sharper form. Numeric tokens like `1`, `2`, `100` appear in a large fraction of all addresses; set-intersection blocking on numeric tokens without frequency control degenerates immediately.

### The structural insight

The recall came from **rare** values — distinctive street names, unusual numbers, uncommon business-name tokens. The cost comes from **common** values — state names, city names, `road`, `street`, small integers. These two populations are cleanly separable, and that separation is the entire optimization. We can discard the expensive half of each key while retaining nearly all of its recall contribution.

## 3. Reduction ratio is the wrong metric

The standard entity-resolution metric is reduction ratio:

```
RR = 1 − (candidates / all possible pairs)
```

At our scale the denominator is 2.2M × 10.6M ≈ 2.3 × 10¹³. Any blocker that does anything at all scores 0.99999-something. The metric has no resolution in the range where our decisions actually live, and it will confidently report that a pipeline-killing blocker is fine.

### Report these three instead

| Metric | Definition | Target |
|---|---|---|
| **PC** (pair completeness) | fraction of true pairs surviving blocking | ≥ 0.97 |
| **Candidates per S1 entity** | mean and p99 of block sizes | mean 20–50, p99 < 500 |
| **PQ** (pair quality) | true pairs ÷ total candidate pairs | maximize at fixed PC |

**The anchor:** there are ~7.64M true pairs across 2,206,821 S1 entities — a mean of 3.46 matches per entity. A *perfect* blocker would emit 3.46 candidates per entity.

- 20 candidates/entity → 44M pairs → comfortable for a scoring model
- 500 candidates/entity → 1.1B pairs → painful but survivable
- 50,000 candidates/entity → 10¹¹ pairs → pipeline is dead

The mean alone is insufficient. A single S1 entity sitting in a 900k-record block will dominate runtime no matter how good the average looks, which is why p99 and max belong in the report.

## 4. Measuring cost without materializing pairs

We never need to build the candidate pairs to know how many there would be. An inverted index of value → document frequency on each side gives the exact count as a sum of products:

```python
from collections import Counter

def key_cost(a_keys, b_keys):
    """a_keys, b_keys: Series of sets (one set of key values per record).
    Returns total pair count and the 20 most expensive key values."""
    da = Counter(v for s in a_keys for v in s)
    db = Counter(v for s in b_keys for v in s)
    per_value = {v: da[v] * db[v] for v in da.keys() & db.keys()}
    total = sum(per_value.values())
    worst = sorted(per_value.items(), key=lambda kv: -kv[1])[:20]
    return total, worst
```

Run on a 200k-record sample per side and scale. The `worst` list is the real diagnostic — it names the specific token values causing the blow-up, which based on the EDA vocabulary will be `maharashtra`, `delhi`, `road`, `street`, `1`, `2`. Five minutes of work that tells us precisely what to fix.

Per-entity distribution, approximated from document frequencies without building pairs:

```python
cand_per_entity = a_keys.map(lambda s: sum(db[v] for v in s))
cand_per_entity.describe(percentiles=[.5, .9, .99, .999])
```

This is an overestimate (it double-counts records sharing multiple key values with the same entity) but it is a correct upper bound and cheap to compute.

## 5. Four mechanisms for bounding cost

### 5.1 Document-frequency ceiling

Drop key values whose corpus DF exceeds a threshold — for example, any value appearing in more than 0.1% of records.

This is stopwording **learned from the corpus** rather than hand-listed, which matters for two reasons already established in the EDA: the legal-suffix vocabulary was derived empirically in §1.7 rather than guessed, and §2.6 requires that vocabularies generalize to France in the test set without a code change. A DF ceiling satisfies both automatically.

Recall cost is limited to records whose *entire* signal consists of common tokens. That population must be measured, not assumed — a record named "The Group Limited" at "Road, Mumbai, Maharashtra" has no rare token and would be lost.

### 5.2 Rare-token keys (top-k by IDF)

Rather than keying on every token, key each record on its *k* rarest tokens (k = 2 or 3) ranked by corpus IDF.

Each record then contributes a bounded number of key values, and the values it contributes are the discriminative ones. This is typically the largest single win because it attacks cost and preserves recall **simultaneously** — the rare tokens are precisely what produced the 0.9576 figure in the first place. Common tokens were contributing cost without contributing recall.

### 5.3 Conjunctive keys

Combine two weak signals into one strong one. The EDA already demonstrated this empirically in §1.4:

| Key | p99 | max |
|---|---|---|
| `(init, locality)` | 23 | 1,167 |
| `(init, locality, house_no)` | 2 | 29 |

Same recall, an order of magnitude less cost. The pattern generalizes: `(rare_name_token, locality)` will be far tighter than either component alone, and `(rare_addr_token, country)` tighter still — noting that country is already a hard partition per §1.14.

### 5.4 Top-k retrieval instead of exact key match

Build a TF-IDF character-n-gram index and retrieve the *k* nearest S2/S3 records per S1 entity.

This **bounds cost by construction**: exactly `2.2M × k` candidates, no tail, no pathological blocks, fully predictable runtime. At k = 30 that is 66M pairs. Given that blocking recall is already at ceiling and that our risk is entirely on the cost side, this is a strong candidate for the primary mechanism, with exact-key blocks retained as supplements for cases n-grams handle poorly — the initialism slice from §1.4 especially, where a 2-character name shares almost no n-grams with its full-form counterpart.

## 6. The decision procedure

### Step 1 — Plot recall@k and pick the knee

For a sample of true pairs, compute the rank of the true match under top-k retrieval. Plot PC against k for k ∈ {5, 10, 20, 50, 100, 200}. The curve rises steeply then flattens.

**Pick the knee, not the maximum.** If PC reaches 0.97 at k = 25 and 0.975 at k = 100, the additional 0.005 recall costs 4× the compute. Refuse it.

### Step 2 — Analyze the residue, not the aggregate

Once a primary scheme is chosen, examine *which* true pairs it misses. The EDA strongly suggests the misses will be concentrated rather than uniform:

- short-name/initialism records (§1.4 — 9,275 in S3, 98–99.9% matched)
- address-less records (§1.2 — ~3% of S2/S3)
- Devanagari/Gujarati records requiring transliteration (§1.6 — ~9% of S2)
- the hard 12.5% with divergent names but matching addresses (§1.12)

If misses are concentrated, add a **narrow targeted key for exactly that slice** rather than loosening the main key. A narrow key over 9k records is nearly free; loosening a key over 2.2M records is not.

### Step 3 — Cascade

This is the architecture we are pursuing. §1.11 of the EDA showed that 87.5% of true pairs have `name_token_set ≥ 60`, and that `addr_ratio` alone nearly separates the classes (true p5 = 59 vs random p95 = 45). Most pairs are easy.

The cascade exploits this directly:

1. **Stage 1 — cheap, tight, high-precision key.** Exact or near-exact match on normalized name + locality. Runs over everything. Resolves the large easy majority.
2. **Stage 2 — moderate.** Rare-token conjunctive keys, run only over S1 entities not confidently resolved in stage 1.
3. **Stage 3 — expensive, recall-oriented.** Top-k n-gram retrieval plus targeted slice keys, run only over the remaining residue.

Because the residue shrinks at each stage, the expensive keys operate on a small fraction of the data. If stage 1 resolves 60% of entities, stage 3 processes 40% of 2.2M rather than all of it — and stage 3 is where the per-pair cost is highest.

Two design requirements for the cascade to be sound:

- **"Confidently resolved" must be defined by a high-precision threshold**, not merely by "found something." A weak stage-1 match must fall through to stage 2, not terminate the search. §1.10 establishes a mean of 3.46 matches per entity, so an entity is rarely finished after one match.
- **Stages must be additive in candidates, not exclusive.** An entity resolved in stage 1 still needs its remaining ~2.5 matches found. The cascade gates *effort*, not *eligibility* — this distinction is what separates a cascade from a premature cutoff that silently destroys recall.

### Step 4 — Price each union increment explicitly

Union keys have diminishing recall returns and additive cost:

```
name_trigram ∪ addr_numeric  = 0.9854
all four                     = 1.0000
```

That final 1.5% is purchased with `addr_token` and `name_token` — the two most expensive keys in the set. Compute the cost of each increment and decide explicitly.

Under macro-F₀.₅ with a mean of 3.46 matches per entity, 1.5% more true pairs is genuinely worth score — missing one match costs ~29% of an entity's recall. But it is only worth it if the pipeline still runs. This is a real tradeoff with a real answer, and the answer depends on numbers we have not yet measured.

## 7. What "optimal" means here

The objective is: **maximize PC subject to a compute budget.**

That makes the correct frame marginal — for each key under consideration, *what recall does it buy per unit of cost?* Rank candidate keys by that ratio and add greedily until the budget is exhausted. A key buying 0.02 recall for 10M pairs beats one buying 0.05 recall for 5B pairs.

**Set the budget first.** Decide what the scoring model can process in the available time, convert that to a pair count, then design blocking to hit that number. Choosing keys first and discovering cost afterward is the failure mode that leaves a team unable to run inference the night before the deadline.

## 8. Target architecture

Based on the EDA and the reasoning above, the working plan:

- **Country partition** as a hard prefilter (§1.14 — zero cross-country true pairs, free)
- **Duplicate collapse** before blocking (§1.3 — ~44k fewer records, verified safe)
- **Cascade** with three stages gated on match confidence, not eligibility
- **DF-capped rare-token conjunctive keys** as the base mechanism
- **Top-k n-gram retrieval** as the bounded-cost safety net
- **Targeted keys** for the initialism slice and the no-address slice
- **Duplicate expansion** in post-processing (§1.3)

### Acceptance criteria before any scoring code is written

- PC ≥ 0.97 on a held-out validation split
- p99 candidates per S1 entity < 500
- Total candidate pairs within the measured compute budget
- Residue analysis showing misses are either negligible or covered by a targeted key