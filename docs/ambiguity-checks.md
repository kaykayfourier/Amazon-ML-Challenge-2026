## Data validation lead — ambiguity checks:

- Column/field definitions match exactly between train and test sets (no silent renames or type shifts)
- Missing value patterns — are they random, or systematic (e.g., missing only for a specific category)?
- Label leakage — any column that indirectly encodes the target
- Units and scales consistent across rows (currency, dimensions, timestamps)
- Duplicate rows or near-duplicate entries with conflicting labels
- Class/label imbalance and whether it matches the real-world distribution or is artificially skewed
- Text/image fields with inconsistent encoding, corrupted files, or placeholder values (e.g., "N/A", blank strings, broken URLs)
- Train/test distribution shift (feature stats, category overlap, unseen categories in test)
- Ambiguous or multi-interpretable target definitions (e.g., "price" — MRP vs selling price vs discounted price)
- Whether evaluation metric rewards behavior consistent with the actual task (e.g., MAPE punishing near-zero targets disproportionately)
- Outliers — genuine vs data entry errors
- Tokenization/parsing edge cases if text is involved (special characters, multilingual entries, HTML remnants)

## Critical checks for other teammates (model/deep learning side):

- Confirm the exact loss function and metric used for leaderboard scoring before optimizing
- Check for data leakage introduced during feature engineering (e.g., using test-time-unavailable info)
- Validate any preprocessing/augmentation pipeline is applied identically to train and inference
- Confirm reproducibility — fixed seeds, deterministic splits
- If using pretrained models, verify license/allowed-use compliance with competition rules
- Sanity-check submission format against sample submission file (column order, ID matching, rounding rules)
- Cross-validation setup matches train/test split logic (no random split when data has group structure, e.g., same product across rows)
- Ensemble/blending weights validated on holdout, not just intuition
- Inference time and memory constraints checked against competition limits (if runtime scoring applies)