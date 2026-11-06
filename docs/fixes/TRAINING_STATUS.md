# RECALL ML Training & Evaluation Status

## Executive Summary

- **Generic `WebDocument` architecture:** Fully implemented and validated across diverse sources (Wikipedia, Python docs, technical blogs).
- **Multi-label Topic Classifier:** DistilBERT model (`models/recall-topic-classifier`) loaded, verified, and integrated with automatic fallback and degraded-mode reporting.
- **Dataset Size & Integrity:** 5,000 raw samples, deduplicated to 4,940 unique canonical samples across 17 taxonomy labels (including `History`).
- **Data Splitting & Leakage Prevention:** Disjoint stratified group splitting with Union-Find Jaccard near-duplicate grouping and normalized text deduplication (0 overlap across train/val/test).
- **Test Suite:** 33 unit and integration tests passing (`app/tests/`, `ml/tests/`).
- **Model Evaluation:** Independently evaluated on the leak-free test split (741 samples) with exact metrics, per-class breakdown, and confusion matrix.

---

## Dataset & Split Verification

- **Source File:** `backend/ml/data/starter_dataset.jsonl`
- **Total Records:** 5,000
- **Unique Records (Normalized):** 4,940
- **Exact Duplicates Merged:** 60 records
- **Split Ratios:** 70% Train (3,458 samples), 15% Validation (741 samples), 15% Test (741 samples)
- **Data Leakage Audit:**
  - Exact Text Overlap: 0 (train/val: 0, train/test: 0, val/test: 0)
  - Normalized Text Overlap: 0 (train/val: 0, train/test: 0, val/test: 0)
  - Result: **PASS (100% Leak-Free)**

### Class Distribution (Overall 5,000 samples)
- `Information Retrieval`: 667
- `Data Science`: 624
- `Machine Learning`: 594
- `Programming`: 553
- `Deep Learning`: 466
- `Artificial Intelligence`: 437
- `Large Language Models`: 349
- `Databases`: 330
- `Web Development`: 324
- `Other`: 305
- `Mathematics`: 273
- `RAG`: 266
- `Cloud Computing`: 265
- `Cybersecurity`: 215
- `Business`: 206
- `Science`: 166
- `History`: 145

---

## Model Evaluation Results (Test Split: 741 samples)

- **Model:** `distilbert-base-uncased` multi-label sequence classification
- **Decision Threshold:** 0.35
- **Subset Accuracy:** 91.23%
- **Hamming Accuracy:** 99.31% (Hamming loss: 0.0069)
- **Micro F1:** 95.29%
- **Macro F1:** 94.92%
- **Weighted F1:** 95.32%
- **Micro Precision:** 94.73%
- **Micro Recall:** 95.86%

### Per-Class Performance

| Topic Class | Precision | Recall | F1 Score | Test Support |
| :--- | :--- | :--- | :--- | :--- |
| **Artificial Intelligence** | 98.4% | 98.4% | 98.4% | 64 |
| **Machine Learning** | 93.1% | 99.0% | 96.0% | 96 |
| **Deep Learning** | 96.8% | 98.4% | 97.6% | 62 |
| **Large Language Models** | 83.9% | 94.5% | 88.9% | 55 |
| **RAG** | 62.9% | 62.9% | 62.9% | 35 |
| **Information Retrieval** | 88.5% | 98.8% | 93.4% | 86 |
| **Data Science** | 100.0% | 98.8% | 99.4% | 83 |
| **Programming** | 97.9% | 98.9% | 98.4% | 93 |
| **Web Development** | 97.8% | 86.3% | 91.7% | 51 |
| **Databases** | 100.0% | 93.9% | 96.8% | 49 |
| **Cloud Computing** | 100.0% | 97.9% | 98.9% | 47 |
| **Cybersecurity** | 95.1% | 95.1% | 95.1% | 41 |
| **Mathematics** | 100.0% | 100.0% | 100.0% | 45 |
| **Science** | 100.0% | 100.0% | 100.0% | 21 |
| **Business** | 100.0% | 96.8% | 98.4% | 31 |
| **Other** | 100.0% | 95.6% | 97.7% | 45 |
| **History** | 100.0% | 100.0% | 100.0% | 14 |

---

## Reproducible Commands

From `backend/`:

1. **Run Full Test Suite:**
   ```powershell
   python -m pytest app/tests/ ml/tests/ -v
   ```

2. **Run Leak-Free Model Evaluation:**
   ```powershell
   python -m ml.evaluate
   ```

3. **Validate End-to-End Ingestion Pipeline:**
   ```powershell
   python -m scripts.validate_web_sources --all
   ```

4. **Retrain Model (if required):**
   ```powershell
   python -m ml.train
   ```
