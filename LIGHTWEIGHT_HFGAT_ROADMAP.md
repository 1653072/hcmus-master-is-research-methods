# Lightweight HFGAT — Top Priorities

Goal: **less RAM/time** on 16GB Mac, Colab, and Kaggle while keeping **HR, NDCG, and compat** close to the current stable branch.

Current stable baseline: `kimnguyen-quoctran-v1.1` — 3-notebook pipeline, decoupled compat (`COMPAT_DETACH_INPUT=True`), train-only user–outfit graph.

---

## Priority list (do in order)

| # | Change | Where | Why | Re-run |
|---|--------|-------|-----|--------|
| **1** | Replace **MultiHeadSelfAttentionLayer** with **SparseGAT + sparse adjacency** (author lite style) | NB3 (NB2 if matrix format changes) | Biggest training speed/RAM reduction | NB2 → NB3 |
| **2** | Set **`MIN_TOP_NEIGHBORS = 10`** and apply in NB2 graph build | `fgat_config.py`, NB2 | Smaller item–item graph; usually small accuracy drop | NB2 → NB3 |
| **3** | Lighter features: **`resnet50`**, **`MAX_TEXT_LENGTH = 128`** (or 256); rebuild cache | `fgat_config.py`, NB1 | NB1 is the heavy step; makes first run feasible on 16GB | NB1 → NB3 |
| **4** | **Train item fusion** (visual + text → 64d), not random untrained `Linear` | NB1 Stage 2C | Main accuracy safeguard when going lite | NB1 → NB3 |
| **5** | Add **`lite` / `full` config profile**; always report **test** metrics (Stage 4I) | `fgat_config.py`, NB3, README | One switch for low-RAM runs; test validates lite vs full | NB3 |

**Rule:** Change **one priority at a time**, then compare **test** HR/NDCG/compat before the next step.

---

## Do not change (stability)

- Train-only `user_outfit_train_index` (no val/test leak in graph propagation)
- Decoupled compat: `COMPAT_DETACH_INPUT=True`, base `item_embs` only
- FITB format (full pos outfit + hard-swapped neg outfits)
- Exclude train outfits from eval negatives (`user_known_outfits_idx`)

---

## Done = “Lightweight HFGAT”

All of the following:

1. **#1 + #2 + #3** implemented (sparse model, top-10 neighbors, lighter NB1)
2. **#4** keeps test metrics near current stable
3. **#5** documents profile + publishes test results

Suggested name: **Lightweight HFGAT — Active User Pipeline (StableCompat)**

---

## Reference configs (targets)

```python
# lite profile (proposal)
MIN_TOP_NEIGHBORS = 10
IMAGE_BACKBONE = "resnet50"
MAX_TEXT_LENGTH = 128
COMPAT_DETACH_INPUT = True
LAMBDA_COMP = 0.15
```

Keep `LEARNABLE_EMBEDDINGS = True`, `EVAL_NEG_SAMPLES = 50`, and early-stop on `HR@K` unless test ablation says otherwise.
