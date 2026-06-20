# Author vs Current FGAT / H-HFGAT Comparison

Revalidation of configs and training logic between the **author original** notebook and the **current active-user pipeline** (3 notebooks + shared `fgat_config.py`).

| | Author | Current |
|---|--------|---------|
| **Source** | `authors/Original_FGAT_Implementation_Final.ipynb` | `fgat-session1-active-user.ipynb`, `fgat-session2-active-user.ipynb`, `fgat-session3-active-user.ipynb` |
| **Config** | Inline in notebook cells | `fgat_config.py` |
| **Branch context** | — | `kimnguyen-quoctran-v1.2` (tuning); stable baseline `kimnguyen-quoctran-v1.1` |

---

## A. Hyperparameters

| Setting | Author | Current (`fgat_config.py`) | Notes |
|---------|--------|----------------------------|-------|
| `LR` | 0.001 | 0.001 | Same |
| `WEIGHT_DECAY` | **1e-5** | **1e-4** | Current regularizes more |
| `LAMBDA_COMP` | **0.5** | **0.15** | Author weights compatibility loss higher |
| `BATCH_SIZE` | 512 | 512 | Same (author also uses 1024 for some DataLoaders) |
| `NEG_PER_POS` | **1** | **5** | Current uses harder BPR training |
| `EVAL_NEG_SAMPLES` | **50** | **99** | Current evaluates with more negatives |
| `DROPOUT` | 0.3 (model init) | **0.4** | Current regularizes more |
| `EPOCHS` | **100** | 40 | Shorter run in current pipeline |
| `PATIENCE` (early stop) | **4** | 10 | Current waits longer before stopping |
| `EMBED_DIM` | 64 | 64 | Same |
| `NUM_HEADS` | 4 | 4 | Same |
| Early-stop metric | **Val HR@10** | **Val HR@10** (`EARLY_STOP_METRIC="HR@K"`) | Same intent |
| LR scheduler | ReduceLROnPlateau, patience 4 | ReduceLROnPlateau, `SCHEDULER_PATIENCE=5` | Similar |
| `MIN_TOP_NEIGHBORS` | Not used | `None` (not wired in NB2 yet) | Effectively same |
| `LEARNABLE_EMBEDDINGS` | Frozen tensors at train time | **`True`** | Current fine-tunes base embeddings |
| `COMPAT_DETACH_INPUT` | N/A (compat on GAT output) | **`True`** | Current stability fix |
| `COMPAT_BPR_MARGIN` | 0 (standard BPR) | 0 | Same |
| Active-user filter | Full / sampled runs | **`MIN_USER_INTERACTIONS=4`** | Smaller user graph (~25k users) |

### v1.1 stable baseline (for reference)

The branch `kimnguyen-quoctran-v1.1` used: `LAMBDA_COMP=0.15`, `WEIGHT_DECAY=1e-5`, `DROPOUT=0.3`, `NEG_PER_POS=1`, `EVAL_NEG_SAMPLES=50`, `COMPAT_DETACH_INPUT=True`. Best val HR@10 ≈ 0.77 @ 40 epochs.

**v1.2** changes regularization and negatives only (`WEIGHT_DECAY`, `NEG_PER_POS`, `EVAL_NEG_SAMPLES`, `DROPOUT`). HR/NDCG numbers are **not directly comparable** to v1.1 because of the 99-negative eval protocol.

---

## B. Model architecture

| Component | Author | Current |
|-----------|--------|---------|
| Item GAT | `MultiHeadSelfAttentionLayer` | Same |
| Item → outfit | `scatter_add` + `nn.Linear` | Same |
| User aggregation | `UserAttentionAggregator` | Same |
| `CompatibilityScorer` | 6 pairwise views, `tanh`, `LayerNorm` | Same + **padding mask** + dropout in scorer |
| `score_recommendation` | Raw dot product: `sum(user * outfit, dim=1)` | **L2-normalized dot product** (cosine-style) |
| `H_HFGAT` forward outputs | `item_updated`, `outfit_updated`, `user_updated` | Same structure |

**Verdict:** Same H-HFGAT skeleton. Current adds small correctness/stability improvements (mask, cosine scoring).

---

## C. Data pipeline and splits

| Area | Author | Current |
|------|--------|---------|
| Notebooks | Single monolithic notebook | NB1 (features) → NB2 (graphs) → NB3 (train/eval) |
| Item features | ResNet + BERT → untrained `Linear(2816→64)` + **`.detach()`** | ResNet152 + BERT → untrained fusion in NB1 (same weakness) |
| User–outfit rec split | **90/10** line split on `train_uo.txt` | **Per-user 80/10/10** (`SPLIT_MODE="per_user"`) |
| FITB (compat) split | **80/10/10** outfit-level | **80/10/10** outfit-level | Aligned |
| Active users | No `MIN_USER_INTERACTIONS` filter in main run | Users with ≥ 4 interactions |

---

## D. Training logic (critical differences)

| Area | Author | Current | Assessment |
|------|--------|---------|------------|
| **Graph at train/eval** | Full `user_outfit` graph (includes val/test edges) | **Train-only** `user_outfit_train_index` for propagation | **Current is stricter** (less leakage) |
| **Eval negatives** | 1 pos + 50 sampled negs; **does not exclude** user's train outfits | 1 pos + 99 negs; **excludes** outfits in `user_known_outfits_idx` | **Current is stricter** |
| **Compat training input** | **`item_updated`** (GAT output) | **Base `item_embs`** with `COMPAT_DETACH_INPUT=True` | **Current is stable**; blending GAT into compat caused train↓ val↑ in our runs |
| **Compat eval accuracy** | Uses **base `item_embs`** | Uses **base `item_embs`** | Author is **inconsistent** (train on GAT, eval on base) |
| **FITB format** | Full positive outfit + 3 negative outfits (hard item swap) | Same | Aligned |
| **Compat padding** | No explicit pad mask in scorer forward | Masked padding in `CompatibilityScorer` | Current avoids spurious `compat_acc=1.0` |
| **Gradient clipping** | Model parameters only | **`OPTIM_PARAMS`** (model + learnable embeddings) | Current clips all trained tensors |
| **Rec loss** | BPR on outfit embeddings | BPR + normalized scores | Same objective family |

---

## E. Checkpoint and monitoring

| Behavior | Author | Current |
|----------|--------|---------|
| Best model selection | **Val HR@10** | **Val HR@10** (`EARLY_STOP_METRIC`) |
| Logged losses | `train_total`, `val_rec`, `val_comp`, `val_total` | Same family + `compat_margin`, `compat_acc` |
| Stop when | HR@10 no improvement for `patience` epochs | Same |
| Expect train/val loss to track HR? | No — author also early-stops on HR, not `val_total_loss` | No — val loss can plateau while HR still improves |

---

## F. Overall verdict

| Question | Answer |
|----------|--------|
| Wrong direction? | **No** |
| Same as author? | **~80%** architecture and protocol |
| Better than author in places? | **Yes:** anti-leak graph, eval train-outfit exclusion, stable compat path, FITB padding mask |
| Intentionally different? | **Yes:** decoupled compat, cosine rec scores, learnable embeddings, v1.2 hyperparams, per-user split |
| Metrics comparable to author notebook? | **No** — different split, user filter, neg count (50 vs 99), and eval rules |
| Metrics fake or leaked? | **No** — validated against train-only graph, proper FITB, and eval exclusion |

**One-line summary:** Current pipeline = author's H-HFGAT + deliberate stability/leak fixes + v1.2 training tweaks. Not paper-identical; numbers are not apples-to-apples with the author notebook without matching protocol.

---

## G. Closest author reproduction (experimental)

To approximate the author notebook **without** reverting stability fixes, change only hyperparams in `fgat_config.py` and re-run NB3 (and NB2 if graph settings change):

```python
LAMBDA_COMP = 0.5
WEIGHT_DECAY = 1e-5
DROPOUT = 0.3
NEG_PER_POS = 1
EVAL_NEG_SAMPLES = 50
PATIENCE = 4
EPOCHS = 100
```

Optional (may reintroduce compat instability):

```python
COMPAT_DETACH_INPUT = False  # compat on GAT item_upd — use with caution
LEARNABLE_EMBEDDINGS = False
```

Also align split protocol: set `SPLIT_MODE` to edge-level or replicate 90/10 on `train_uo.txt` if strict author parity is required.

---

## H. Re-run order after config changes

| Changed | Re-run |
|---------|--------|
| NB3 training only | Config cell → 4D (if `NEG_PER_POS` changed) → 4E → 4F → 4G → 4H → 4I |
| NB1 features | NB1 → NB2 → NB3 |
| NB2 graphs / `MIN_TOP_NEIGHBORS` | NB2 → NB3 |

---

## I. Related docs

- `LIGHTWEIGHT_HFGAT_ROADMAP.md` — priorities for a lighter model without breaking current stability choices
- `fgat_config.py` — single source of truth for current hyperparameters

---

*Last updated: June 2026 — compared against `authors/Original_FGAT_Implementation_Final.ipynb` and `fgat_config.py` (v1.2).*
