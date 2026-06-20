# FGAT Testing Plan — v1.3 Series

Step-by-step action plan for config and logic experiments on the active-user pipeline.

Use this document when running each phase manually or with AI assistance. **Change one phase at a time**; record results before starting the next.

---

## Reference files

| File | Role |
|------|------|
| `fgat_config.py` | Shared hyperparameters (edit per branch) |
| `fgat-session1-active-user.ipynb` | NB1 — features + item/user/outfit embeddings |
| `fgat-session2-active-user.ipynb` | NB2 — graph matrices |
| `fgat-session3-active-user.ipynb` | NB3 — train / val / test |
| `AUTHOR_VS_CURRENT_COMPARISON.md` | Author vs current logic |
| `LIGHTWEIGHT_HFGAT_ROADMAP.md` | Longer-term lightweight goals |
| `output_fgat_active_user/` | Default local output root |

**Notebooks shorthand:** NB1, NB2, NB3.

---

## Global rules (all phases)

1. **One branch per phase** — create git branch before editing config or code.
2. **One main change per phase** — isolate cause of metric shifts.
3. **Do not compare HR/NDCG across different `EVAL_NEG_SAMPLES`** (50 vs 99).
4. **Primary checkpoint metric:** Val **HR@10** (`EARLY_STOP_METRIC="HR@K"`). Do not pick model by `val_total_loss`.
5. **Always report Test metrics** from NB3 **Stage 4I** in addition to best val epoch.
6. **Keep stable logic** unless the phase explicitly changes it:
   - Train-only `user_outfit_train_index` (no val/test leak in GAT propagation)
   - FITB: full pos outfit + 3 hard-swapped neg outfits
   - Eval excludes user's train outfits from negatives (`user_known_outfits_idx`)
7. After editing `fgat_config.py`, **reload config** in the notebook (`importlib.reload(cfg)` or restart kernel + run setup cells).

### Metrics to log every run

| Metric | Meaning |
|--------|---------|
| Val / Test **HR@10**, **NDCG@10** | Main ranking quality |
| **AUC** | Ranking separability |
| **compat_acc**, **compat_margin** | Outfit compatibility |
| **val_rec_loss**, **val_comp_loss** | Diagnostic only |
| Best epoch, early-stop epoch | Reproducibility |

### Baselines for comparison

| Branch | Role |
|--------|------|
| `kimnguyen-quoctran-v1.1` | Stable reference (HR@10 ~0.77 @ 40 ep, 50 eval negs) |
| `kimnguyen-quoctran-v1.2` | Stronger reg + 99 eval negs (not directly comparable HR) |
| **`kimnguyen-quoctran-v1.3`** | **New baseline for this plan** |

---

## Run order (recommended)

```text
Phase 1 → v1.3      (config baseline)              [READY]
Phase 2 → v1.3.1    (trained item fusion)          [needs NB1 code]
Phase 3 → v1.3.2    (top-10 item neighbors)        [needs NB2 code]
Phase 4 → v1.4      (compat coupled: DETACH=False) [after v1.3]
Optional → v1.3.3   (v1.3.1 + v1.3.2 if both win)  [later]
```

---

## Shared config — Phase 1 baseline (`v1.3`)

All phases **inherit these** unless noted otherwise.

```python
# fgat_config.py — v1.3 baseline
RANDOM_SEED = 42
MIN_USER_INTERACTIONS = 4
SPLIT_MODE = "per_user"
MIN_TOP_NEIGHBORS = None          # Phase 3 changes to 10

# Model
EMBED_DIM = 64
NUM_HEADS = 4
DROPOUT = 0.3
MAX_OUTFIT_ITEMS_FOR_COMP = 10

# Training
EPOCHS = 50
LR = 0.001
WEIGHT_DECAY = 1e-5
LAMBDA_COMP = 0.3
COMPAT_DETACH_INPUT = True
COMPAT_BPR_MARGIN = 0.0
COMPAT_LR_MULT = 1.0
BATCH_SIZE = 512
NEG_PER_POS = 3
PATIENCE = 10
SCHEDULER_PATIENCE = 5
LEARNABLE_EMBEDDINGS = True
EVAL_EVERY = 1
EARLY_STOP_METRIC = "HR@K"
TOP_K = 10
EVAL_NEG_SAMPLES = 50
```

**Unchanged from current defaults:** feature extraction (`resnet152`, `bert-base-chinese`), `FORCE_REBUILD_FEATURES=False`.

---

# Phase 1 — `kimnguyen-quoctran-v1.3`

**Goal:** Config-only baseline — stronger compat weight + v1.1-style regularization + moderate train negs + 50 eval negs.

**Status:** Ready (no new code required).

**Depends on:** Existing NB1 + NB2 outputs in `output_fgat_active_user/`.

### What changes vs v1.1 / v1.2

| Setting | v1.1 | v1.2 | v1.3 |
|---------|------|------|------|
| `LAMBDA_COMP` | 0.15 | 0.15 | **0.3** |
| `NEG_PER_POS` | 1 | 5 | **3** |
| `WEIGHT_DECAY` | 1e-5 | 1e-4 | **1e-5** |
| `DROPOUT` | 0.3 | 0.4 | **0.3** |
| `EVAL_NEG_SAMPLES` | 50 | 99 | **50** |
| `EPOCHS` | 40 | 40 | **50** |

### Action items

- [ ] **1.1** Create branch: `git checkout -b kimnguyen-quoctran-v1.3`
- [ ] **1.2** Edit `fgat_config.py` to v1.3 baseline values (table above)
- [ ] **1.3** Verify with `python -c "import fgat_config as c; c.print_config_summary()"`
- [ ] **1.4** Open `fgat-session3-active-user.ipynb`
- [ ] **1.5** Run NB3 cells in order:

| Step | Stage | Required? | Why |
|------|-------|-----------|-----|
| 1 | §0 Install | If fresh kernel | Dependencies |
| 2 | §1 Imports & Setup | **Yes** | Reload `fgat_config` |
| 3 | 4A Load subsample | Yes | Metadata |
| 4 | 4B Load embeddings & matrices | Yes | Tensors |
| 5 | 4C Splits + FITB | Skip if unchanged | Same `SPLIT_MODE` / subsample |
| 6 | **4D** Dataset & DataLoader | **Yes** | `NEG_PER_POS=3` |
| 7 | **4E** Model H_HFGAT | **Yes** | `DROPOUT=0.3` |
| 8 | **4F** Loss & Evaluation | Yes | Uses config flags |
| 9 | **4G** Init model & optimizer | **Yes** | Fresh weights |
| 10 | **4H** Training loop | **Yes** | Main run |
| 11 | **4I** Test evaluation | **Yes** | Final metrics |
| 12 | 4J Training curves | Optional | Plot |

- [ ] **1.6** Save artifacts:
  - `output_fgat_active_user/models/best_model.pt`
  - `output_fgat_active_user/models/training_curves.png`
  - Copy epoch log / notebook output to a results note

### Expected outcomes

| Metric | Expectation |
|--------|-------------|
| HR@10 / NDCG@10 | Similar to v1.1 (± small swing); comparable because 50 eval negs |
| compat_acc / compat_margin | **Higher** than v1.1 (λ=0.3) |
| val_comp_loss | **Lower** than v1.1 |
| val_rec_loss | Similar band to v1.1 |

### Success criteria

- Val HR@10 ≥ v1.1 − ~0.02 **or** clear compat gain with ≤ small HR drop
- No train/val compat divergence (val_comp stable or improving)
- Test 4I metrics recorded as **official v1.3 baseline**

### AI handoff prompt (Phase 1)

```text
Branch: kimnguyen-quoctran-v1.3
Apply v1.3 config in fgat_config.py (see TESTING_PLAN_v1.3.md Phase 1).
Re-run NB3 from Imports through 4I. Do not change NB1/NB2.
Compare val/test HR@10, NDCG@10, compat_acc vs kimnguyen-quoctran-v1.1.
```

---

# Phase 2 — `kimnguyen-quoctran-v1.3.1`

**Goal:** Replace random untrained item fusion (NB1 Stage 2C) with **trained** visual+text → 64d projection.

**Status:** **Blocked until NB1 Stage 2C training logic is implemented.**

**Depends on:** Phase 1 complete (use v1.3 config + compare against v1.3).

### Config

Same as **Phase 1 (v1.3)** — no additional config changes.

### Prerequisites (code work — before running)

- [ ] **2.0a** Implement trained fusion in NB1 **Stage 2C** (replace one-pass random `Linear` + `.detach()`)
- [ ] **2.0b** Define training objective (pick one and document it), e.g.:
  - **Option A:** Contrastive — items in same outfit closer than random items
  - **Option B:** Autoencoder / reconstruction on concatenated features
  - **Option C:** Supervised proxy aligned with outfit co-occurrence graph
- [ ] **2.0c** Add config knobs if needed (e.g. `FUSION_EPOCHS`, `FUSION_LR`) — optional, can live in NB1 cell for now
- [ ] **2.0d** Add `FORCE_REBUILD_ITEM_EMBEDDINGS=True` **or** document manual cache delete (see below)

### Cache invalidation (critical)

NB1 Stage 2C **skips rebuild** if this file exists:

```text
output_fgat_active_user/embeddings/item_embeddings.npy
output_fgat_active_user/embeddings/item_embeddings.csv   # if present
```

After new fusion logic, **delete those files** before re-running 2C.

Also re-run **Stage 2D** (user/outfit embeddings depend on item embeddings):

```text
output_fgat_active_user/embeddings/user_embeddings.npy
output_fgat_active_user/embeddings/outfit_embeddings.npy
```

**Do not delete** unless regenerating:

- `cache/item_features.pt` — ResNet/BERT features unchanged
- `subsample/*` — same active-user filter

### Action items

- [ ] **2.1** Create branch: `git checkout -b kimnguyen-quoctran-v1.3.1` (from v1.3)
- [ ] **2.2** Confirm `fgat_config.py` matches Phase 1 baseline
- [ ] **2.3** Implement / merge NB1 Stage 2C fusion training (prerequisite)
- [ ] **2.4** Delete stale embedding files (list above)
- [ ] **2.5** Run NB1:

| Step | Stage | Required? |
|------|-------|-----------|
| 1 | §0, §1 Setup | Yes |
| 2 | 1A–1D | Skip if subsample exists |
| 3 | 2A Features | Skip if `item_features.pt` exists |
| 4 | **2C** Item fusion | **Yes** |
| 5 | **2D** User & outfit embeddings | **Yes** |

- [ ] **2.6** Run **full NB2** (Stage 1 → 3A → 3B → 3C → summary)  
  Matrices unchanged in logic but safe to re-run; skip if you confirm item set identical.
- [ ] **2.7** Run **full NB3** (§1 → 4A → 4B → 4C → 4D → 4E → 4F → 4G → 4H → 4I)  
  **4C required** if splits depend on embedding paths; minimum **4B onward** after new embeddings.
- [ ] **2.8** Record test metrics vs **v1.3** (not v1.1)

### Expected outcomes

| Metric | Expectation |
|--------|-------------|
| HR@10 / NDCG@10 | **Potential lift** — largest accuracy lever in roadmap |
| compat_acc | May improve (better item geometry) |
| Training time | NB1 longer (fusion training epoch loop) |

### Success criteria

- Test HR@10 or NDCG@10 **>** v1.3 baseline
- No regression on compat_acc vs v1.3

### AI handoff prompt (Phase 2)

```text
Branch: kimnguyen-quoctran-v1.3.1
Config: same as v1.3 (TESTING_PLAN_v1.3.md).
Task 1: Implement trained item fusion in NB1 Stage 2C (visual+text → 64d).
Task 2: Delete old item/user/outfit embedding npy files.
Task 3: Re-run NB1 (2C, 2D) → NB2 → NB3 (4A–4I).
Compare test metrics to v1.3 baseline.
Document fusion loss and training epochs used.
```

---

# Phase 3 — `kimnguyen-quoctran-v1.3.2`

**Goal:** Sparsify item–item graph to **top-10 neighbors** per item (`MIN_TOP_NEIGHBORS=10`).

**Status:** **Blocked until NB2 Stage 3C applies `MIN_TOP_NEIGHBORS`.**  
Today the config is imported but **not used** in graph build (“not applied yet” in NB2).

**Depends on:** Phase 1 complete. Independent of Phase 2 (can run in either order after blocker removed).

### Config change vs v1.3

```python
MIN_TOP_NEIGHBORS = 10   # was None
```

All other settings = **Phase 1 baseline**.

### Prerequisites (code work — before running)

- [ ] **3.0** Implement top-K sparsification in NB2 **Stage 3C** after full item–item weights are computed:
  - For each item row, keep only top `MIN_TOP_NEIGHBORS` edges by weight
  - Symmetrize if needed (undirected graph)
  - Log `nnz` before/after for verification

### Cache invalidation

Delete cached matrix so NB2 rebuilds:

```text
output_fgat_active_user/matrices/item_item_matrix.npz
```

### Action items

- [ ] **3.1** Create branch: `git checkout -b kimnguyen-quoctran-v1.3.2` (from v1.3)
- [ ] **3.2** Set `MIN_TOP_NEIGHBORS = 10` in `fgat_config.py`
- [ ] **3.3** Implement NB2 Stage 3C top-K logic (prerequisite)
- [ ] **3.4** Delete `item_item_matrix.npz`
- [ ] **3.5** Run NB2: §1 → Stage 1 → 3A → 3B → **3C** → summary  
  Confirm log shows reduced `nnz` vs v1.3 (~73k → lower)
- [ ] **3.6** Run NB3: §1 → 4A → **4B** → 4C (optional) → 4D → 4E → 4F → 4G → 4H → 4I
- [ ] **3.7** Compare test metrics vs **v1.3**

### Expected outcomes

| Metric | Expectation |
|--------|-------------|
| Training speed / RAM | **Better** (fewer item–item edges) |
| HR@10 / NDCG@10 | Slight drop possible; often small if top-10 preserves signal |
| item_item nnz | **Clearly lower** than v1.3 |

### Success criteria

- `item_item_matrix.nnz` reduced vs v1.3
- Test HR@10 within ~0.02 of v1.3 **or** acceptable speed/memory tradeoff documented

### AI handoff prompt (Phase 3)

```text
Branch: kimnguyen-quoctran-v1.3.2
Config: v1.3 baseline + MIN_TOP_NEIGHBORS=10.
Task 1: Wire MIN_TOP_NEIGHBORS into NB2 Stage 3C item-item graph build.
Task 2: Delete item_item_matrix.npz, re-run NB2 then NB3 (4B–4I).
Report nnz before/after and test HR/NDCG vs v1.3.
```

---

# Phase 4 — `kimnguyen-quoctran-v1.4`

**Goal:** Test **coupled compatibility** — compat gradients flow into base `item_embs` (author-style risk experiment).

**Status:** Ready after Phase 1 (config + NB3 only). **Run last** — highest instability risk.

**Note:** Original plan said `COMPAT_DETACH_INPUT=True` — that duplicates v1.3. Phase 4 correctly uses **`False`**.

### Config change vs v1.3

```python
COMPAT_DETACH_INPUT = False   # was True
LAMBDA_COMP = 0.3             # keep same for fair comparison
```

All other settings = **Phase 1 baseline**.

### Action items

- [ ] **4.1** Create branch: `git checkout -b kimnguyen-quoctran-v1.4` (from v1.3)
- [ ] **4.2** Set `COMPAT_DETACH_INPUT = False` in `fgat_config.py`
- [ ] **4.3** Run NB3:

| Step | Stage | Required? | Why |
|------|-------|-----------|-----|
| 1 | §1 Imports & Setup | Yes | Reload config |
| 2 | 4A, 4B | Yes | Load data |
| 3 | 4C | Skip if unchanged | |
| 4 | 4D | Yes | Same negs |
| 5 | 4E, 4F | Yes | Compat path uses detach flag in 4F |
| 6 | 4G, 4H, 4I | Yes | Train + test |

- [ ] **4.4** Watch for **train compat ↓ but val compat ↑** (known failure mode)
- [ ] **4.5** Compare vs **v1.3** — ranking metrics primary

### Expected outcomes

| Metric | Expectation |
|--------|-------------|
| train_comp_loss | Likely **lower** |
| val_comp_loss | May **rise** after early epochs |
| compat_acc | May rise train-side; val unstable |
| HR@10 / NDCG@10 | **Risk of plateau or drop** vs v1.3 |

### Success criteria

- Document behavior even if HR drops (negative result is valid)
- Adopt `DETACH=False` only if test HR **and** compat both ≥ v1.3

### AI handoff prompt (Phase 4)

```text
Branch: kimnguyen-quoctran-v1.4
Config: v1.3 baseline but COMPAT_DETACH_INPUT=False.
Re-run NB3 from Imports through 4I.
Compare train vs val comp_loss, compat_acc, HR@10, NDCG@10 against v1.3.
Flag any train↓ val↑ compat divergence.
```

---

# Optional Phase 5 — `kimnguyen-quoctran-v1.3.3` (combined)

**Only if Phase 2 and Phase 3 both show individual gains.**

**Goal:** Trained item fusion **+** top-10 neighbors + v1.3 training config.

### Config

Phase 1 baseline + `MIN_TOP_NEIGHBORS=10` + trained fusion (Phase 2 code).

### Action items

- [ ] Branch `kimnguyen-quoctran-v1.3.3`
- [ ] Merge Phase 2 (NB1 fusion) + Phase 3 (NB2 top-K) code
- [ ] Delete: `item_embeddings.npy`, `user/outfit embeddings`, `item_item_matrix.npz`
- [ ] Full pipeline: NB1 (2C, 2D) → NB2 → NB3 (4A–4I)
- [ ] Compare to v1.3, v1.3.1, v1.3.2 separately

---

## Results template (copy per phase)

```markdown
## Phase X — branch name — date

### Config snapshot
- LAMBDA_COMP:
- COMPAT_DETACH_INPUT:
- NEG_PER_POS:
- EVAL_NEG_SAMPLES:
- MIN_TOP_NEIGHBORS:
- (other changes):

### Best val epoch
- Epoch:
- Val HR@10:
- Val NDCG@10:
- compat_acc:
- compat_margin:
- val_rec_loss / val_comp_loss:

### Test (Stage 4I)
- Test HR@10:
- Test NDCG@10:
- Test compat_acc:

### vs baseline (v1.3)
- Δ HR@10:
- Δ NDCG@10:
- Δ compat_acc:

### Notes


### Decision
- [ ] Keep for next phase
- [ ] Revert change
- [ ] Needs retune
```

---

## Quick re-run cheat sheet

| What changed | Re-run |
|--------------|--------|
| Training hyperparams only (`LAMBDA_COMP`, `WEIGHT_DECAY`, `NEG_PER_POS`, `DROPOUT`, `EPOCHS`, `EVAL_NEG_SAMPLES`, `COMPAT_DETACH_INPUT`) | NB3: §1 → 4D* → 4E* → 4F → 4G → 4H → 4I |
| Item embeddings (fusion) | NB1 2C+2D → NB2 → NB3 4B–4I |
| Item–item graph (top-K) | NB2 3C → NB3 4B–4I |
| Subsample / splits / `SPLIT_MODE` | NB1 1C+ → NB2 → NB3 full |
| ResNet/BERT features | NB1 2A+ → full pipeline |

\* 4D if `NEG_PER_POS` changed; 4E if `DROPOUT` / model arch changed.

---

## Phase status summary

| Phase | Branch | Ready? | Blocker |
|-------|--------|--------|---------|
| 1 | `kimnguyen-quoctran-v1.3` | **Yes** | None |
| 2 | `kimnguyen-quoctran-v1.3.1` | No | NB1 fusion training + delete embedding cache |
| 3 | `kimnguyen-quoctran-v1.3.2` | No | NB2 top-K in Stage 3C + delete item_item matrix |
| 4 | `kimnguyen-quoctran-v1.4` | **Yes** | Use `COMPAT_DETACH_INPUT=False` (not True) |
| 5 | `kimnguyen-quoctran-v1.3.3` | No | Phases 2 + 3 success |

---

*Last updated: June 2026 — aligned with `fgat_config.py` and active-user notebooks.*
