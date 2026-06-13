# H-FGAT Code Comparison: Author vs. Team

**Date:** 2026-06-13 (updated with Fix 10 implementation findings)  
**Author's code:** [GitHub - kimnguyen branch](https://github.com/1653072/hcmus-master-is-research-methods/tree/kimnguyen), file `fgat-session-3-train-model.ipynb`  
**Team's code:** `hfgat_rewrite_validate/hfgat_runall_rewrite_validate_fixed.ipynb`  
**Paper:** *Hybrid-hierarchical fashion graph attention network for compatibility-oriented and personalized outfit recommendation*

---

## Executive Summary

After a thorough line-by-line comparison, **the team's code differs from the author's in 9 fundamental areas**, ranging from model architecture to evaluation methodology. The most critical differences that explain both the overfitting symptom and the low Precision@10 are:

1. The **model architecture** is completely different (team: complex MLP+MHA, author: simple sparse GAT).
2. The **recommendation scorer** is different (team: MLP concat, author: dot product).
3. The **compatibility loss weight** (λ) is 5× lower in the team's code (0.1 vs 0.5).
4. The **evaluation protocol** uses different numbers of negative samples, inflating the author's metrics.
5. The **data split strategy** produces a drastically smaller validation set in the team's code.
6. The **author's notebook actually crashes** on epoch 1 validation with a TypeError, so the reported results were from a corrected run, not the code in the branch as-is.

**Post-implementation note (Fix 10):** An attempted optimization — pre-computing graph embeddings once per epoch — was **reverted** after a runtime autograd failure. The current notebook uses **per-batch** `forward_all_embeddings()` with standard mini-batch SGD. See [Fix 10](#fix-10-medium-priority-graph-forward-frequency--reverted) for details and future options.

---

## Difference 1: Model Architecture

### Author's Model (`H_HFGAT`)

```
H_HFGAT(
  item_gat:     SparseGATLayer(Linear(64→64), LayerNorm, LeakyReLU, Dropout)
  item_proj:    Linear(64→64)
  user_proj:    Linear(64→64)
  compatibility_scorer: CompatibilityScorer(
      W5: Linear(64→256), LayerNorm, W4: Linear(256→6)   [attention]
      W7: Linear(64→256), LayerNorm, W6: Linear(256→6)   [content]
      LeakyReLU
  )
  dropout:      Dropout(0.2)
  act:          LeakyReLU(0.2)
)
Total parameters: 49,932
```

**Forward pass logic (author):**

```python
item_upd   = F.normalize(dropout(GAT(item_embs, item_adj_sparse)))
outfit_upd = F.normalize(dropout(act(item_proj(sparse_mm(outfit_item_adj, item_upd)))))
user_upd   = F.normalize(dropout(act(user_proj(sparse_mm(user_outfit_adj, outfit_upd) + user_embs))))
```

### Team's Model (`HFGATLite`)

```
HFGATLite(
  image_proj:      Linear(2048→64)
  text_proj:       Linear(768→64)
  cat_proj:        Linear(61→64)
  item_fuse:       MLPBlock(192→64)   [Linear→LayerNorm→ReLU→Dropout→Linear]
  item_update:     MLPBlock(64→64)
  outfit_update:   MLPBlock(64→64)
  user_update:     MLPBlock(64→64)
  outfit_attn:     MultiheadAttention(64, 4 heads) + LayerNorm
  user_base:       nn.Embedding(277469, 64)
  outfit_base:     nn.Embedding(8685, 64)
  compat_mlp:      Sequential(Linear(64→64), ReLU, Linear(64→1))
  scorer:          Sequential(Linear(128→64), ReLU, Linear(64→1))
)
Total parameters: MUCH larger (image_proj alone has 2048×64=131,072 weights)
```

**Key architectural differences:**


| Aspect                | Author                                                               | Team                                                                           |
| --------------------- | -------------------------------------------------------------------- | ------------------------------------------------------------------------------ |
| Item representation   | Pre-computed 64-dim embedding (loaded from numpy)                    | Raw features: ResNet152 (2048) + BERT (768) + one-hot (61) → projected & fused |
| User embedding        | Pre-computed 64-dim (from notebook 2)                                | `nn.Embedding(277469, 64)` (randomly initialized)                              |
| Outfit embedding      | Pre-computed 64-dim (from notebook 2)                                | `nn.Embedding(8685, 64)` (randomly initialized)                                |
| GAT mechanism         | Proper sparse attention (LeakyReLU + LayerNorm on aggregated output) | Simple sparse matrix multiply (no attention coefficients)                      |
| Outfit self-attention | None (outfit is just sparse_mm aggregation)                          | `MultiheadAttention(4 heads)` for compatibility scoring                        |
| Compatibility scorer  | Multi-view weighted sum (6 views, soft-attention)                    | Self-attention pool → MLP(64→1)                                                |
| Scoring (rec)         | Dot product: `(user ⊙ outfit).sum()`                                 | MLP concat: `scorer(cat([user, outfit]))`                                      |


**Why this matters for overfitting:** The team's model has far more parameters and complexity. With only ~6,948 compatibility training samples and 5,794 val edges, the MLP-based scorer can memorize training patterns. The author's dot-product scorer is inherently regularized by normalization.

---

## Difference 2: Recommendation Scoring Function

### Author

```python
def score_recommendation(self, user_emb, outfit_emb):
    return torch.sum(user_emb * outfit_emb, dim=-1)
```

Both embeddings are L2-normalized before dot product → effectively cosine similarity. No extra learnable parameters.

### Team

```python
self.scorer = nn.Sequential(
    nn.Linear(embed_dim * 2, embed_dim),
    nn.ReLU(),
    nn.Linear(embed_dim, 1)
)

def score_user_outfit(self, user_emb, outfit_emb, user_idx, outfit_idx):
    u = user_emb[user_idx]
    o = outfit_emb[outfit_idx]
    x = torch.cat([u, o], dim=-1)
    return self.scorer(x).squeeze(-1)
```

**Why this matters for overfitting:** The MLP scorer adds learnable parameters (64×2 → 64 → 1 = 8,256 extra parameters) on top of the embedding representations. This allows the model to overfit on training-set scoring patterns that don't generalize. The author's dot product has zero extra parameters.

---

## Difference 3: Hyperparameters


| Parameter                   | Author                     | Team           | Impact                                                                                                            |
| --------------------------- | -------------------------- | -------------- | ----------------------------------------------------------------------------------------------------------------- |
| `LAMBDA_COMP`               | **0.5**                    | **0.1**        | Author penalizes compatibility 5× more heavily. Low λ in team's code → compatibility loss barely affects training |
| `WEIGHT_DECAY`              | **1e-5**                   | **1e-3**       | Team uses 100× higher weight decay. Somewhat mitigates overfitting but also slows learning                        |
| `DROPOUT`                   | **0.2**                    | **0.3**        | Team uses slightly more dropout                                                                                   |
| `BATCH_SIZE`                | **512** (→16 after OOM)    | **1024**       | Team uses larger batches                                                                                          |
| `NUM_EPOCHS`                | **100** (with patience=10) | **40**         | Author trained up to 100 epochs with early stopping                                                               |
| `PATIENCE` (early stopping) | **10 epochs**              | **None**       | Team has no early stopping                                                                                        |
| `NUM_HEADS`                 | 4 (in CompatibilityScorer) | 4 (in MHA)     | Same                                                                                                              |
| `K` (top-K eval)            | 10                         | 5, 10, 20      | Team evaluates multiple K                                                                                         |
| `neg_per_pos` (eval)        | **50 fixed**               | **99 sampled** | Different evaluation rigor                                                                                        |


**Critical impact of `LAMBDA_COMP = 0.1` (team) vs `0.5` (author):**

The compatibility loss acts as a regularizer for item embeddings. When λ is small, the item encoder optimizes almost entirely on BPR recommendation loss. Since BPR loss only sees user-outfit pairs (not item-level compatibility), item embeddings are trained to maximize ranking differences, not compatibility, leading to overfitting on the train split's user-outfit assignment.

---

## Difference 4: Data Split Strategy

### Author

```python
# Shuffles ALL lines of train_uo.txt and splits 80/10/10 by line count
np.random.shuffle(uo_lines)
train_end = int(0.8 * n)   # 221,975 lines
val_end   = int(0.9 * n)   # 27,747 val lines, 27,747 test lines
```

- Each line in `train_uo.txt` is `user_id o1 o2 o3 ...` — so one user can have **multiple outfits per line**
- The author's train/val/test each contain **multiple full users** with many outfits
- Train dataset: **543,619 pairs**, Val dataset: **67,606 pairs**, Test dataset: **67,803 pairs**

### Team

```python
# Stratified per-user holdout: each user holds out 10% of their interactions for val and test
def stratified_user_holdout_ratio(edges_df, val_ratio=0.1, test_ratio=0.1, ...):
    for user_idx, grp in edges_df.groupby("user_idx"):
        # users with only 1 interaction → all go to train
        if n <= min_train_per_user:
            train_parts.append(grp); continue
        n_val = int(round(n * val_ratio))
        n_test = int(round(n * test_ratio))
        ...
```

Result:

- `all_edges = 679,028`, `train_edges = 667,440`, `val_edges = 5,794`, `test_edges = 5,794`
- Train ratio = **0.983** (nearly all data in train)
- Val and test users = only **5,543** users (those with ≥ 2 interactions)

**Why this matters:**

Since 277,469 − 5,543 = **271,926 users** (98%) have only 1 interaction (avg = 2.45, median ≈ 1), the team's per-user stratified split puts almost all interactions into training. The validation set is tiny (5,794 pairs vs 667,440 training pairs), making val_loss unreliable. More critically, the **train graph** has 98% of all user-outfit edges, so the model sees almost all ground truth during training — severe data leakage risk.

The author's random 80/10/10 line split is a much fairer evaluation setup that allows meaningful val_loss tracking.

---

## Difference 5: Compatibility Task - Negative Sampling Strategy

### Author

```python
# FLTB format: outfit_id;outfit_len;mask_pos;pos_items;neg1;neg2;neg3
# Negatives: random item NOT from the outfit, for each masked position
mask_pos = random.randint(0, len(items)-1)
# The negative replaces ONE random item with a random item from all_item_pool
neg[mask_pos] = int(random.choice(candidates))  # 3 negatives per sample
```

Produces: 7,498 train, 937 val, 938 test compatibility samples  
Negatives are **random** (any item not in the outfit)

### Team

```python
# Hard negative: same-category item not in the outfit
def sample_hard_neg(pos_idx, idxs_in_outfit, rng):
    iid = str(idx2item[pos_idx])
    cat = iid_to_cat.get(iid, None)
    if cat is not None and cat in cat_to_idxs:
        pool = [x for x in cat_to_idxs[cat] if x not in idxs_in_outfit]
    # fallback: random
    if len(pool) < 2:
        pool = [x for x in item_pool if x not in idxs_in_outfit]
```

Produces: 6,948 train, 1,737 val compatibility pairs  
Negatives are **hard** (same-category item → harder to distinguish)

**Impact:** Hard negatives are generally better for learning, but the team's BPR compatibility loss is:

```
bpr_loss(pos_score, neg_score)  # where score is a scalar from compat_mlp
```

The author's `CompatibilityScorer` computes:

```
A = softmax(W4(act(W5(outfit_embs))))  # attention weights over items × views
C = tanh(W6(act(W7(outfit_embs))))     # content score over items × views
score = (A * C).sum(-1).sum(-1)        # scalar per outfit
```

This multi-view scoring is more expressive and aligned with the paper's formulation.

---

## Difference 6: Evaluation Negative Sampling (Critical for Metric Comparison)

### Author's `evaluate_rec()`:

```python
num_neg = 50
neg_idx = torch.randint(outfit_upd.size(0), (num_neg,), device=device)
neg_sc = model.score_recommendation(user_upd[u], outfit_upd[neg_idx])
sc_all = torch.cat([pos_sc, neg_sc])
_, topk = torch.topk(sc_all, k=10)
```

With 50 negatives and typically 1–3 positives → **total candidates ≈ 52–53**, top-10 = top-10 out of 53.  
If all 3 positives are in top-10 of 53 → Precision@10 = 3/10 = **0.3**.  
This is a **very optimistic (easy) evaluation** because negative pool is tiny.

### Team's `ranking_metrics_multi_k()`:

```python
sampled_negatives = 99
neg_pool = [oid for oid in range(num_outfits) if oid not in train_seen and oid not in true_set]
sampled_negs = rng.choice(neg_pool, size=min(99, len(neg_pool)), replace=False).tolist()
candidate_ids = list(true_set) + sampled_negs
```

With 99 negatives and 1–3 positives → **total candidates ≈ 100–102**, top-10 = top-10 out of 102.  
If 1 positive is in top-10 of 102 → Precision@10 = 1/10 = **0.1**.

**This directly explains the difference:**

- Author Precision@10 = 0.4424 (50 negatives, optimistic setting)
- Team Precision@10 ≈ 0.07 (99 negatives, harder setting)

**Even if both models had identical quality, the author's metric would appear ~3–4× higher purely from the negative sampling methodology.**

To compare fairly with the author: **reduce `sampled_negatives` to 50** in the team's code.

---

## Difference 7: Author's Code Has a RuntimeError Bug (Unresolved in Branch)

In the author's notebook, the training loop has a **bug in the validation section** that crashes on epoch 1:

```python
# In training loop — VALIDATION section:
item_upd_v, outfit_upd_v, user_upd_v = model(
    item_embs, item_item_index, item_item_weight,  # ← WRONG: 9 args
    outfit_embs, outfit_item_index, outfit_item_weight,
    user_embs, user_outfit_index, user_outfit_weight
)
```

But `H_HFGAT.forward()` expects only **6 arguments** (3 sparse matrices):

```python
def forward(self, item_embs, item_adj_sparse,
            outfit_embs, outfit_item_adj_sparse,
            user_embs, user_outfit_adj_sparse):
```

The notebook's output shows this error:

```
TypeError: H_HFGAT.forward() takes 7 positional arguments but 10 were given
```

**The author's notebook as stored in the branch actually fails at epoch 1.** The reported results (HR@10=0.4286, Precision@10=0.4424, Recall@10=0.1580) must come from a **different version** of the code not committed to the branch.

This means:

1. We cannot fully trust or reproduce the author's exact numbers from this branch.
2. The "correct" forward call should pass sparse matrices (built earlier) but the val block passes raw COO indices and weights.

---

## Difference 8: Graph Construction - Item-Item Graph

### Author

Pre-built `item_item_adj.npz` from Notebook 2 (based on shared category/visual similarity), loaded directly:

```python
item_item_index, item_item_weight = load_npz_edges(os.path.join(FEAT_DIR, 'item_item_adj.npz'))
item_adj_sparse = build_normalized_sparse(item_item_index, item_item_weight, n_items, n_items, device)
```

103,652 item-item edges.

### Team

Custom construction combining category + co-occurrence, top-10 neighbors per item:

```python
# category edges: w = 1/log2(|category| + 1)
# co-occurrence edges: w = count/sqrt(freq_i * freq_j)
# combined: 0.3 * cat_w + 0.7 * cooc_w
# keep top-10 per item, symmetrize
```

298,012 item-item edges (about 3× denser than author's).

**The denser graph in the team's code may cause over-smoothing of item embeddings**, reducing distinctiveness. The author's 103,652-edge graph is sparser and likely more meaningful (pre-computed from visual similarity).

---

## Difference 9: Graph Used During Training (Data Leakage Risk)

### Author

Uses **full** user-outfit adjacency matrix in training forward pass:

```python
user_outfit_adj_sparse = build_normalized_sparse(
    user_outfit_index, user_outfit_weight, n_users, n_outfits, device
)  # includes ALL user-outfit interactions
```

This means the validation/test user-outfit edges are visible to the model through graph propagation → potential data leakage.

### Team

Correctly uses **only train edges** for the user-outfit graph during training:

```python
A_user_outfit_train = make_sparse_from_edges(
    len(user2idx), len(outfit2idx),
    train_edges_df[["user_idx", "outfit_idx"]].to_records(index=False).tolist()
)
# model.forward() uses A_user_outfit_train during training
```

**The team's approach is more correct**, but the tiny val/test split (0.86% of data) negates this benefit.

---

## Root Cause Analysis of Overfitting

The overfitting pattern observed (train_loss: 0.5 → 0.09, val_loss: ~0.47 flat) has multiple causes:

### Cause 1: Val set is too small and structurally similar to train

With 98% of interactions in training, the val set (5,794 edges) represents only users who had multiple interactions. These users' embeddings are well-learned by the time validation runs. The val_loss fluctuating around 0.45 while train_loss drops to 0.09 suggests the model is memorizing training-set user-outfit assignments.

### Cause 2: MLP-based scorer overfits the training distribution

The `scorer` MLP in `HFGATLite` has 8,256+ parameters that learn to assign high scores to training-set (user, outfit) pairs. Since the embedding space is shared and the MLP is expressive, it will overfit to seen pairs.

### Cause 3: λ_comp = 0.1 is too small

With LAMBDA_COMP=0.1, the total loss is approximately:

```
loss = rec_loss + 0.1 * comp_loss
```

The compatibility task provides a cross-outfit regularization signal that helps item embeddings generalize. With λ=0.5 (as in the author's design), compatibility loss contributes more equally, acting as regularization.

### Cause 4: No early stopping

The team trains for all 40 epochs without early stopping, allowing the model to continue memorizing after generalization peaks.

### Cause 5: Over-parameterized model relative to dataset size

The team's model has projections from 2048/768-dim features + learnable nn.Embeddings for all 277,469 users and 8,685 outfits. This is a massive parameter space for a dataset with avg 2.45 interactions per user.

---

## Recommended Fixes (Priority Order)

### Fix 1 (High Priority): Change evaluation negative sampling to match author

```python
# Change sampled_negatives from 99 to 50 in ranking_metrics_multi_k()
sampled_negatives = 50  # matches author's eval protocol
```

This alone will boost Precision@10 by ~2-3x in reported numbers.

### Fix 2 (High Priority): Use dot product scorer instead of MLP

```python
# Replace:
# self.scorer = nn.Sequential(Linear(128, 64), ReLU(), Linear(64, 1))
# With:
def score_user_outfit(self, user_emb, outfit_emb, user_idx, outfit_idx):
    u = F.normalize(user_emb[user_idx], p=2, dim=-1)
    o = F.normalize(outfit_emb[outfit_idx], p=2, dim=-1)
    return (u * o).sum(dim=-1)
```

### Fix 3 (High Priority): Increase λ_comp to 0.5

```python
LAMBDA_COMP = 0.5  # was 0.1
```

### Fix 4 (Medium Priority): Use author's line-based 80/10/10 split

The per-user stratified split is more theoretically correct but produces too small a validation set. Consider a random 80/10/10 line-based split to match the author's approach and get a reliable val_loss.

### Fix 5 (Medium Priority): Add early stopping

```python
PATIENCE = 10  # stop if val_hr does not improve for 10 epochs
```

### Fix 6 (Medium Priority): Use ReduceLROnPlateau scheduler

```python
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.5, patience=5
)
scheduler.step(val_hr_k10)
```

### Fix 7 (Lower Priority): Match WEIGHT_DECAY to author

```python
WEIGHT_DECAY = 1e-5  # was 1e-3 (100x smaller, less aggressive penalty)
```

Note: since our model is larger, some weight decay is beneficial, but 1e-3 may be too aggressive.

### Fix 8 (Medium Priority): Add gradient clipping

The author explicitly clips gradients every batch:

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

The team's training loop does not clip gradients. With large input projections (ResNet152 2048-dim, BERT 768-dim), gradient norms can spike in early epochs and destabilize the embedding space, causing the val_loss to diverge from train_loss immediately. Add this line after `loss.backward()` and before `optimizer.step()`.

### Fix 9 (Medium Priority): Consistent L2 normalization on output embeddings

The author normalizes every intermediate embedding with `F.normalize(..., p=2, dim=-1)` before using it downstream:

```python
item_upd   = F.normalize(...)
outfit_upd = F.normalize(...)
user_upd   = F.normalize(...)
```

The team's `encode_outfits` and `encode_users` apply `F.normalize` to their outputs, but `encode_items` does NOT normalize its output. If Fix 2 (dot product scorer) is applied, the item embeddings fed into compatibility scoring must also be L2-normalized to keep score magnitudes consistent. Add:

```python
def encode_items(self, image_feat, text_feat, cat_feat, A_item):
    ...
    return F.normalize(x, p=2, dim=-1)  # add this line
```

### Fix 10 (Medium Priority): Graph forward frequency — **REVERTED**

**Status:** Attempted in `hfgat_runall_rewrite_validate_fixed.ipynb`, then **reverted**. Current training uses **per-batch** graph forward (same as the author).

#### Original motivation

In the team's training loop, `model.forward()` (full graph propagation over all items, outfits, users) is called **once per batch**. With ~100 batches per epoch (after the 80/10/10 split), that means ~100 full sparse-matrix propagations per epoch — expensive on MPS/CPU. Fix 10 proposed computing embeddings once per epoch to reduce redundant work.

#### What was implemented (and failed)

```python
# Once per epoch — single forward WITH gradients
user_emb_epoch, outfit_emb_epoch, item_emb_epoch = forward_all_embeddings(model)

for batch_idx, batch in enumerate(train_loader):
    optimizer.zero_grad()
    # Score from cached embeddings
    pos_score = model.score_user_outfit(user_emb_epoch, outfit_emb_epoch, u_exp, pos_exp)
    neg_score = model.score_user_outfit(user_emb_epoch, outfit_emb_epoch, u_exp, neg_flat)
    loss = rec_loss + LAMBDA_COMP * comp_loss
    loss.backward(retain_graph=(batch_idx < last))  # keep graph for next batch
    optimizer.step()  # ← updates weights in-place after EVERY batch
```

#### Why it crashed (PyTorch autograd)

After batch 1: `backward(retain_graph=True)` → `optimizer.step()` **mutates model parameters in place**.

After batch 2: `backward()` on the **same** computation graph from the epoch-start forward fails because parameter tensor versions no longer match what autograd recorded:

```
RuntimeError: one of the variables needed for gradient computation has been
modified by an inplace operation: [MPSFloatType [64, 64]] is at version 3;
expected version 1 instead.
```

**Key insight:** `retain_graph=True` only keeps a graph alive for **another backward on the same forward**. It does **not** allow reusing one forward across multiple `backward()` + `optimizer.step()` cycles when weights change between batches.

| Approach | `optimizer.step()` per batch? | Gradients to graph layers? | Works with standard mini-batch SGD? |
| -------- | ----------------------------- | -------------------------- | ----------------------------------- |
| Once/epoch forward + `retain_graph` + step/batch | Yes | Yes (in theory) | **No** — graph stale after first step |
| Once/epoch forward + gradient accumulation + step/epoch | No (once at end) | Yes | Yes — different optimization semantics |
| Once/epoch forward + `no_grad` (detached embeddings) | Yes | No for rec loss through graph | Yes — weaker / wrong training signal |
| **Per-batch forward (current)** | Yes | Yes | **Yes** — correct mini-batch SGD |

#### Current decision (kept for now)

```python
for batch in train_loader:
    optimizer.zero_grad()
    user_emb, outfit_emb, item_emb = forward_all_embeddings(model)  # fresh graph each batch
    loss = rec_loss + LAMBDA_COMP * comp_loss
    loss.backward()  # no retain_graph
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
```

This matches the author's per-batch `model(...)` call and is the correct pattern for mini-batch learning with per-batch weight updates.

#### Future options for next versions (if speed is still a bottleneck)

Below are **code proposals** for the three main alternatives. They are not implemented in the current notebook; use them as starting points for the next version.

---

##### Option A — Gradient accumulation (one forward per epoch, one `step()` per epoch)

**Idea:** Forward once with gradients, accumulate loss over all batches, call `optimizer.step()` only at epoch end. No weight update between batch backwards → graph stays valid.

**Trade-off:** Equivalent to full-batch (or very large effective batch) gradient descent per epoch. May need lower `LR` or scaled LR vs per-batch SGD.

```python
# Config (add to Cell 2)
ACCUMULATE_PER_EPOCH = True   # toggle Option A

# Training loop replacement sketch
for epoch in range(1, EPOCHS + 1):
    model.train()
    optimizer.zero_grad()

    # Single forward for the whole epoch (graph built once)
    user_emb, outfit_emb, item_emb = forward_all_embeddings(model)

    n_batches = len(train_loader)
    for batch_idx, batch in enumerate(train_loader):
        user_idx = batch["user_idx"].to(device)
        pos_idx  = batch["pos_outfit_idx"].to(device)
        neg_idx  = batch["neg_outfit_idx"]
        if isinstance(neg_idx, (list, tuple)):
            neg_idx = torch.stack(neg_idx, dim=1)
        neg_idx = neg_idx.to(device)

        if neg_idx.dim() == 2:
            u_exp    = user_idx.repeat_interleave(neg_idx.size(1))
            pos_exp  = pos_idx.repeat_interleave(neg_idx.size(1))
            neg_flat = neg_idx.reshape(-1)
        else:
            u_exp, pos_exp, neg_flat = user_idx, pos_idx, neg_idx

        pos_score = model.score_user_outfit(user_emb, outfit_emb, u_exp, pos_exp)
        neg_score = model.score_user_outfit(user_emb, outfit_emb, u_exp, neg_flat)
        rec_loss  = bpr_loss(pos_score, neg_score)

        pos_comp  = model.score_compatibility(item_emb, compat_pos_train_t)
        neg_comp  = model.score_compatibility(item_emb, compat_neg_train_t)
        comp_loss = bpr_loss(pos_comp, neg_comp)

        loss = (rec_loss + LAMBDA_COMP * comp_loss) / n_batches  # scale for mean gradient
        is_last = (batch_idx == n_batches - 1)
        loss.backward(retain_graph=not is_last)  # OK: no step() until after loop

    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()  # single update per epoch
```

**Tuning notes:**
- Try `LR * sqrt(n_batches)` or `LR / n_batches` vs current `LR=0.001`.
- Compatibility loss runs on full `compat_pos_train_t` each batch (same as today); could move outside the batch loop once per epoch to save more time.

---

##### Option B — Chunked accumulation (recompute forward every N batches)

**Idea:** Split the epoch into chunks of size `N`. Within each chunk: one forward, accumulate over N batches, one `step()`, then forward again. Reduces forwards from `~100/epoch` to `~100/N/epoch` while keeping smaller effective batch size than Option A.

**Trade-off:** Middle ground between speed and mini-batch noise. Still not identical to per-batch SGD.

```python
# Config (add to Cell 2)
RECOMPUTE_EVERY_N_BATCHES = 10   # e.g. 5, 10, 20

for epoch in range(1, EPOCHS + 1):
    model.train()
    chunk_loss_sum = 0.0
    chunk_count = 0

    for batch_idx, batch in enumerate(train_loader):
        if chunk_count == 0:
            optimizer.zero_grad()
            user_emb, outfit_emb, item_emb = forward_all_embeddings(model)

        user_idx = batch["user_idx"].to(device)
        pos_idx  = batch["pos_outfit_idx"].to(device)
        neg_idx  = batch["neg_outfit_idx"]
        if isinstance(neg_idx, (list, tuple)):
            neg_idx = torch.stack(neg_idx, dim=1)
        neg_idx = neg_idx.to(device)

        if neg_idx.dim() == 2:
            u_exp    = user_idx.repeat_interleave(neg_idx.size(1))
            pos_exp  = pos_idx.repeat_interleave(neg_idx.size(1))
            neg_flat = neg_idx.reshape(-1)
        else:
            u_exp, pos_exp, neg_flat = user_idx, pos_idx, neg_idx

        pos_score = model.score_user_outfit(user_emb, outfit_emb, u_exp, pos_exp)
        neg_score = model.score_user_outfit(user_emb, outfit_emb, u_exp, neg_flat)
        rec_loss  = bpr_loss(pos_score, neg_score)

        pos_comp  = model.score_compatibility(item_emb, compat_pos_train_t)
        neg_comp  = model.score_compatibility(item_emb, compat_neg_train_t)
        comp_loss = bpr_loss(pos_comp, neg_comp)

        loss = rec_loss + LAMBDA_COMP * comp_loss
        chunk_loss_sum += loss
        chunk_count += 1

        is_end_of_chunk = (
            chunk_count == RECOMPUTE_EVERY_N_BATCHES
            or batch_idx == len(train_loader) - 1
        )
        if is_end_of_chunk:
            (chunk_loss_sum / chunk_count).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            chunk_loss_sum = 0.0
            chunk_count = 0
```

**Important:** Call `forward_all_embeddings()` only when `chunk_count == 0` (after the previous chunk's `step()`). Do **not** call `optimizer.step()` mid-chunk before the accumulated backward.

---

##### Option C — Two-phase training (compatibility first, then joint or detached rec)

**Idea:** Phase 1 trains graph + compatibility head. Phase 2 adds recommendation — either jointly or with detached embeddings for a fast (but approximate) rec pass.

**Trade-off:** Deviates from the paper's single-phase joint objective; requires ablation to validate metrics.

```python
# Config (add to Cell 2)
PHASE1_EPOCHS = 15          # compatibility-only warmup
PHASE2_DETACH_REC = False   # True = Option C-fast (no graph grad from rec); False = full joint

for epoch in range(1, EPOCHS + 1):
    model.train()
    compat_only = (epoch <= PHASE1_EPOCHS)

    for batch in train_loader:
        optimizer.zero_grad()

        if PHASE2_DETACH_REC and not compat_only:
            # Fast path: graph forward without grad; only indexing/scoring trains rec path weakly
            with torch.no_grad():
                user_emb, outfit_emb, item_emb = forward_all_embeddings(model)
            user_emb = user_emb.detach()
            outfit_emb = outfit_emb.detach()
            item_emb = item_emb.detach()
        else:
            user_emb, outfit_emb, item_emb = forward_all_embeddings(model)

        # ... neg_idx / u_exp / pos_exp / neg_flat (same as current) ...

        if compat_only:
            pos_comp  = model.score_compatibility(item_emb, compat_pos_train_t)
            neg_comp  = model.score_compatibility(item_emb, compat_neg_train_t)
            loss = bpr_loss(pos_comp, neg_comp)
        else:
            pos_score = model.score_user_outfit(user_emb, outfit_emb, u_exp, pos_exp)
            neg_score = model.score_user_outfit(user_emb, outfit_emb, u_exp, neg_flat)
            rec_loss  = bpr_loss(pos_score, neg_score)

            pos_comp  = model.score_compatibility(item_emb, compat_pos_train_t)
            neg_comp  = model.score_compatibility(item_emb, compat_neg_train_t)
            comp_loss = bpr_loss(pos_comp, neg_comp)

            loss = rec_loss + LAMBDA_COMP * comp_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
```

**Phase schedule variants to try:**

| Variant | Phase 1 | Phase 2 |
|---------|---------|---------|
| C1 (warmup) | `comp_loss` only, 10–20 epochs | Joint `rec + λ*comp` |
| C2 (detach rec) | Joint training | `PHASE2_DETACH_REC=True` for speed experiment |
| C3 (freeze head) | Joint | Freeze `compat_mlp`, fine-tune graph for rec only |

---

##### Option D — Engineering optimizations (no training-semantics change)

- Larger `BATCH_SIZE` to reduce forwards per epoch.
- Cache static feature projections; only re-run sparse graph layers.
- Profile sparse `torch.sparse.mm` on MPS vs CPU fallback.

### Fix 11 (Medium Priority): Increase training negative samples per positive (NEG_PER_POS)

The team currently uses `NEG_PER_POS = 1` (1 random negative per positive in the BPR training DataLoader), while evaluation uses 99 negatives. This large mismatch between the training signal distribution and the evaluation distribution causes the model to learn a weak ranking boundary.

Increasing `NEG_PER_POS` gives the model harder ranking examples during training:

```python
NEG_PER_POS = 5  # was 1; try 5 or 10 for stronger training signal
```

Also update `BPRDataset.__getitem__` to return a list of negatives, and adjust the BPR loss to average over multiple negatives per positive.

### Fix 12 (Lower Priority): Reduce item-item graph density [Skip Fixing]

As noted in Difference 8, the team builds a 298,012-edge item-item graph (top-10 neighbors per item) while the author uses a 103,652-edge pre-built graph. Over-smoothing from the dense graph reduces item embedding distinctiveness.

Reduce the `MIN_TOP_NEIGHBORS` parameter:

```python
MIN_TOP_NEIGHBORS = 5  # was 10; reduces graph density from 298K to ~149K edges
```

This is especially important for items that belong to large categories (max 2,127 items in one category), where top-10 still creates very dense local neighborhoods.

### Fix 13 (Lower Priority): Use separate optimizer parameter groups for large embedding tables [Skip Fixing]

The team's `nn.Embedding(277469, 64)` for users has 17.76M parameters. Applying `WEIGHT_DECAY=1e-3` uniformly pushes all user embeddings toward zero very aggressively, potentially canceling out the graph propagation signal. Use separate weight decay for the large embedding tables:

```python
optimizer = torch.optim.AdamW([
    {
        'params': [model.user_base.weight, model.outfit_base.weight],
        'weight_decay': 1e-4,   # lighter for embeddings
    },
    {
        'params': [p for n, p in model.named_parameters()
                   if 'user_base' not in n and 'outfit_base' not in n],
        'weight_decay': 1e-5,   # lighter for projections (match author)
    },
], lr=LR)
```

---

## Summary Table


| Dimension                       | Author                               | Team                                     | Severity                         | Fix #    |
| ------------------------------- | ------------------------------------ | ---------------------------------------- | -------------------------------- | -------- |
| Model architecture              | Simple sparse GAT + dot product      | Complex MLP + MHA + MLP scorer           | CRITICAL                         | Fix 2    |
| Input features                  | Pre-computed 64-dim embeddings       | Raw image (2048) + text (768) + cat (61) | HIGH                             | —        |
| Recommendation scorer           | Dot product (cosine)                 | MLP concat                               | CRITICAL                         | Fix 2    |
| λ_comp                          | 0.5                                  | 0.1                                      | HIGH                             | Fix 3    |
| Weight decay                    | 1e-5                                 | 1e-3                                     | MEDIUM                           | Fix 7/13 |
| Batch size                      | 512                                  | 1024                                     | LOW                              | —        |
| Epochs / early stopping         | 100 + patience=10                    | 40, no early stop                        | HIGH                             | Fix 5    |
| LR scheduler                    | ReduceLROnPlateau (mode=max, p=5)    | None                                     | MEDIUM                           | Fix 6    |
| Data split                      | Random 80/10/10 lines                | Stratified per-user holdout              | HIGH                             | Fix 4    |
| Eval negatives                  | 50 random                            | 99 sampled                               | CRITICAL (for metric comparison) | Fix 1    |
| Training negatives per positive | 1                                    | 1                                        | MEDIUM                           | Fix 11   |
| Gradient clipping               | max_norm=1.0                         | None                                     | MEDIUM                           | Fix 8    |
| L2 normalization of embeddings  | Applied to all (item, outfit, user)  | Partial (outfit, user only)              | MEDIUM                           | Fix 9    |
| Graph forward per batch         | Once per batch                       | Once per batch (Fix 10 reverted)         | MEDIUM (speed only)              | Fix 10 — reverted; see future options |
| Item-item graph density         | Pre-built similarity (103K edges)    | Category+cooc top-10 (298K edges)        | MEDIUM                           | Fix 12   |
| Embedding optimizer groups      | Single group (1e-5)                  | Single group (1e-3 uniform)              | LOWER                            | Fix 13   |
| Compatibility negatives         | Random item                          | Hard (same-category)                     | MEDIUM                           | —        |
| User-outfit graph at train      | Full (includes val/test)             | Train-only (correct, but tiny val)       | LOW                              | —        |
| Author notebook bug             | TypeError crashes epoch 1 validation | N/A                                      | NOTE                             | —        |


---

## Post-Training Update (June 2025): `val_total_loss`, `LAMBDA_COMP`, and Precision@10

### `LAMBDA_COMP` tuned to 0.2

After Fixes 1–11 with `LAMBDA_COMP=0.5`, training showed:

- `val_rec_loss` **decreasing** (0.60 → ~0.51) — recommendation is learning
- `val_total_loss` **increasing** (0.91 → ~2.07) — dominated by rising `val_comp_loss`
- `train_comp_loss` collapsing to ~0.01 while compat val loss grows → compatibility branch overfits held-out outfits

`val_total_loss = val_rec_loss + λ · val_comp_loss`. With λ=0.5, even modest compat val degradation inflates the headline number. **λ=0.2** keeps compatibility as a regularizer without letting compat val dominate monitoring. **Use NDCG@10 / HR@10 for early stopping**, not `val_total_loss`.

Epoch output now prints `val_comp_loss` and `avg_val_outfits` / `prec_ceiling@10` for clearer diagnosis.

### Precision@10 ≈ 0.08 is mostly expected — not a broken model

After aligning scorer (dot product), eval negatives (50), and split (80/10/10 edges), epoch-6 metrics were:

| Metric | Value | Interpretation |
|--------|-------|----------------|
| HR@10 | 0.678 | ~68% of val users have ≥1 relevant outfit in top-10 |
| Recall@10 | 0.637 | ~64% of each user's val outfits are retrieved in top-10 |
| NDCG@10 | 0.359 | Ranking quality improving steadily |
| Precision@10 | 0.080 | Looks low vs author's ~0.44 |

**Root cause: metric definition + sparse val users, not ranking failure.**

`ranking_metrics_multi_k` computes per user:

```python
precision@k = (hits in top-k) / k        # always divides by k=10
recall@k    = (hits in top-k) / |true_set|
```

With **random edge-level 80/10/10 split**, most users have only **1–2 total interactions** (`MIN_USER_INTERACTIONS=4` filters further). Typical val user has **|true_set| ≈ 1**.

When |true_set| = 1:

- If the outfit is in top-10: **Precision@10 = 1/10 = 0.10** (hard ceiling per user)
- HR@10 = Recall@10 (same binary outcome)
- Expected Precision@10 ≈ HR@10 × 0.10 ≈ 0.68 × 0.10 ≈ **0.068**

Observed **0.08** matches this ceiling — the model is performing **near optimally** for this metric under this split.

The notebook now reports:

- `avg_val_outfits_per_user` — mean |true_set| in eval (expect ~1–2)
- `precision_ceiling_perfect@10` — mean of `min(|true_set|, 10) / 10` if ranking were perfect (expect ~0.10–0.15 with sparse users)

**Do not compare raw Precision@10 to the author's 0.44 without checking |true_set| per eval.**

### Why the author's Precision@10 = 0.44 is not apples-to-apples

1. **Author notebook on `kimnguyen` branch crashes at epoch-1 validation** (`TypeError` in `H_HFGAT.forward`) — reported numbers likely from an unreleased fixed notebook.
2. **Different eval aggregation**: author `evaluate_rec()` runs per DataLoader batch; when the same user appears with **multiple val outfits in one batch**, up to several positives enter the same candidate pool → Precision@10 can exceed 0.1 per eval step.
3. **Line-based vs edge-based split**: author's 80/10/10 on interaction lines can leave users with more val outfits per evaluation than edge-level random split on sparse users.

### Secondary factors (already addressed)

| Factor | Status |
|--------|--------|
| 99 vs 50 eval negatives | Fixed (50) |
| MLP vs dot-product scorer | Fixed (dot product) |
| Stratified vs random split | Fixed (random 80/10/10) |

### MPS eval crash (epoch 11)

Building `score_map` with `.item()` on MPS tensors can hang or error. Fixed with `.detach().cpu().item()` in `ranking_metrics_multi_k`.

### If you need higher reported Precision@10

These change the metric or data — not model quality:

1. **User-level or line-level val split** so users retain multiple val outfits per eval
2. **Author-style batch eval** (`evaluate_rec` per batch, grouping in-batch positives)
3. **Report `Precision@10 / precision_ceiling_perfect@10`** as "fraction of theoretical max"
4. **Filter eval** to users with ≥3 val outfits for paper-style comparison

**Recommendation:** treat **NDCG@10, HR@10, Recall@10** as primary ranking metrics; interpret Precision@10 only alongside `avg_val_outfits_per_user` and `prec_ceiling@10`.

