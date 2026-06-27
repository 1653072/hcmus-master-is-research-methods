# Project context: HFGAT / Lightweight FGAT

Read this file when drafting LNCS content **in this repository** (`hcmus-master-is-research-methods`).

**Active branch (writing baseline):** `kimnguyen-quoctran-v1.3`

---

## Method naming

| Use in prose | Do not use |
|--------------|------------|
| **Lightweight** (pipeline / method) | Active-User |
| **FGAT** (author baseline / paper) | Informal nicknames for our method |
| **HFGAT** | Only if user explicitly wants this acronym in title/branding |

---

## Writing rules (project-specific)

- Vietnamese first for drafts; **English** for LNCS submission (abstract, body, captions)
- Use **chúng tôi** / **we**, humble but confident; no defensive hedging
- **Never use em dash (`—`)** in paper prose (English or Vietnamese)
- No notebook mentions, file extensions, or code variable names in report prose
- New paper prose: always run **humanizer** (write-lncs-paper Step 8) before save or final delivery
- No semicolons in formal proposal-style lists (match `PROPOSED_MODIFICATIONS.md` tone)
- Distinguish **implemented** (Sections 4.3–4.8) vs **proposed** (Section 4.9 Other Proposals)
- Do not claim unfair metric comparisons

---

## Evaluation comparison table (validation, best ranking checkpoint)

Use for Section 5.4 / Experiments narrative. **Do not include AUC row** when comparing to FGAT paper (paper AUC is compatibility; ours was recommendation AUC).

| Metric | FGAT (paper, test) | Lightweight (validation, best) |
|--------|-------------------:|-----------------------------:|
| HR@10 | 0.4286 | 0.7737 |
| NDCG@10 | 0.1340 | 0.4645 |
| Recall@10 | 0.1580 | 0.7722 |
| Precision@10 | 0.4424 | 0.0779 |
| Compatibility accuracy (Fill In The Blank) | 0.8956 | 0.7195 |

**Notes:**
- `0.7195` = `compat_acc` at best validation epoch (HR@10 peak, epoch 40). Test `compat_acc` = **0.7044**, use only if column is **test**, not validation.
- Paper reports **test**; we report **validation** at best checkpoint, state this limitation once at the start of the comparison.
- Precision@10 and compatibility accuracy are hard to compare directly across protocols.
- HR@10 is the recommendation success metric (no separate "Accuracy (recommendation)" row).

---

## Config snapshot (v1.3: for method/experiments text)

Reference `fgat_config.py` on branch `kimnguyen-quoctran-v1.3`:

- `LAMBDA_COMP=0.3`, `WEIGHT_DECAY=1e-5`, `DROPOUT=0.3`
- `NEG_PER_POS=3`, `EPOCHS=50`, `EVAL_NEG_SAMPLES=50`
- `COMPAT_DETACH_INPUT=True`, `LEARNABLE_EMBEDDINGS=True`
- Early stopping on validation HR@10 (`EARLY_STOP_METRIC=HR@K`, `TOP_K=10`)
- Train-only user–outfit graph for propagation

Describe these in plain language in the paper, not as config key dumps.

---

## Repo reference docs (read-only context)

| File | Purpose |
|------|---------|
| `PROPOSED_MODIFICATIONS.md` | Section 4 structure; implemented vs proposed |
| `AUTHOR_VS_CURRENT_COMPARISON.md` | Pipeline differences vs author FGAT |
| `TESTING_PLAN_v1.3.md` | Experiment phases (do not paste into paper) |
| `output_fgat_active_user/models/test_results.json` | Canonical numbers for test / best-val metrics |

---

## Future work (max 2 items in conclusion)

1. Train the visual–text fusion layer
2. Neighbor pruning + sparse propagation on the item–item graph

Do **not** list "re-run author benchmark with identical protocol" as future work unless the user asks.

---

## Vietnamese ↔ English terms (HFGAT)

| English (LNCS) | Vietnamese (draft) |
|------------------|-------------------|
| Lightweight pipeline | Pipeline graph attention nhẹ |
| Fill In The Blank | Điền chỗ trống (Fill In The Blank) |
| Compatibility accuracy | Độ chính xác compatibility |
| Hit Rate@10 | Hit Rate@10 (giữ nguyên) |
| Train-only graph | Đồ thị chỉ từ tập huấn luyện |
| Detached compatibility path | Nhánh compatibility tách khỏi embedding cập nhật |
