# LNCS Writing Examples

Short patterns for common sections. Adapt facts to the actual paper; do not copy verbatim.

---

## Abstract (~120 words)

> Outfit recommendation systems must model both user preferences and item compatibility within an outfit. Existing graph attention approaches often train on graphs that leak future interactions and under-specify compatibility evaluation. We present a lightweight graph attention pipeline that builds user–outfit graphs from training interactions only, separates compatibility scoring from recommendation ranking, and evaluates both tasks with clearly defined metrics. On the Polyvore dataset, our method achieves a validation Hit Rate@10 of 0.77 and NDCG@10 of 0.46, while maintaining a Fill In The Blank compatibility accuracy of 0.72. The design trades a small amount of compatibility accuracy for more stable ranking performance under an active-user training setup. Our code and configuration will be made available upon acceptance.

**Keywords:** Outfit recommendation · Graph attention · Compatibility modeling · Polyvore

---

## Introduction: opening paragraph

> Fashion outfit recommendation suggests coherent item sets that match a user's style. Recent methods represent users, items, and outfits as nodes in a graph and apply attention to propagate relational signals. However, graph construction and evaluation protocols strongly influence reported performance, and compatibility is often assessed with metrics that are not comparable across papers.

## Introduction: contributions (numbered)

> This paper makes the following contributions:
> 1. We describe a lightweight graph attention pipeline that restricts graph edges to training interactions to avoid leakage at evaluation time.
> 2. We decouple compatibility scoring from recommendation ranking and document the evaluation protocol for each task.
> 3. We report ranking and compatibility results on Polyvore and analyze protocol differences against a published FGAT baseline.

## Introduction: outline (one sentence)

> The remainder of the paper is organized as follows: Sect. 2 reviews related work, Sect. 3 presents our method, Sect. 4 describes experiments, and Sect. 5 concludes.

---

## Related Work: thematic paragraph

> **Graph-based outfit recommendation.** Early methods model outfits as unordered item sets and learn compatibility with metric learning or autoregressive decoders. Graph attention networks extend this line by propagating signals over item–item and user–outfit edges. FGAT jointly optimizes recommendation and Fill In The Blank compatibility with a shared encoder. Unlike FGAT, we build the user–outfit graph from training interactions only and report compatibility accuracy separately from ranking metrics.

---

## Method: subsection openers

> **Problem formulation.** Let $\mathcal{U}$, $\mathcal{I}$, and $\mathcal{O}$ denote sets of users, items, and outfits. Each outfit $o \in \mathcal{O}$ is a set of items. The recommendation task ranks outfits for user $u$; the compatibility task scores whether an outfit is internally consistent.

> **Compatibility scoring.** Given item embeddings $\mathbf{e}_i$, we score outfit $o$ with a permutation-invariant aggregator over its items, trained with Fill In The Blank negatives.

---

## Experiments: setup + caveat

> **Datasets and splits.** We use the Polyvore split with train, validation, and test partitions for recommendation and a separate Fill In The Blank set for compatibility. User–outfit edges in the propagation graph are taken from training data only.

> **Metrics.** We report Hit Rate@10 and NDCG@10 for recommendation, and accuracy for Fill In The Blank compatibility. Precision@10 is sensitive to the number of ground-truth outfits per user in our validation split, so we emphasize Hit Rate@10 and NDCG@10 in the comparison.

## Experiments: results (interpret, don't repeat)

> Table 1 shows that our method improves Hit Rate@10 and NDCG@10 over the published FGAT test results, while compatibility accuracy is lower. The gap in compatibility is partly explained by different negative sampling and by our choice to stabilize ranking through a detached compatibility path. We therefore treat the compatibility numbers as indicative rather than as a strict head-to-head comparison.

---

## Table caption (above table)

> **Table 1.** Recommendation and compatibility results. Author FGAT values are from the original paper (test set). Our results are on validation at the best ranking checkpoint. Compatibility accuracy follows the Fill In The Blank task.

| Metric | FGAT (paper, test) | Ours (val., best) |
|--------|-------------------:|------------------:|
| HR@10 | 0.4286 | 0.7737 |
| NDCG@10 | 0.1340 | 0.4645 |
| Compatibility accuracy | 0.8956 | 0.7195 |

---

## Figure caption (below figure)

> **Fig. 1.** Overview of the lightweight graph attention pipeline. Item and user embeddings are updated through three graph layers; compatibility and recommendation use separate scoring heads.

---

## Conclusion

> We presented a lightweight graph attention pipeline for outfit recommendation with an explicit train-only graph and separate compatibility evaluation. Experiments on Polyvore show strong ranking metrics on validation, with compatibility accuracy that leaves room for improvement. Future work includes training the visual–text fusion layer and pruning item–item neighbors for faster propagation.

---

## Phrases to avoid → prefer

| Avoid | Prefer |
|-------|--------|
| In this section, we will discuss… | Section 4 evaluates… |
| It is worth noting that… | (state the fact directly) |
| State-of-the-art performance | Hit Rate@10 of X on dataset Y |
| The reader should note… | We note that… / (omit) |
| Obviously / Clearly | (omit or justify) |
| Novel paradigm | We propose / We extend |
| `compat_acc` in prose | compatibility accuracy |
| Em dash (`—`) | Comma, period, or parentheses |
| 71.95% and 0.7195 mixed | Pick one format per table |

---

## Limitations (brief, in Conclusion or Experiments)

> **Limitations.** Our comparison to FGAT uses different splits and checkpoint selection criteria. We report validation results at the best ranking epoch, whereas the baseline reports test results. A strictly aligned benchmark would require harmonized graph construction, negative sampling, and checkpoint rules.
