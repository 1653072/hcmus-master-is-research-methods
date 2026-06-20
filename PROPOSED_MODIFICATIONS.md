# 4. Proposed Lightweight Modifications

## 4.1 Introduction and Design Rationale

This section documents how the **Lightweight H-HFGAT** implementation adapts the author’s Hierarchical Hybrid Fashion Graph Attention Network for runs on limited memory (personal machine, cloud runtime, or competition platform). Sections 4.3–4.8 describe changes already in the codebase. Section 4.9 lists improvements not yet implemented. The focus is on what was changed, why it was changed, and how each part behaves—not on experimental scores.

## 4.2 Baseline: Author H-HFGAT versus Lightweight Implementation

Both implementations share the same core model: `MultiHeadSelfAttentionLayer`, `UserAttentionAggregator`, `CompatibilityScorer`, and `H_HFGAT`. They differ mainly in data scope, training protocol, graph usage at train time, and evaluation rules. The table below compares them row by row. The *Rationale* column explains why the Lightweight side chose its approach.


| Aspect                                | Description                                                        | Author implementation                                                                | Lightweight implementation                                                                       | Rationale (Lightweight)                                                                                                          |
| ------------------------------------- | ------------------------------------------------------------------ | ------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------- |
| User scope                            | Which users and interactions enter the graph                       | Full interaction corpus                                                              | Users with at least four interactions                                                            | Keeps users with enough history for meaningful splits and stable graph signal; reduces graph size for faster Lightweight runs    |
| Visual and textual features           | How item images and titles become vectors                          | ResNet-152 and BERT Chinese; text truncated to 64 tokens; per-item extraction        | Same ResNet-152 and BERT Chinese pipeline; batched extraction; optional persistent feature cache | Same feature models as the author; cache and batching avoid repeating costly extraction on every run                             |
| Item embedding fusion                 | How visual and textual vectors combine into item embeddings        | Untrained linear projection from concatenated features; gradients detached at fusion | Untrained linear projection from concatenated features; gradients detached at fusion             | Unchanged from author at this stage; fusion weights are not trained before the graph model.                                      |
| Recommendation data split             | How user–outfit interactions are divided                           | Ninety–ten line split on the interaction file (train versus validation only)         | Per-user eighty–ten–ten split into train, validation, and test                                   | Each user keeps a local holdout for validation and test, which better reflects per-user generalization than one global shuffle   |
| User–outfit graph during training     | Which edges are used in message passing during optimization        | Full user–outfit graph, including validation interactions                            | Only training interactions                                                                       | Stops validation and test preferences from influencing user and outfit embeddings during forward passes                          |
| Compatibility training signal         | Item vectors fed to the compatibility scorer during training       | Graph Attention Network–updated item vectors                                         | Base item embeddings                                                                             | Separates compatibility learning from the recommendation path so the two losses do not pull item geometry in opposite directions |
| Compatibility evaluation signal       | Item vectors used when measuring compatibility on validation data  | Base item embeddings                                                                 | Base item embeddings                                                                             | Matches training inputs when detachment is enabled, so compatibility metrics are measured consistently                           |
| Recommendation scoring                | How user and outfit vectors form a preference score                | Raw inner product                                                                    | L2-normalized vectors, then inner product (cosine-style)                                         | Normalizes scale across users and outfits so Bayesian Personalized Ranking (BPR) scores are comparable                           |
| Embedding optimization                | Whether base item, outfit, and user vectors update during training | Fixed tensors; only the graph model and scorer train                                 | Trainable parameters are used                                                                    | Allow randomly initialized user and outfit vectors to adapt jointly with the graph during training                               |
| Graph Attention Network normalization | How attention is normalized over edges                             | Softmax over all edges together (global)                                             | Per-destination softmax (each node normalizes its incoming edges)                                | Standard graph attention behavior: each node’s neighbors form a proper weight distribution                                       |
| Recommendation evaluation negatives   | How negative outfits are chosen at evaluation                      | Random sample; outfits from the user’s training history may appear as negatives      | Random sample with training outfits removed per user                                             | Avoids ranking metrics inflated by negatives the user already interacted with in training                                        |


**Impact of the main protocol changes:** 

- In the author setup, validation user–outfit edges still take part in graph propagation, so the model can indirectly “see” validation preferences while training. The Lightweight implementation removes those edges from message passing and uses only training interactions when updating user and outfit representations. 
- For compatibility, the author backpropagates through Graph Attention Network–refined items, which ties compatibility tightly to the recommendation pathway. Meanwhile, the Lightweight implementation scores compatibility on base item embeddings and can stop gradients at that input so the compatibility scorer learns without fighting recommendation updates. 
- Finally, at recommendation evaluation time, the Lightweight implementation does not treat a user’s training outfits as negative candidates, which makes the ranking task harder but more faithful to real recommendation.

## 4.3 Efficient Data Pipeline

The Lightweight data path follows the author’s feature philosophy (same visual and text backbones) and adds **filtering**, **batching**, and **caching** so development and re-training do not repeat heavy work.

**Filtering:** Raw user–outfit interactions are deduplicated. Users below the minimum interaction count are dropped; outfits, items, and edges are filtered in cascade so every remaining edge refers to valid entities. Items without an on-disk image can be excluded so visual features are well defined.

**Feature extraction:** Visual vectors come from ResNet-152; textual vectors from BERT Chinese — the same family of models as the author. Images and titles are processed in batches (`IMAGE_BATCH_SIZE`, `TEXT_BATCH_SIZE`) instead of one item at a time. When `MAX_TEXT_LENGTH` is unset, the tokenizer uses the model’s maximum length (512 for BERT base Chinese) while the author caps text at 64 tokens.

**Caching:** After the first successful extraction, visual and textual features are stored in a cache. Later runs load the cache when `FORCE_REBUILD_FEATURES` is false, skipping ResNet and BERT passes. This is an engineering addition, so it does not change the definition of the features themselves.


| Step | Operation                             | Expected input          | Expected output                                                                    |
| ---- | ------------------------------------- | ----------------------- | ---------------------------------------------------------------------------------- |
| 1    | Interaction filtering                 | Raw user–outfit records | Subgraph of users with at least four interactions and consistent items and outfits |
| 2    | Batched visual and textual extraction | Item images and titles  | Fixed-size visual and textual vectors per item                                     |
| 3    | Feature cache read or write           | Extracted vectors       | Reused features on subsequent runs, or fresh cache after rebuild                   |
| 4    | Subsample export                      | Filtered tables         | Serialized subsample used for graph construction and training                      |


## 4.4 Learnable Embeddings and Hybrid Edge Weighting

This subsection will cover two linked ideas: **Who learns during training** and **How edge weights are assigned** in the three-level graph (item, outfit, user).

**Learnable embeddings:** The author loads item, outfit, and user vectors as fixed tensors, and the optimizer updates only the Graph Attention Network layers and the compatibility scorer. In the Lightweight implementation, when the `LEARNABLE_EMBEDDINGS` config is true, those base vectors are registered as trainable parameters and included in the optimizer and gradient clipping. Recommendation uses Bayesian Personalized Ranking (BPR) on updated user and outfit representations, while Compatibility uses Fill In The Blank (FITB) batches. Because user and outfit vectors start from random initialization, allowing them to move during training often improves fit more than freezing them.

**Hybrid edge weighting:** The graph uses three connection types, and each type is weighted differently on purpose - that is what “hybrid” means.

1. **Item–item edges (weighted by category).** When two items appear in the same outfit, the link between them is strong or weak based on how often their *categories* (shirt, pants, shoes, and so on) co-occur across outfits. Outfit statistics produce a *category co-occurrence factor*, min–max normalized (Equation 3.3). That factor becomes the item–item edge weight. If categories are unlinked, a small default weight is used. In this case, items from categories that often appear together get a stronger connection than unrelated pairs.
2. **Outfit–item edges (weight = 1):** These links only record *membership*: "this item belongs to this outfit". There is no category formula, meaning that every outfit–item edge has unit weight.
3. **User–outfit edges (weight = 1, and training only at train time):** These links record *interaction*: "this user engaged with this outfit". A full user–outfit adjacency may be built from all data, but during training and validation forward passes, message passing uses *only training user–outfit edges*, so validation and test interactions do not update user or outfit embeddings.

**How weights affect the item Graph Attention Network.** For item–item edges, the model computes attention between neighbors, multiplies it by the edge weight, then applies softmax. Category-related pairs with higher weights therefore influence an item’s updated embedding more. Outfit–item and user–outfit edges stay at weight 1 and serve a structural role in aggregation.


| Edge type                   | Weight source                 | Role in the model                                                                          |
| --------------------------- | ----------------------------- | ------------------------------------------------------------------------------------------ |
| Item–item                   | Category co-occurrence factor | Encodes fashion relatedness between items that coappear in outfits                         |
| Outfit–item                 | Unity (1.0)                   | Aggregates item vectors into an outfit representation                                      |
| User–outfit (training only) | Unity (1.0)                   | Aggregates outfit vectors into a user preference representation without validation leakage |


## 4.5 Graph Construction and Attention-based Propagation

**Graph construction:** Category statistics are computed from outfit records. The weighted item–item adjacency, binary outfit–item adjacency, and user–outfit adjacency are stored as sparse matrices and converted to edge index and edge weight tensors for training. The training module rebuilds a **train-only** user–outfit edge set from the recommendation train split.

**Propagation (three stages):** 

- (1) **Item layer:** Multi-head graph attention over item–item edges - attention is normalized per destination node, and dropout and L2 normalization are applied on item outputs. 
- (2) **Outfit layer:** Updated item vectors are summed (scatter aggregation) into each outfit, then passed through a linear layer. 
- (3) **User layer:** The `UserAttentionAggregator` method combines outfit vectors for each user over training user–outfit edges only, with per-user attention softmax.


| Stage                        | Expected input                                                        | Expected output                            |
| ---------------------------- | --------------------------------------------------------------------- | ------------------------------------------ |
| Item Graph Attention Network | Item embeddings with weighted item–item edges                         | Context-aware item embeddings              |
| Outfit aggregation           | Item embeddings with outfit–item edges                                | Outfit embeddings                          |
| User aggregation             | User embeddings with outfit embeddings and training user–outfit edges | User embeddings for recommendation scoring |


## 4.6 Training and Evaluation Modifications

The table lists training and evaluation behaviors beyond the author baseline. The *Configuration* column names the setting in shared configuration where applicable.


| Modification                                 | Purpose                                                                                                 | Configuration                                                    |
| -------------------------------------------- | ------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------- |
| Per-user train, validation, and test split   | Each user’s interactions are split locally instead of shuffling all lines globally                      | `SPLIT_MODE = "per_user"`                                        |
| Train-interaction-only graph in forward pass | Validation and test edges do not propagate into embeddings                                              | Train-only `user_outfit_train_index` built from train split      |
| Decoupled compatibility on base embeddings   | Compatibility loss does not backprop into base item embeddings or the item Graph Attention Network path | `COMPAT_DETACH_INPUT = True`                                     |
| Fill In The Blank with hard item swap        | One correct full outfit versus three outfits each with one item replaced                                | Fill In The Blank files with three negative outfits per positive |
| Padding mask in compatibility scorer         | Padded item slots in a batch do not affect compatibility scores                                         | `MAX_OUTFIT_ITEMS_FOR_COMP = 10`; mask on pad index              |
| Cosine-style recommendation scoring          | User and outfit vectors are L2-normalized before the preference dot product                             | Implemented in `score_recommendation` method.                    |
| Evaluation negative filtering                | Negative outfits at evaluation exclude those in the user’s training history                             | `EVAL_NEG_SAMPLES = 50`; `user_known_outfits_idx`                |
| Multi-negative Bayesian Personalized Ranking | Several negative outfits per positive during training                                                   | `NEG_PER_POS = 3`                                                |
| Joint recommendation and compatibility loss  | Total loss combines ranking loss and weighted compatibility loss                                        | `LAMBDA_COMP = 0.3`                                              |
| Checkpoint on ranking Hit Rate               | Best model is saved by validation Hit Rate at ten (HR@10), not by combined validation loss              | `EARLY_STOP_METRIC = "HR@K"`, `TOP_K = 10`, `PATIENCE = 10`      |
| Learnable base embeddings                    | Item, outfit, and user base vectors can update during training                                          | `LEARNABLE_EMBEDDINGS = True`                                    |


**How these pieces work together:** 

- The per-user split ensures that validation and test interactions for a given user never appear in that user’s training set. 
- The train-only graph guarantees that even if validation edges exist in the data structures, they are not used when computing user and outfit embeddings in the forward pass. 
- Decoupled compatibility means the compatibility scorer learns outfit plausibility from base item vectors without forcing those vectors to satisfy both ranking and compatibility at once. 
- Evaluation negative filtering ensures the model is ranked against outfits the user has not already seen in training, which is a stricter and more realistic recommendation test.

## 4.7 End-to-End Processing Overview


| Step | Component                    | Expected input                 | Expected output                                                          |
| ---- | ---------------------------- | ------------------------------ | ------------------------------------------------------------------------ |
| 1    | Ingestion and filtering      | Raw fashion dataset            | Lightweight subgraph (filtered users, items, outfits, edges)             |
| 2    | Feature extraction and cache | Images and titles              | Per-item visual and textual vectors (cached for reuse)                   |
| 3    | Embedding initialization     | Features and random seeds      | Item, outfit, and user embedding matrices                                |
| 4    | Graph construction           | Subsample and category factors | Sparse adjacency structures (item–item weighted and others binary)       |
| 5    | Splitting                    | Interactions and outfits       | Recommendation train, validation, test, and Fill In The Blank partitions |
| 6    | Training                     | Graphs, embeddings, splits     | Trained graph model, compatibility scorer, updated embeddings            |
| 7    | Evaluation                   | Held-out partitions            | Ranking and compatibility outputs.                                       |


## 4.8 Summary of Implemented Modifications


| Modification                                     | Category      | Primary intent                                         |
| ------------------------------------------------ | ------------- | ------------------------------------------------------ |
| Minimum-interaction user filtering               | Data pipeline | Denser subgraph and feasible Lightweight runs          |
| Feature cache on same backbones as author        | Data pipeline | Avoid repeated ResNet and BERT extraction              |
| Learnable base embeddings                        | Embeddings    | Adapt weak random initialization during training       |
| Category-weighted item–item graph                | Graph         | Encode outfit-level category structure in edge weights |
| Train-only user–outfit propagation               | Graph         | Prevent validation and test leakage in message passing |
| Decoupled compatibility training                 | Training      | Stable multi-task optimization                         |
| Per-user split and evaluation negative filtering | Evaluation    | Fairer, stricter recommendation protocol               |
| Per-destination graph attention                  | Graph         | Correct local attention normalization                  |
| Cosine-style recommendation scoring              | Training      | Scale-stable preference scores                         |


## 4.9 Other Proposals

The following extensions are planned but not yet implemented in the Lightweight codebase.

**Top-K neighbor pruning**: After the full item–item graph is built, each item would retain only its highest-weight neighbors up to a fixed cap. This would reduce memory usage and propagation cost. At present, there is a configuration parameter (i.e., MIN_TOP_NEIGHBORS), which is already defined, but it is not applied to the graph construction for the neighbor pruning yet.

**Sparse matrix propagation**: The item-level layer could be refactored to perform message passing directly on a compressed sparse adjacency structure rather than on a dense edge list with full attention. The intended benefit is lower memory consumption during training.

**Lighter feature extraction**: The feature pipeline could optionally use a smaller visual backbone and a shorter maximum text length when the feature cache is rebuilt. This would trade some representational capacity for faster extraction and lower storage.

**Trained item fusion**: The linear projection that maps concatenated visual and textual vectors into item embeddings could be trained with supervision, replacing the current untrained projection at initialization.

**Post-pruning weight re-normalization**: If top-K pruning is adopted, edge weights retained per node could be re-scaled so that attention magnitudes remain stable after the graph is sparsified.

