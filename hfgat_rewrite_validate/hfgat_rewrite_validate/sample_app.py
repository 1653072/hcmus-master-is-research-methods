import json
from pathlib import Path
from typing import List, Dict
from PIL import Image
import torch
import pandas as pd
import streamlit as st
import torch.nn as nn
import torch.nn.functional as F


# =========================
# CONFIG
# =========================
ARTIFACT_DIR = Path("output_hfgat_notebook")
EXPORT_DIR = ARTIFACT_DIR / "exported_embeddings"


# =========================
# MODEL DEFINITION
# must match training notebook
# =========================
class MLPBlock(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class HFGATDetailed(nn.Module):
    def __init__(self, image_dim, text_dim, cat_dim, num_users, num_outfits, num_items, embed_dim=128, dropout=0.1):
        super().__init__()
        self.image_proj = nn.Linear(image_dim, embed_dim)
        self.text_proj = nn.Linear(text_dim, embed_dim)
        self.cat_proj = nn.Linear(cat_dim, embed_dim)

        self.item_fuse = MLPBlock(embed_dim * 3, embed_dim, dropout)
        self.item_update = MLPBlock(embed_dim, embed_dim, dropout)
        self.outfit_update = MLPBlock(embed_dim, embed_dim, dropout)
        self.user_update = MLPBlock(embed_dim, embed_dim, dropout)

        self.user_base = nn.Embedding(num_users, embed_dim)
        self.outfit_base = nn.Embedding(num_outfits, embed_dim)
        self.dropout = nn.Dropout(dropout)

        # PHẢI trùng tên với checkpoint đã train
        self.compat_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1),
        )

    def encode_items(self, image_x, text_x, cat_x, A_item_item):
        xi = F.normalize(self.image_proj(image_x), p=2, dim=-1)
        xt = F.normalize(self.text_proj(text_x), p=2, dim=-1)
        xc = F.normalize(self.cat_proj(cat_x), p=2, dim=-1)

        x = torch.cat([xi, xt, xc], dim=-1)
        x = self.item_fuse(x)
        x = F.normalize(x, p=2, dim=-1)

        x_prop = torch.sparse.mm(A_item_item, x)
        x = x + self.dropout(self.item_update(x_prop))
        return F.normalize(x, p=2, dim=-1)

    def encode_outfits(self, item_emb, A_outfit_item):
        agg = torch.sparse.mm(A_outfit_item, item_emb)
        base = F.normalize(self.outfit_base.weight, p=2, dim=-1)
        out = base + self.dropout(self.outfit_update(agg))
        return F.normalize(out, p=2, dim=-1)

    def encode_users(self, outfit_emb, A_user_outfit):
        agg = torch.sparse.mm(A_user_outfit, outfit_emb)
        base = F.normalize(self.user_base.weight, p=2, dim=-1)
        usr = base + self.dropout(self.user_update(agg))
        return F.normalize(usr, p=2, dim=-1)

    def forward(self, image_x, text_x, cat_x, A_item_item, A_outfit_item, A_user_outfit):
        item_emb = self.encode_items(image_x, text_x, cat_x, A_item_item)
        outfit_emb = self.encode_outfits(item_emb, A_outfit_item)
        user_emb = self.encode_users(outfit_emb, A_user_outfit)
        return user_emb, outfit_emb, item_emb

    def score_compatibility(self, item_emb, outfit_item_batch):
        mask = (outfit_item_batch >= 0).float().unsqueeze(-1)
        safe_idx = outfit_item_batch.clamp_min(0)
        x = item_emb[safe_idx] * mask
        pooled = x.sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        pooled = F.normalize(pooled, p=2, dim=-1)
        return self.compat_mlp(pooled).squeeze(-1)


# =========================
# LOAD ARTIFACTS
# =========================
@st.cache_resource
def load_artifacts():
    model_path = ARTIFACT_DIR / "model.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"Khong thay {model_path}. Ban can train xong truoc.")

    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]

    model = HFGATDetailed(
        image_dim=cfg["image_dim"],
        text_dim=cfg["text_dim"],
        cat_dim=cfg["cat_dim"],
        num_users=cfg["num_users"],
        num_outfits=cfg["num_outfits"],
        num_items=cfg["num_items"],
        embed_dim=cfg["embed_dim"],
        dropout=cfg["dropout"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    user_emb = torch.load(EXPORT_DIR / "user_embeddings.pt", map_location="cpu", weights_only=False)
    outfit_emb = torch.load(EXPORT_DIR / "outfit_embeddings.pt", map_location="cpu", weights_only=False)
    item_emb = torch.load(EXPORT_DIR / "item_embeddings.pt", map_location="cpu", weights_only=False)

    user2idx = json.loads((EXPORT_DIR / "user2idx.json").read_text(encoding="utf-8"))
    outfit2idx = json.loads((EXPORT_DIR / "outfit2idx.json").read_text(encoding="utf-8"))
    item2idx = json.loads((EXPORT_DIR / "item2idx.json").read_text(encoding="utf-8"))

    idx2user = {v: k for k, v in user2idx.items()}
    idx2outfit = {v: k for k, v in outfit2idx.items()}
    idx2item = {v: k for k, v in item2idx.items()}

    outfit_items_path = EXPORT_DIR / "outfit_items.json"
    if outfit_items_path.exists():
        outfit_items = json.loads(outfit_items_path.read_text(encoding="utf-8"))
    else:
        outfit_items = {}

    item_meta_path = EXPORT_DIR / "item_meta_ordered.csv"
    item_meta = pd.read_csv(item_meta_path) if item_meta_path.exists() else pd.DataFrame()

    history_path = EXPORT_DIR / "train_uo_sub.csv"
    if history_path.exists():
        history_df = pd.read_csv(history_path)
    else:
        history_df = pd.DataFrame(columns=["user_id", "outfit_id"])

    return {
        "model": model,
        "user_emb": F.normalize(user_emb.float(), p=2, dim=-1),
        "outfit_emb": F.normalize(outfit_emb.float(), p=2, dim=-1),
        "item_emb": F.normalize(item_emb.float(), p=2, dim=-1),
        "user2idx": user2idx,
        "outfit2idx": outfit2idx,
        "item2idx": item2idx,
        "idx2user": idx2user,
        "idx2outfit": idx2outfit,
        "idx2item": idx2item,
        "outfit_items": outfit_items,
        "item_meta": item_meta,
        "history_df": history_df,
    }


# =========================
# FUNCTIONS
# =========================
def render_outfit_items_small(outfit_id: str, bundle: Dict, max_items: int = 5, img_width: int = 70):
    item_ids = bundle["outfit_items"].get(outfit_id, [])[:max_items]

    if not item_ids:
        st.caption("No items")
        return

    cols = st.columns(len(item_ids))
    for col, item_id in zip(cols, item_ids):
        img_path = find_item_image(item_id)
        with col:
            if img_path is not None:
                st.image(str(img_path), width=img_width)
            else:
                st.write("No image")
            st.caption(item_id)

def find_item_image(item_id: str):
    image_dir = Path("Dataset") / "fashion_item_images"
    for ext in [".png", ".jpg", ".jpeg", ".webp"]:
        p = image_dir / f"{item_id}{ext}"
        if p.exists():
            return p
    return None

def find_item_image(item_id: str):
    image_dir = Path("Dataset") / "fashion_item_images"
    for ext in [".png", ".jpg", ".jpeg", ".webp"]:
        p = image_dir / f"{item_id}{ext}"
        if p.exists():
            return p
    return None


def render_outfit_items_small(outfit_id: str, bundle: Dict, max_items: int = 5, img_width: int = 70):
    item_ids = bundle["outfit_items"].get(outfit_id, [])[:max_items]

    if not item_ids:
        st.caption("No items")
        return

    cols = st.columns(len(item_ids))
    for col, item_id in zip(cols, item_ids):
        img_path = find_item_image(item_id)
        with col:
            if img_path is not None:
                st.image(str(img_path), width=img_width)
            else:
                st.write("No image")
            st.caption(item_id)


def render_outfit_items(outfit_id: str, bundle: Dict, max_items: int = 6):
    item_ids = bundle["outfit_items"].get(outfit_id, [])
    if not item_ids:
        st.caption("Không có item cho outfit này.")
        return

    item_ids = item_ids[:max_items]
    cols = st.columns(len(item_ids))

    for col, item_id in zip(cols, item_ids):
        img_path = find_item_image(item_id)
        with col:
            if img_path is not None:
                st.image(str(img_path), use_container_width=True)
            else:
                st.write("No image")
            st.caption(item_id)

def topk_user_outfits(user_id: str, bundle: Dict, k: int = 10):
    if user_id not in bundle["user2idx"]:
        return None

    uidx = bundle["user2idx"][user_id]
    user_vec = bundle["user_emb"][uidx:uidx + 1]
    scores = torch.matmul(user_vec, bundle["outfit_emb"].T).squeeze(0)

    vals, idxs = torch.topk(scores, k=min(k, scores.numel()))
    rows = []
    for score, oidx in zip(vals.tolist(), idxs.tolist()):
        oid = bundle["idx2outfit"][oidx]
        rows.append({
            "rank": len(rows) + 1,
            "outfit_id": oid,
            "score": round(float(score), 4),
            "items": ", ".join(bundle["outfit_items"].get(oid, [])),
        })
    return pd.DataFrame(rows)


def similar_outfits(outfit_id: str, bundle: Dict, k: int = 10):
    if outfit_id not in bundle["outfit2idx"]:
        return None

    oidx = bundle["outfit2idx"][outfit_id]
    q = bundle["outfit_emb"][oidx:oidx + 1]
    scores = torch.matmul(q, bundle["outfit_emb"].T).squeeze(0)
    scores[oidx] = -1e9

    vals, idxs = torch.topk(scores, k=min(k, max(1, scores.numel() - 1)))
    rows = []
    for score, j in zip(vals.tolist(), idxs.tolist()):
        oid = bundle["idx2outfit"][j]
        rows.append({
            "rank": len(rows) + 1,
            "outfit_id": oid,
            "similarity": round(float(score), 4),
            "items": ", ".join(bundle["outfit_items"].get(oid, [])),
        })
    return pd.DataFrame(rows)


def compatibility_score(item_ids: List[str], bundle: Dict):
    valid = [bundle["item2idx"][x] for x in item_ids if x in bundle["item2idx"]]
    if not valid:
        return None, []

    padded = valid[:8]
    if len(padded) < 8:
        padded += [-1] * (8 - len(padded))

    x = torch.tensor([padded], dtype=torch.long)
    with torch.no_grad():
        score = bundle["model"].score_compatibility(bundle["item_emb"], x).item()

    prob = 1 / (1 + torch.exp(torch.tensor(-score))).item()
    label = "Hop" if prob >= 0.5 else "Chua hop"

    return {
        "raw_score": round(float(score), 4),
        "compatibility_prob": round(float(prob), 4),
        "label": label,
    }, [bundle["idx2item"][i] for i in valid]


# =========================
# UI
# =========================
st.set_page_config(page_title="H-FGAT Demo App", layout="wide")
st.title("H-FGAT Recommender Demo")
st.caption("3 chuc nang: user recommendation · outfit compatibility · similar outfit")

try:
    bundle = load_artifacts()
except Exception as e:
    st.error(str(e))
    st.stop()

tab1, tab2, tab3 = st.tabs([
    "Man 1 · User recommendation",
    "Man 2 · Outfit compatibility",
    "Man 3 · Similar outfit",
])
with tab1:
    st.subheader("User recommendation")
    user_id = st.text_input("Nhap user_id")

    if st.button("Recommend outfit"):
        history_df = bundle["history_df"]
        user_history = history_df[history_df["user_id"].astype(str) == user_id.strip()]

        st.markdown("### Top 10 outfit recommend")
        df = topk_user_outfits(user_id.strip(), bundle, k=10)

        if df is None:
            st.warning("user_id khong ton tai trong tap train/export.")
        else:
            # header
            h1, h2, h3, h4 = st.columns([1, 2, 2, 6])
            h1.markdown("**Rank**")
            h2.markdown("**Outfit ID**")
            h3.markdown("**Score**")
            h4.markdown("**Items / Images**")

            for row in df.itertuples(index=False):
                c1, c2, c3, c4 = st.columns([1, 2, 2, 6])

                c1.write(row.rank)
                c2.write(row.outfit_id)
                c3.write(row.score)

                with c4:
                    render_outfit_items_small(row.outfit_id, bundle, max_items=5, img_width=65)

                st.divider()

        st.markdown("### Lich su tuong tac")
        if len(user_history) > 0:
            hist_outfits = user_history["outfit_id"].astype(str).tolist()

            for oid in hist_outfits[:5]:
                c1, c2 = st.columns([2, 8])
                c1.write(oid)
                with c2:
                    render_outfit_items_small(oid, bundle, max_items=5, img_width=60)
                st.divider()
        else:
            st.info("User chua co interaction.")



with tab2:
    st.subheader("Cham diem compatibility cho mot nhom item")
    item_text = st.text_area(
        "Nhap item_id, phan tach bang dau phay",
        placeholder="VD: 1001,1002,1003"
    )

    if st.button("Check compatibility"):
        item_ids = [x.strip() for x in item_text.split(",") if x.strip()]
        result, valid_items = compatibility_score(item_ids, bundle)

        if result is None:
            st.warning("Khong co item_id hop le.")
        else:
            c1, c2, c3 = st.columns(3)
            c1.metric("Label", result["label"])
            c2.metric("Compatibility prob", result["compatibility_prob"])
            c3.metric("Raw score", result["raw_score"])

            st.markdown("### Items da nhap")
            cols = st.columns(min(len(valid_items), 6)) if valid_items else []
            for col, item_id in zip(cols, valid_items[:6]):
                img_path = find_item_image(item_id)
                with col:
                    if img_path is not None:
                        st.image(str(img_path), use_container_width=True)
                    else:
                        st.write("No image")
                    st.caption(item_id)

with tab3:
    st.subheader("Tim outfit tuong tu")
    outfit_id = st.text_input("Nhap outfit_id", placeholder="VD: o456")

    if st.button("Find similar outfit"):
        query_id = outfit_id.strip()

        if query_id not in bundle["outfit2idx"]:
            st.warning("outfit_id khong ton tai trong tap train/export.")
        else:
            # ===== 1) show outfit input =====
            st.markdown("### Outfit duoc nhap vao")
            q1, q2 = st.columns([2, 8])
            q1.write(f"**Outfit ID:** {query_id}")
            with q2:
                render_outfit_items_small(query_id, bundle, max_items=5, img_width=55)

            st.divider()

            # ===== 2) show similar outfits in table-like layout =====
            df = similar_outfits(query_id, bundle, k=10)

            if df is None or len(df) == 0:
                st.info("Khong tim thay outfit tuong tu.")
            else:
                st.markdown("### Ket qua outfit tuong tu")

                h1, h2, h3, h4 = st.columns([1, 2, 2, 7])
                h1.markdown("**Rank**")
                h2.markdown("**Outfit ID**")
                h3.markdown("**Similarity**")
                h4.markdown("**Items / Images**")

                for row in df.itertuples(index=False):
                    c1, c2, c3, c4 = st.columns([1, 2, 2, 7])

                    c1.write(row.rank)
                    c2.write(row.outfit_id)
                    c3.write(row.similarity)

                    with c4:
                        render_outfit_items_small(
                            row.outfit_id,
                            bundle,
                            max_items=5,
                            img_width=55
                        )

                    st.divider()

with st.expander("Ghi chu demo / pipeline"):
    st.markdown(
        """
**Pipeline:**
1. Subsample ~30k user.
2. Loc outfit va item lien quan.
3. Chi embed item thuoc subsample bang ResNet + BERT.
4. Build graph `user -> outfit -> item`.
5. Train H-FGAT-style model.
6. Export san `user/outfit/item embeddings`.
7. App load embedding da export de suy luan nhanh.

**Uu diem cho demo:**
- khong can encode lai moi lan mo app
- suy luan nhanh
- de giai thich
- de mo rong sang API sau
"""
    )