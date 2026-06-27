# Humanizer: Vietnamese support notes

## Quick answer

**The upstream humanizer skill does not officially support Vietnamese.** All 33 patterns, examples, and AI vocabulary lists are **English**. README and SKILL.md never mention Vietnamese.

**Partial support is possible** by applying language-agnostic patterns plus a short list of Vietnamese AI tells below. Quality will be lower than English humanization.

---

## Patterns that transfer to Vietnamese (apply in any language)

From the main humanizer skill, these still apply to Vietnamese academic drafts:

| # | Pattern | Vietnamese note |
|---|---------|-----------------|
| 14 | Em/en dash | Same hard ban as `write-lncs-paper` |
| 10 | Rule of three | "ba điểm chính", forced triple lists |
| 5 | Vague attributions | "theo các chuyên gia", "nhiều nghiên cứu cho thấy" without citation |
| 4 | Promotional tone | "vượt trội", "đột phá", "toàn diện", "mạnh mẽ" without evidence |
| 23 | Filler | "nhằm mục đích", "trong bối cảnh hiện tại", "cần lưu ý rằng" |
| 24 | Excessive hedging | "có thể có khả năng", "dường như như là" |
| 20 | Chatbot artifacts | "Hy vọng điều này hữu ích", "Hãy cho tôi biết nếu bạn cần" |
| 28 | Signposting | "Hãy cùng tìm hiểu", "Trong phần này chúng ta sẽ" |
| 29 | Fragmented headers | Heading plus one-line restatement; also `**Nhãn.**` body fragments (use `Về [chủ đề], ...`) |
| 25 | Generic conclusions | "Tương lai hứa hẹn nhiều điều tích cực" |

---

## Vietnamese-specific AI tells (informal list)

Watch and rewrite:

- **Inline bold fragment headers:** `**Luồng xử lý tổng thể.** Dữ liệu thô...` → `Về luồng xử lý tổng thể, dữ liệu thô...` (English: `Regarding the overall pipeline, raw data...` or `In terms of the overall pipeline, raw data...`). Exception: table/figure captions (`**Bảng 1.**`, `**Table 1.**`) keep LNCS form.
- **Unused formal notation:** `Gọi G = (V, E) với V = V_U ∪ V_O ∪ V_I` when \(V_U\), \(V_O\), \(V_I\) never appear again. Prefer plain prose unless symbols recur in equations.
- **Stacked intro formulas:** "Trong thời đại công nghệ 4.0...", "Trong bối cảnh phát triển mạnh mẽ của trí tuệ nhân tạo..."
- **Over-polished parallelism:** Three clauses with identical grammar and length
- **Translationese:** Calques from English ("đóng góp vào", "đóng vai trò then chốt", "nhấn mạnh tầm quan trọng")
- **Empty breadth:** "không chỉ... mà còn..." repeated across paragraphs
- **Meta narration:** "Như đã trình bày ở trên", "Như đã đề cập" every paragraph
- **Uniform sentence length:** All medium-length sentences, no variation

**Preserve for LNCS/thesis drafts:**

- **chúng tôi** (appropriate for papers)
- Technical English terms (Hit Rate@10, NDCG, graph attention)
- Formal register (do not casualize into blog voice)

---

## Recommended workflow for Vietnamese HFGAT content

1. Draft in Vietnamese (`write-lncs-paper` VI mode)
2. Humanize with this file + humanizer §14 and filler/signposting rules
3. Re-scan for em dash and AI filler
4. Convert to English LNCS when ready (`write-lncs-paper` VI→EN)
5. Humanize **English** with full humanizer skill before submission

---

## Future improvement

A dedicated `humanizer-vi.md` pattern list (mirroring the 33 English patterns) would be needed for full Vietnamese parity. Upstream repo: https://github.com/blader/humanizer
