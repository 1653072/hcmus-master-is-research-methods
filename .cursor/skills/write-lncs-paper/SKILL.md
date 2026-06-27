---
name: write-lncs-paper
description: >-
  Draft and revise Springer LNCS conference papers in English or Vietnamese,
  including Vietnamese-first drafting then LNCS English for submission. Covers
  section structure, prose tone, abstract/keywords, figures/tables, citations,
  and llncs.cls. Use for /write-lncs-paper, bài LNCS, viết paper Springer,
  HFGAT/LNCS content, chuyển tiếng Việt sang tiếng Anh LNCS, or llncs.cls help.
  Chains to humanizer skill before final file write or chat delivery.
---

# Write LNCS Paper (English + Vietnamese)

Skill for drafting and revising **Springer LNCS** papers in **Vietnamese and/or English**. Final submission text is **American English**; Vietnamese is supported for ideation, first drafts, and review with the author.

**References (read when unsure):**
- [reference.md](reference.md): Springer formatting (English submission)
- [examples.md](examples.md): English section snippets
- [examples-vi.md](examples-vi.md): Vietnamese section snippets
- [project-context.md](project-context.md): **project repo only**; HFGAT terminology and metrics rules
- [humanizer](../humanizer/SKILL.md): mandatory final pass before saving paper content (see Step 8)

---

## Forbidden punctuation (mandatory)

**NEVER use the em dash character (U+2014, `—`) in any generated paper content**, English or Vietnamese. It is a common AI tell and must not appear in abstracts, sections, captions, tables, or lists.

Before delivering any draft:
1. Scan the full output for `—` and remove every occurrence.
2. Rewrite with a comma, period, colon, parentheses, or two short sentences.

| Instead of | Use |
|------------|-----|
| `X — Y` (aside or break) | `X. Y` or `X, Y` or `X (Y)` |
| `We propose X — a method that…` | `We propose X, a method that…` |
| Vietnamese `A — B` | Same rule: comma, period, or parentheses |

En dash in number ranges (e.g. `70-150` words) is acceptable. Hyphen in compounds (e.g. `user-outfit`) is acceptable. **Em dash is not.**

Apply this rule to **all skill examples** when generating new content, not only final LaTeX.

---

## Language modes

Detect intent from the user message. If unclear, **ask once**.

| Mode | Trigger | Output |
|------|---------|--------|
| **VI draft** | User writes in Vietnamese; "nháp tiếng Việt", "viết HFGAT bằng tiếng Việt" | Vietnamese prose, LNCS structure |
| **EN LNCS** | "LNCS English", "bản tiếng Anh", "submit", LaTeX | American English, submission-ready |
| **VI → EN** | "dịch sang LNCS", "chuyển sang tiếng Anh" | English LNCS + optional short change notes |
| **Bilingual** | "song ngữ", "cả hai" | Vietnamese block then English block (same section) |

**Default for this workflow:** Vietnamese first for content generation → English LNCS when the user asks to finalize or submit.

**Submission rule:** Abstract, body, captions, and keywords for Springer must be **English** (70–150 word abstract). Keep a Vietnamese working draft only in repo notes if the user wants it, not in the `.tex` unless they explicitly ask.

---

## When to use

- `/write-lncs-paper`
- Draft/revise: Tóm tắt (Abstract), Giới thiệu, Các nghiên cứu liên quan, Phương pháp, Thực nghiệm, Kết luận
- Generate **HFGAT / Lightweight** paper content in Vietnamese first
- Convert thesis or report text to LNCS English
- Check LaTeX, references, or compliance

**Ask first if missing:**
- Output language for *this* turn (VI / EN / both)
- Page limit (short ~6–11 pp vs full ~12–15+ pp)
- LaTeX `.tex` vs plain prose
- Single-blind vs double-blind
- Venue-specific extra sections

**In this repository:** always read [project-context.md](project-context.md) before drafting HFGAT / Lightweight content (terminology, metrics table, v1.3 config).

---

## Workflow

```
LNCS draft progress:
- [ ] Language mode confirmed (VI draft / EN / VI→EN / bilingual)
- [ ] Scope, contribution, page limit locked
- [ ] Outline (≤2 numbered heading levels)
- [ ] Abstract (EN: 70–150 words) + keywords
- [ ] Body sections
- [ ] Figures/tables + captions + in-text refs
- [ ] Conclusion (no new results)
- [ ] Acknowledgments / competing interests (if required)
- [ ] References (splncs04, DOIs)
- [ ] EN compliance pass (checklist below)
- [ ] **Humanizer pass (Step 8) before write to file or final chat delivery**
```

### Step 1: Lock the contribution (any language)

State in 2–3 sentences / 2–3 câu:
1. **Problem** / Vấn đề
2. **Gap** / Khoảng trống nghiên cứu
3. **Contribution** / Đóng góp (2–4 bullets)

### Step 2: Section map (EN ↔ VI)

| EN (LNCS) | VI (working draft) |
|-----------|-------------------|
| Abstract | Tóm tắt |
| 1 Introduction | 1 Giới thiệu |
| 2 Related Work | 2 Các nghiên cứu liên quan |
| 3 Proposed Method | 3 Phương pháp đề xuất |
| 4 Experiments | 4 Thực nghiệm và đánh giá |
| 5 Conclusion | 5 Kết luận |
| Acknowledgments | Lời cảm ơn |
| References | Tài liệu tham khảo |

Keep **at most two numbered heading levels** in both languages.

### Step 3: Prose style

#### English (final LNCS)

- American English; short paragraphs (3–6 sentences)
- Active voice: "We propose…", "We evaluate…"
- Use **we**; avoid "the reader" and "In this section we will…"
- Confident but precise; **implemented** vs **proposed** vs **planned**
- No marketing hype without evidence
- No variable names, file paths, or notebook mentions in prose
- Avoid semicolon-heavy sentences
- **Never use em dash (`—`)** in paper prose (see Forbidden punctuation above)
- Define acronyms once: "Graph Attention Network (GAT)"
- Cross-refs: "Sect. 4", "Fig. 2", "Table 1"

#### Vietnamese (working draft)

- Văn phong học thuật, rõ ràng, câu ngắn vừa phải
- Dùng **chúng tôi**; tránh "người đọc", "ở phần này chúng tôi sẽ…"
- Khiêm tốn nhưng tự tin, chỉ khẳng định điều đã có bằng chứng thực nghiệm
- Phân biệt **đã triển khai** / **đề xuất** / **dự kiến**
- Không dùng tên biến, đường dẫn file, hoặc notebook trong văn bản
- Hạn chế dấu chấm phẩy; tách thành hai câu
- **Không dùng em dash (`—`)** trong văn bản báo (xem Forbidden punctuation)
- Giữ thuật ngữ kỹ thuật tiếng Anh khi phổ biến (Hit Rate@10, NDCG, Fill In The Blank, graph attention)
- Định nghĩa viết tắt lần đầu: "mạng Graph Attention (GAT)"
- Tham chiếu: "Mục 4", "Hình 2", "Bảng 1"

### Step 4: Vietnamese → English pass

When converting VI → EN LNCS:

1. Preserve **all numbers, metrics, and factual claims** exactly
2. Replace informal VI with standard CS English (not literal word-for-word)
3. Apply Springer heading/caption rules in English
4. Short change notes only if terminology shifted (e.g. official method name)

### Step 5: Abstract and keywords (English only for submission)

- **70–150 words**; no citations or footnotes
- Structure: context → problem → approach → main result → implication
- `\keywords{Keyword one \and Keyword two}`, capitalize first letter of each

### Step 6: Experiments

- Define metrics; state when cross-paper comparison is not apples-to-apples
- Table caption **above**; figure caption **below**
- Interpret results, do not only repeat the table

### Step 7: References

- Numeric citations: `[3]`, `[4-6,9]`
- `splncs04.bst`; DOIs when available

### Step 8: Humanize (mandatory before delivery)

**Every newly generated paper section must pass through the humanizer skill before** writing to a destination file (`.tex`, `.md`, thesis doc) **or** presenting as final content in chat. Do not skip for "small" edits or single paragraphs.

#### Procedure

1. **Draft** the section (Steps 1–7), including LNCS structure and forbidden-punctuation scan.
2. **Load humanizer:** read [../humanizer/SKILL.md](../humanizer/SKILL.md). For Vietnamese drafts, also read [../humanizer/vietnamese.md](../humanizer/vietnamese.md). If `../humanizer/project-context.md` exists in the current project, read it.
3. **Run humanizer Process and Output** on the full section text:
   - Identify AI patterns
   - Draft rewrite → brief "still-AI" audit → **final rewrite**
4. **LNCS re-guard** on the humanizer output (humanizer can reintroduce tells):
   - No em dash (`—`) or en dash (`–`) unless hyphen in a compound or range
   - All metrics, names, and factual claims unchanged
   - Section headings, labels (`Sect.`, `Fig.`, `Bảng.`), citations intact
   - Academic register preserved (no blog voice, no chatbot closers)
   - `write-lncs-paper` voice rules still apply (`we` / **chúng tôi**, no notebook paths)
5. **Deliver** only the Step 8 final text:
   - **If saving to a file:** write the humanized version, not the pre-humanizer draft
   - **If replying in chat:** show the humanized version as the main body; optional one-line note "Humanized" unless user wants audit bullets

#### When to invoke explicitly

User may also run `/humanizer` alone. When using `/write-lncs-paper`, Step 8 is **automatic** and does not require a separate user command.

#### Exemptions (humanizer optional)

- Raw outline bullets or contribution stubs not meant for publication
- User says "skip humanizer" or "draft only, no polish"
- LaTeX preamble, `\bib` entries, or `\cite{key}` only (no prose)

---

## LaTeX quick template

Use official `llncs.cls`. See [reference.md](reference.md) for full rules.

```latex
\documentclass[runningheads]{llncs}
% ... packages ...
\begin{document}
\title{Paper Title}
% ... authors ...
\begin{abstract}
Abstract (70--150 words, English).
\keywords{Keyword one \and Keyword two}
\end{abstract}
\section{Introduction}
% ...
\bibliographystyle{splncs04}
\bibliography{references}
\end{document}
```

---

## Output modes

| Request | Deliver |
|---------|---------|
| "Viết mục X bằng tiếng Việt" | Vietnamese LNCS prose → **Step 8 humanize** → deliver |
| "Draft section X" (English context) | English LNCS prose → **Step 8 humanize** → deliver or save |
| "Dịch / chuyển sang LNCS" | English LNCS → **Step 8 humanize** → deliver |
| "Song ngữ mục X" | VI block then EN block, **each humanized** in Step 8 |
| "Revise for LNCS" | Edited text → **Step 8 humanize** → deliver |
| Table / figure caption | Caption → **Step 8 humanize** → deliver |

Preserve user facts and numbers unless they request correction.

---

## Pre-submission checklist (English)

- [ ] **No em dash (`—`) anywhere in the manuscript**
- [ ] Abstract 70–150 words; keywords present
- [ ] Only two numbered section levels
- [ ] Figure captions below; table captions above
- [ ] All floats cited in text
- [ ] Equations numbered `(1)`, `(2)` globally
- [ ] No footnotes in abstract
- [ ] splncs04 + DOIs
- [ ] Competing interests / acknowledgments if required
- [ ] Page limit met
- [ ] **Humanizer pass completed (Step 8)**

---

## Additional resources

- Springer LNCS: https://www.springer.com/gp/computer-science/lncs/conference-proceedings-guidelines
- [reference.md](reference.md) · [examples.md](examples.md) · [examples-vi.md](examples-vi.md)
