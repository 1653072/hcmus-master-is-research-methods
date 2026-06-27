# LNCS Reference: Springer Proceedings Rules

Condensed from Springer *Instructions for Authors of Papers to be Published in Springer Computer Science Proceedings* and `llncs.cls` documentation. When a conference publishes stricter rules, **venue rules override** this file.

---

## Document class and files

| Item | Rule |
|------|------|
| Class | `\documentclass[runningheads]{llncs}` |
| Bibliography | `\bibliographystyle{splncs04}` + BibTeX `.bib` |
| Fonts | Use template fonts (CMR for LaTeX); avoid custom font packages |
| Source | Submit all `.tex`, `.bib`, `.bbl`, figures, and matching PDF |

Avoid self-defined environments that conflict with the class. Do not use `\textheight`, `\vspace` hacks, or `baselinestretch` except in rare cases.

---

## Title and authors

- Full author names at top; given name before surname
- Affiliations: institution, city, country (not full postal address)
- Multiple affiliations: superscript Arabic numerals, one per line
- **One** corresponding author; email mandatory (mark with `\email{}`)
- ORCID encouraged: `\orcidID{...}`
- `\authorrunning{F. Author et al.}` for short running head if needed
- No academic titles in addresses (omit or footnote on first page)

---

## Headings

| Level | Example | Style |
|-------|---------|-------|
| Title | Centered | 14 pt bold |
| 1st | `1 Introduction` | 12 pt bold, numbered |
| 2nd | `2.1 Subsection` | 10 pt bold, numbered |
| 3rd | **Run-in Heading.** Text… | 10 pt bold, unnumbered |
| 4th | *Lowest Level Heading.* Text… | 10 pt italic, unnumbered |

Rules:
- Capitalize major words (nouns, verbs, adjectives, etc.); articles/prepositions/conjunctions lowercase unless first word
- Hyphenated headings: capitalize second part only if first word stands alone (see Springer Table 1 examples)
- Do **not** number section 0
- Only levels 1–2 are numbered

---

## Prose (paper output)

- **Never use em dash (`—`)** in abstract, body, captions, or tables. Use comma, period, colon, or parentheses.
- American English preferred by Springer

- Mandatory; **70–150 words**
- Summarize problem, approach, and main results
- No footnotes in abstract
- Keywords in `\keywords{...}` inside `abstract`; capitalize first letter; `\and` between keywords

---

## Page length

Typical proceedings:
- **Short papers**: ~6–11 pages
- **Full papers**: ~12–15+ pages
- Very short (<4 pages) may move to back matter (not indexed as separate paper)

Confirm exact limit with the conference CFP.

---

## Figures and tables

**Figures**
- Caption **below** figure, 9 pt
- Numbered consecutively: Fig. 1, Fig. 2
- Referenced in text before or near appearance
- Vector preferred; ≥800 dpi (1200 dpi preferred) for line art
- Label text ≥6 pt (~2 mm height)
- Color allowed in electronic version; ensure legibility in grayscale for print
- No color in body text or equations

**Tables**
- Caption **above** table, 9 pt
- Use editable text (not pasted images)
- `booktabs` (`\toprule`, `\midrule`, `\bottomrule`) is standard in CS papers

**Accessibility**
- Springer may require alt text for figures; provide short descriptive alt text when asked

---

## Equations

- Display equations centered on own line
- Numbered consecutively: `(1)`, `(2)`, right-aligned, no section prefix
- Punctuate like sentences (comma before continued condition, period at end)
- Editable math (not images); no color

---

## Theorems and definitions

Use LNCS-provided environments (`theorem`, `lemma`, `proof`, etc.).

- Number **globally**: Lemma 1, Lemma 2, Theorem 3
- **Not** Lemma 2.1 (no section counter in theorem numbers)

---

## Citations

**In text (numeric style)**
- Single: `Miller [9]` or `... as shown in [9].`
- Multiple: `[4-6, 9]` in numerical order
- Brackets, not superscript

**Reference list**
- Alphabetical order (splncs04 default) or order of citation, be consistent
- Author names: `Surname, Initials`
- Include DOI in `.bib` `doi = {...}` field when available
- Conference papers: `In: Proc. Name. LNCS, vol. X, pp. Y–Z. Springer (year)`
- More than six authors → shorten with `et al.` (BibTeX style handles this)

**splncs04 entry types (typical)**

```bibtex
@inproceedings{key,
  author    = {Author, A. and Author, B.},
  title     = {Paper Title},
  booktitle = {Proc. Conference Name},
  series    = {LNCS},
  volume    = {1234},
  pages     = {1--12},
  year      = {2024},
  publisher = {Springer},
  doi       = {10.1007/...}
}

@article{key,
  author  = {Author, A.},
  title   = {Article Title},
  journal = {Journal Name},
  volume  = {10},
  pages   = {1--20},
  year    = {2023},
  doi     = {10.1007/...}
}
```

---

## Footnotes

- Superscript numeral after word or after punctuation in phrase
- **No footnotes in abstract**
- Header remarks (title/authors): symbols, not numbers

---

## Program code

Inline code: typewriter font. Listings: use `listings` or `minted` sparingly; keep pages readable.

---

## Back matter

**Appendix** (if any): before References. Single appendix = "Appendix"; multiple = "Appendix 1", "Appendix 2". Continue figure/table/equation numbering.

**Acknowledgments**: `\begin{credits}` with `\subsubsection{\ackname}`

**Competing interests**: `\subsubsection{\discintname}`, often required

**Disclosure of interests example**
> The authors have no competing interests to declare that are relevant to the content of this article.

---

## Cross-references (LaTeX)

Use `cleveref`:
- `\Cref{sec:method}` → "Sect. 3" at sentence start
- `\cref{fig:arch}` → "Fig. 2" mid-sentence

LNCS convention: **Sect.**, **Fig.**, **Table**, **Eq.** (template-specific; match sample paper).

---

## AI-generated content

Springer Nature has policies on AI use in manuscripts. Disclose AI assistance if required by the venue or publisher policy at submission time.

---

## Common desk-reject formatting mistakes

1. Wrong document class or modified margins
2. Abstract outside 70–150 words
3. Three levels of numbered sections (1, 1.1, 1.1.1)
4. Table captions below table
5. Missing DOIs / wrong bibliography style
6. Figures as low-resolution raster screenshots
7. Hyperlinks only in PDF references (splncs04 print style: no URLs in ref list unless venue allows)
