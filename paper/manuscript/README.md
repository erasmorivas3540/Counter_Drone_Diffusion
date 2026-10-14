# Manuscript skeleton: Military Operations Research (MORS)

Structure and placeholders only; nothing has been drafted. Build with `latexmk -pdf main.tex`.

## MOR requirements (2020 editorial policy; confirm against the current version)

- [ ] 10-35 pages, double spaced, 12-point font, illustrations included
- [ ] **All figures grayscale**, 300 dpi, each a separate camera-ready file, numbered and captioned
- [ ] No footnotes: parenthetical notes go in a Notes section (superscript letters) before acknowledgments
- [ ] Headings: major sections capitalized and bold; subsections bold Title Case; sub-subsections Title Case
- [ ] Equations numbered sequentially; avoid long expressions
- [ ] Acronyms defined on first use; keep to a minimum
- [ ] References: author-year in text; alphabetical list; "et al." for more than three authors
- [ ] Separate statement of the major contribution (`statement_of_contribution.tex`)
- [ ] Choose application area(s) and OR method(s) from the policy's Table 1
- [ ] Unclassified; release statement if appropriate; not under review elsewhere
- [ ] Submission: ScholarOne, https://mc04.manuscriptcentral.com/mor_journal (Word or PDF for review)
- [ ] After acceptance: final manuscript in **Microsoft Word**, track changes removed, separate figure files
- [ ] Check MORS policy on AI-assisted research and add a disclosure if required
- [ ] Single-blind review (author names stay on the manuscript)

## Page budget (target about 30 of the 35 pages, figures included)

| Section | File | Pages |
|---|---|---|
| Abstract | `sections/00_abstract.tex` | 0.5 |
| Introduction | `sections/01_introduction.tex` | 3 |
| Related work | `sections/02_related_work.tex` | 2 |
| Model | `sections/03_model.tex` | 3 |
| Numerical methods and validation | `sections/04_methods_validation.tex` | 3 |
| Q1-Q2: does randomness help the drone? | `sections/05_randomness.tex` | 4 |
| Q3: where should the sensors go? | `sections/06_placement.tex` | 6 |
| Q4: downtime and redundancy | `sections/07_downtime.tex` | 4 |
| Real-world interpretation | `sections/08_real_world.tex` | 3 |
| Discussion and conclusion | `sections/09_discussion.tex`, `10_conclusion.tex` | 2.5 |
| Back matter (notes, acknowledgments, references) | `sections/11_back_matter.tex` | ~1.5 |
| Appendices | `sections/12_appendices.tex` | outside budget if possible |

## Figures planned (all need grayscale re-renders)

1. `figures/validation.png` (Section 4)
2. `figures/regime_map.png` (Section 5)
3. `figures/regime_equal_time.png` (Section 5)
4. `figures/regime_gap_test.png` or `regime_layouts.png` (Section 5)
5. `figures/q3_best_ring.png` (Section 6)
6. `figures/q3_adjoint_search.png` (Section 6)
7. `figures/q4_downtime_switching.png` (Section 7)
8. `figures/q4_redundancy.png` (Section 7)

## Existing material to draw on

- Model and methods: `README.md`, `src/cdd/`, the model.py and exact.py docs
- Validation: `results/validation.md`
- Theory: `paper/theory_section.tex` (draft)
- Results: `results/*.csv`, `results/*.md`, `notebooks/findings_so_far.ipynb`
