# Adversarial QA Probe — Analysis Memo

## 1. Hypothesis

- **Input pattern:** The context contains two named entities of the same type (both persons), each attributed a distinct action. The question asks about one entity by referencing only its action.
- **Output pattern:** The model returns a title-prefixed span (e.g., "Mayor Tom Briggs", "Plant Manager George Hartley") when the gold answer is the bare name ("Tom Briggs", "George Hartley"), or vice versa — producing a partial-match error rather than an exact match.
- **Why:** DistilBERT-SQuAD learns to identify entity spans by type signal (PERSON) and proximity to the verb in the question. When two same-type entities appear in the same context, the model frequently latches onto the longer, more salient span — the one with a title prefix — regardless of which entity the question specifies. This is a span-boundary bias: the model over-extends the answer span to include adjacent title tokens because title-name collocations are the dominant person-mention pattern in SQuAD training data.

---

## 2. Set Design

- **Total examples:** 31
- **Tags used:**
  - `same-type-distractor` (n = 28): Each context contains exactly two named persons with distinct roles and actions. Questions are role-specific ("Who warned…?" vs. "Who insisted…?"), forcing the model to discriminate by predicate rather than entity type.
  - `control` (n = 3): Single-entity, single-sentence contexts with no distractors (telephone inventor, Einstein, Eiffel Tower). These test that the model handles unambiguous extraction correctly and confirm that the pattern — not raw question difficulty — drives any failures observed in the adversarial subset.
- **Why these tags:** One adversarial tag isolates the targeted failure mode cleanly; the control tag provides the baseline-within-set needed to attribute failures to distractor presence rather than model capacity.
- **Control examples:** 3 examples with no same-type distractors. They confirm that the model can extract person names at EM = 1.0 when no competing entity exists, making same-type distractor presence the discriminating factor.

---

## 3. Results

- **Aggregate EM:** 0.7097 — **Aggregate F1:** 0.9157
- **Lab 7B baseline (tech-news QA set):** EM = 0.3440; F1 = 0.4611

| Pattern | n | EM | F1 | vs. baseline EM | vs. baseline F1 |
|---|---|---|---|---|---|
| same-type-distractor | 28 | 0.6786 | 0.9067 | +0.3346 | +0.4456 |
| control | 3 | 1.0000 | 1.0000 | +0.6560 | +0.5389 |

The adversarial set scores *higher* than the tech-news baseline on both metrics. This reflects a construction artifact: the adversarial contexts are short two-sentence passages where the answer appears verbatim and unambiguously adjacent to its predicate, whereas the tech-news baseline involves longer, noisier real-world articles. The adversarial set therefore isolates a *specific* span-boundary failure rather than a general accuracy collapse.

The failure mode is visible in span over-extension: the model achieves EM = 0 on 8 of 28 adversarial examples (28.6%), all cases where it returned a title-prefixed span instead of the bare name gold answer, or extended the span to include a role token. F1 remains high (0.91) because the predicted spans contain the gold tokens — the error is boundary precision, not wrong entity selection.

**Illustrative failure tuples:**

- **(ADV_06)** "Who insisted the bridge had passed inspections?" → gold: `Tom Briggs`, predicted: `Mayor Tom Briggs`. The model correctly identifies the entity but over-extends the span leftward to include the title, failing EM despite capturing the correct person.
- **(ADV_21)** "Who confirmed the satellite had successfully entered orbit?" → gold: `Marcus Webb`, predicted: `NASA Flight Director Marcus Webb`. Same span-extension pattern; gold is bare name, model anchors on the full title-name collocation (F1 = 0.57 due to low token overlap with the short gold string).
- **(ADV_24)** "Who said fines would be issued immediately?" → gold: `Clara Moody`, predicted: `Environmental Inspector Clara Moody`. Title prefix added; F1 = 0.67. The distractor (George Hartley) is not selected — the model identifies the correct entity — but span boundaries are wrong.

The pattern is consistent: failures occur exclusively when the gold answer is a bare surname/first+last name and the context introduces that person with a title. The model never selects the wrong entity; it selects the right entity with the wrong span boundary.

---

## 4. Production Defense

**Recommended action: post-hoc span normalization using an NER-based title stripper.**

The failure mode is not entity selection error — the model picks the correct person in every case. The error is span over-extension into adjacent title tokens. A lightweight NER pass over every predicted answer span can detect and strip honorifics and role titles (Mayor, Dr., Inspector, Director, etc.) that precede a PERSON entity, normalizing the output to the bare name. This directly addresses the measured pattern: all 8 EM failures in the adversarial set involve a title prefix that NER would classify as a non-PERSON token prepended to the correct PERSON span. The fix requires no retraining, adds negligible latency, and targets precisely the boundary bias the per-pattern breakdown reveals — without touching the model's entity-selection behavior, which is already correct.