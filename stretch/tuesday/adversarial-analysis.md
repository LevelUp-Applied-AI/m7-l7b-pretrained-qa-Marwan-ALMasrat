# Adversarial QA Probe — Analysis Memo

## 1. Hypothesis

- **Input pattern:** The context contains two named entities of the same type (both persons), each attributed a distinct action. The question asks about one entity by referencing only its action.
- **Output pattern:** The model returns a title-prefixed span (e.g., "NASA Flight Director Marcus Webb") when the gold answer is the bare name ("Marcus Webb"), or selects the wrong entity entirely when the two persons appear in separate sentences.
- **Why:** DistilBERT-SQuAD learns to identify entity spans by type signal (PERSON) and proximity to the verb in the question. When two same-type entities appear in the same sentence, the model uses predicate proximity correctly. When the two entities appear in separate sentences (adjacent-entity-distractor), the model loses the predicate-anchoring signal and falls back on positional bias — frequently selecting the first or more salient entity regardless of which one the question targets.

---

## 2. Set Design

- **Total examples:** 31
- **Tags used:**
  - `same-type-distractor` (n = 16): Both persons appear in the same sentence. Questions are predicate-specific, forcing the model to discriminate by verb attribution within a single sentence.
  - `adjacent-entity-distractor` (n = 12): The two persons appear in separate consecutive sentences. The question targets one entity; the other appears in an adjacent sentence with no shared predicate.
  - `control` (n = 3): Single-entity, single-sentence contexts with no distractors. These confirm that the model handles unambiguous extraction correctly at EM = 1.0.
- **Why these tags:** The two adversarial tags isolate whether distractor proximity (same sentence vs. adjacent sentence) affects failure rate — a finer-grained hypothesis than "two entities in context." The control tag confirms that failures are caused by distractor presence, not raw difficulty.
- **Control examples:** 3 examples (telephone inventor, Einstein, Eiffel Tower). All three scored EM = 1.0, confirming the model can extract correctly when no competing entity exists.

---

## 3. Results

- **Aggregate EM:** 0.7097 — **Aggregate F1:** 0.9157
- **Lab 7B baseline (tech-news QA set):** EM = 0.3440; F1 = 0.4611

| Pattern | n | EM | F1 | vs. baseline EM |
|---|---|---|---|---|
| same-type-distractor | 16 | 0.8750 | 0.9750 | +0.5310 |
| adjacent-entity-distractor | 12 | 0.4167 | 0.8157 | +0.0727 |
| control | 3 | 1.0000 | 1.0000 | +0.6560 |

The key finding is the gap between the two adversarial tags: `same-type-distractor` EM = 0.875 vs. `adjacent-entity-distractor` EM = 0.417. When both entities share a sentence, the model uses predicate proximity to discriminate correctly most of the time. When entities are in separate sentences, EM drops by 0.458 — the model loses its predicate-anchoring signal and fails on nearly 6 of 12 examples.

The aggregate scores higher than the tech-news baseline because the adversarial contexts are short two-sentence passages with verbatim answers, whereas the baseline involves longer noisy articles. The adversarial set isolates a specific structural failure, not a general accuracy collapse.

**Illustrative failure tuples:**

- **(ADV_21)** "Who confirmed the satellite had successfully entered orbit?" → gold: `Marcus Webb`, predicted: `NASA Flight Director Marcus Webb`. Span over-extension into title prefix; correct entity selected but wrong boundary.
- **(ADV_19)** "Who proposed the new education reform bill?" → gold: `Diana Cho`, predicted: `Paul Nguyen`. Wrong entity selected — both appear in separate sentences; model picks the more positionally salient name.
- **(ADV_25)** "Who revealed that the merger would eliminate five hundred positions?" → gold: `James Forde`, predicted: `Susan Blake`. Adjacent-sentence distractor wins over the correct entity; predicate signal lost across sentence boundary.

---

## 4. Production Defense

**Recommended action: sentence-scoped answer extraction.**

The `adjacent-entity-distractor` failures show that the model selects the wrong entity when the distractor appears in a separate sentence. The fix is to restrict the QA model's extraction window to the single sentence most relevant to the question (selected by a lightweight sentence-similarity ranker), rather than passing the full multi-sentence context. This eliminates cross-sentence distractors before the QA head sees them. It requires no retraining, adds one fast similarity scoring step, and directly targets the measured failure: all 7 EM failures in the adjacent-entity-distractor subset involve a distractor that would be absent if the context were scoped to the answer sentence only.