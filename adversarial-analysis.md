# Adversarial QA Probe — Analysis Memo

> Replace each placeholder section. Memo target: ~1 page. The TA rubric rewards specificity grounded in your data.

## 1. Hypothesis

State your targeted failure mode operationally:
- **Input pattern:** _(what kind of input triggers the failure?)_
- **Output pattern:** _(what does the model do wrong?)_
- **Why you hypothesize this:** _(2–3 sentences on the underlying cause — model architecture, training data gap, span-prediction bias, etc.)_

## 2. Set Design

- Total examples: _(N)_
- Tags used: _(list pattern_tag values and their counts)_
- Why these tags: _(one sentence per tag)_
- Control examples: _(how many; what they isolate; why they confirm the pattern is the discriminator)_

## 3. Results

- Aggregate EM: _(value)_; Aggregate F1: _(value)_
- Lab 7B baseline (from your `qa_metrics.json`): EM _(value)_; F1 _(value)_
- Per-pattern_tag breakdown:

| Pattern | n | EM | F1 | vs. baseline |
|---|---|---|---|---|
| _(tag 1)_ | _(n)_ | _(em)_ | _(f1)_ | _(±diff)_ |
| _(tag 2)_ | _(n)_ | _(em)_ | _(f1)_ | _(±diff)_ |
| control | _(n)_ | _(em)_ | _(f1)_ | _(±diff)_ |

Cite at least 3 specific (qid, question, gold, predicted) tuples that illustrate the patterns:

- **(qid)** _question_ → gold: _gold_, predicted: _pred_. _(commentary)_
- _(repeat)_

## 4. Production Defense

Pick **one** specific engineering action that follows from your findings. Examples (don't list all five — pick one and reason concretely):

- Confidence-threshold filter that routes below-threshold queries to humans.
- Retraining with adversarial data added to the fine-tuning set.
- Replacing the QA model with one trained for no-answer support.
- Restricting the QA model to only contexts that pass an upstream filter.
- Shrinking the production input distribution to exclude the failure pattern.

Explain in 2–3 sentences why this defense follows from your per-pattern numbers.
