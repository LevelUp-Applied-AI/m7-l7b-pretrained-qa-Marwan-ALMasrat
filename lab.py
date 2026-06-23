"""
Module 7 Week B — Applied Lab: Pre-Trained QA Evaluation on Tech/Entertainment News.

Implement the functions below. See the lab guide for full task descriptions.
"""

import json
import os
import re
import string

import pandas as pd


# -- Helpers (provided — do NOT modify) --------------------------------------

def get_data_path() -> str:
    """
    Return DATA_PATH env var if set (CI smoke fixture), otherwise the default
    path to the curated tech/entertainment QA CSV.

    Provided helper. Do not modify.
    """
    return os.environ.get("DATA_PATH", "data/tech_news_qa.csv")


def get_qa_model_name() -> str:
    """Return the QA model name. Provided helper. Do not modify."""
    return "distilbert-base-cased-distilled-squad"


def load_examples(data_path: str) -> pd.DataFrame:
    """
    Load the curated QA CSV.

    Returns a DataFrame with columns: qid, question, context, gold_answer.

    The default `data/tech_news_qa.csv` ships ~1,000 extractive QA tuples
    derived from glnmario/news-qa-summarization (CNN tech / entertainment
    slice). Each gold answer is a literal substring of its context, by
    construction (filtered during curation).

    Provided helper. Do not modify.
    """
    df = pd.read_csv(data_path)
    required = {"qid", "question", "context", "gold_answer"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{data_path} is missing required columns: {missing}")
    return df


# -- Task 1: Normalization + EM + F1 (same as drill) -------------------------

def normalize_answer(s: str) -> str:
    """SQuAD-style normalization."""
    # Step 1: Lowercase
    s = s.lower()
    # Step 2: Remove punctuation
    s = s.translate(str.maketrans("", "", string.punctuation))
    # Step 3: Strip articles (a, an, the) at word boundaries
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    # Step 4: Collapse whitespace
    s = " ".join(s.split())
    return s


def exact_match(pred: str, gold: str) -> int:
    """Return 1 if normalized prediction equals normalized gold."""
    return int(normalize_answer(pred) == normalize_answer(gold))


def token_f1(pred: str, gold: str) -> float:
    """
    Token-F1 between prediction and gold after normalization.

    Empty handling:
      - both empty -> 1.0
      - one empty  -> 0.0
    Returns float in [0.0, 1.0]; never NaN.
    """
    pred_tokens = normalize_answer(pred).split()
    gold_tokens = normalize_answer(gold).split()

    # Both empty → perfect match
    if len(pred_tokens) == 0 and len(gold_tokens) == 0:
        return 1.0
    # One empty → no overlap
    if len(pred_tokens) == 0 or len(gold_tokens) == 0:
        return 0.0

    # Multiset intersection
    pred_counts = {}
    for t in pred_tokens:
        pred_counts[t] = pred_counts.get(t, 0) + 1

    gold_counts = {}
    for t in gold_tokens:
        gold_counts[t] = gold_counts.get(t, 0) + 1

    common = sum(
        min(pred_counts.get(t, 0), gold_counts.get(t, 0))
        for t in gold_counts
    )

    if common == 0:
        return 0.0

    precision = common / len(pred_tokens)
    recall = common / len(gold_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return f1


# -- Task 2: Build the QA pipeline -------------------------------------------

def build_qa_pipeline(model_name: str):
    """Construct a Hugging Face question-answering pipeline."""
    from transformers import pipeline
    return pipeline("question-answering", model=model_name)


# -- Task 3: Predict one answer ---------------------------------------------

def predict_one(qa, question: str, context: str) -> str:
    """
    Run the QA pipeline on one (question, context) pair.

    Returns the answer STRING only (not the full pipeline output dict).
    """
    result = qa(question=question, context=context)
    return result["answer"]


# -- Task 4: Evaluate over the dataset ---------------------------------------

def evaluate_qa(qa, examples: pd.DataFrame) -> dict:
    """
    Evaluate the QA pipeline over a DataFrame of examples.

    Returns:
        {
          "em": float,   # mean EM
          "f1": float,   # mean token-F1
          "n": int,
          "predictions": [
            {qid, question, context_excerpt, gold_answer, predicted_answer, em, f1},
            ...
          ],
        }
    context_excerpt is the first 80 chars of the context (CSV-friendly).
    """
    predictions = []
    em_scores = []
    f1_scores = []

    for _, row in examples.iterrows():
        qid = str(row["qid"])
        question = str(row["question"])
        context = str(row["context"])
        gold_answer = str(row["gold_answer"])

        predicted_answer = predict_one(qa, question, context)

        em_score = exact_match(predicted_answer, gold_answer)
        f1_score = token_f1(predicted_answer, gold_answer)

        em_scores.append(em_score)
        f1_scores.append(f1_score)

        predictions.append({
            "qid": qid,
            "question": question,
            "context_excerpt": context[:80],
            "gold_answer": gold_answer,
            "predicted_answer": predicted_answer,
            "em": em_score,
            "f1": f1_score,
        })

    return {
        "em": sum(em_scores) / len(em_scores) if em_scores else 0.0,
        "f1": sum(f1_scores) / len(f1_scores) if f1_scores else 0.0,
        "n": len(predictions),
        "predictions": predictions,
    }


# -- Task 5: Orchestrate -----------------------------------------------------

def main() -> None:
    """Load data, build pipeline, evaluate, write artifacts."""
    examples = load_examples(get_data_path())
    qa = build_qa_pipeline(get_qa_model_name())
    result = evaluate_qa(qa, examples)

    # Write predictions CSV
    pred_df = pd.DataFrame(result["predictions"])
    pred_df.to_csv("qa_predictions.csv", index=False)

    # Write metrics JSON
    metrics = {
        "em": result["em"],
        "f1": result["f1"],
        "n": result["n"],
        "model": get_qa_model_name(),
    }
    with open("qa_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Aggregate EM = {result['em']:.4f}")
    print(f"Aggregate F1 = {result['f1']:.4f}")
    print(f"n = {result['n']}")


if __name__ == "__main__":
    main()