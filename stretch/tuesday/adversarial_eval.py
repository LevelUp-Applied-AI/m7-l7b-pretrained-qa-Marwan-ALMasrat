"""
Module 7 Week B — Tuesday Stretch (Honors): Adversarial QA Probe.

Reuses the QA pipeline + EM/F1 functions from `lab.py`. Implement the TODO
functions below; see the stretch page for full task description.
"""

import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import lab  # noqa: E402

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def load_adversarial_set(path: str = os.path.join(BASE_DIR, "adversarial_set.csv")) -> pd.DataFrame:
    required_columns = {"qid", "question", "context", "gold_answer", "pattern_tag"}
    df = pd.read_csv(path)
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"adversarial_set.csv is missing required columns: {missing}")

    bad = df[~df.apply(lambda r: r["gold_answer"] in r["context"], axis=1)]
    if not bad.empty:
        raise ValueError(f"gold_answer not in context for qids: {bad['qid'].tolist()}")

    return df


def evaluate_adversarial(qa, df: pd.DataFrame) -> dict:
    tag_lookup = dict(zip(df["qid"], df["pattern_tag"]))
    aggregate = lab.evaluate_qa(qa, df)
    for pred in aggregate["predictions"]:
        pred["pattern_tag"] = tag_lookup.get(pred["qid"], "unknown")
    per_pattern = {}
    for pred in aggregate["predictions"]:
        tag = pred["pattern_tag"]
        if tag not in per_pattern:
            per_pattern[tag] = {"em_sum": 0.0, "f1_sum": 0.0, "n": 0}
        per_pattern[tag]["em_sum"] += pred["em"]
        per_pattern[tag]["f1_sum"] += pred["f1"]
        per_pattern[tag]["n"]      += 1
    per_pattern_final = {
        tag: {
            "em": round(vals["em_sum"] / vals["n"], 4),
            "f1": round(vals["f1_sum"] / vals["n"], 4),
            "n":  vals["n"],
        }
        for tag, vals in per_pattern.items()
    }
    return {
        "em":          aggregate["em"],
        "f1":          aggregate["f1"],
        "n":           aggregate["n"],
        "per_pattern": per_pattern_final,
        "predictions": aggregate["predictions"],
    }


def main() -> None:
    df = load_adversarial_set()
    qa = lab.build_qa_pipeline(lab.get_qa_model_name())
    result = evaluate_adversarial(qa, df)

    pred_df = pd.DataFrame(result["predictions"])
    pred_df.to_csv(os.path.join(BASE_DIR, "adversarial_predictions.csv"), index=False)

    metrics = {
        "em":          result["em"],
        "f1":          result["f1"],
        "n":           result["n"],
        "per_pattern": result["per_pattern"],
        "model":       lab.get_qa_model_name(),
    }
    with open(os.path.join(BASE_DIR, "adversarial_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Aggregate EM = {result['em']:.4f}")
    print(f"Aggregate F1 = {result['f1']:.4f}")
    print(f"n = {result['n']}")
    print(f"Per-pattern: {list(result['per_pattern'].keys())}")


if __name__ == "__main__":
    main()