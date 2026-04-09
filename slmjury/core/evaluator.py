"""Judge evaluator — computes accuracy, IFR, and disagreement metrics.

Evaluates judge verdicts against ground truth using dataset-appropriate
answer comparison. Supports per-file evaluation and cross-file aggregation.
"""

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

from slmjury.parsers.answer import extract_answer, get_dataset_type
from slmjury.parsers.normalizer import answers_are_equivalent

logger = logging.getLogger(__name__)


class JudgeEvaluator:
    """Evaluates judge model performance against ground truth.

    Computes:
    - Accuracy: % of verdicts matching ground truth.
    - IFR (Instruction Following Rate): % of valid verdicts.
    - Disagreements: Cases where judge verdict != expected.

    Args:
        judge_model: Judge model key.
        student_model: Student model key.
        dataset: Dataset name.
        max_tokens: Token setting used.
        judgements: List of judgement dicts.
    """

    def __init__(
        self,
        judge_model: str,
        student_model: str,
        dataset: str,
        max_tokens: int,
        judgements: list[dict],
    ):
        self.judge_model = judge_model
        self.student_model = student_model
        self.dataset = dataset
        self.dataset_type = get_dataset_type(dataset)
        self.max_tokens = max_tokens
        self.judgements = judgements

        self.accuracy = 0.0
        self.ifr = 0.0
        self.disagreement_rate = 0.0
        self.disagreements: list[dict] = []
        self.summary: dict = {}

    def evaluate(self) -> dict:
        """Run evaluation and compute all metrics.

        Returns:
            Summary dict with accuracy, IFR, counts, and timestamp.
        """
        logger.info(
            "Evaluating %s vs %s on %s (tokens=%d, n=%d)",
            self.judge_model, self.student_model,
            self.dataset, self.max_tokens, len(self.judgements),
        )

        self._compute_metrics()
        self._build_summary()
        self._print_summary()

        logger.info(
            "Results: accuracy=%.2f%%, IFR=%.2f%%, disagreements=%d/%d",
            self.accuracy * 100, self.ifr * 100,
            len(self.disagreements), len(self.judgements),
        )
        return self.summary

    def _compute_metrics(self):
        """Compute accuracy, IFR, and collect disagreements in one pass."""
        correct_count = 0
        valid_count = 0
        self.disagreements = []

        for item in self.judgements:
            expected, student_answer = self._get_expected_verdict(item)
            judge_verdict = item.get("judgement", "Undefined").lower()

            if judge_verdict in ("correct", "incorrect"):
                valid_count += 1
                if judge_verdict == expected:
                    correct_count += 1
                else:
                    self.disagreements.append({
                        "problem_id": item.get("problem_id"),
                        "question": item.get("question"),
                        "ground_truth_reasoning": item.get("ground_truth_reasoning"),
                        "ground_truth_answer": item.get("ground_truth_answer"),
                        "student_reasoning": item.get("student_reasoning"),
                        "student_answer": student_answer,
                        "expected_verdict": expected,
                        "judge_verdict": judge_verdict,
                        "judge_response": item.get("judge_response"),
                    })

        n = len(self.judgements)
        self.accuracy = correct_count / n if n > 0 else 0.0
        self.ifr = valid_count / n if n > 0 else 0.0
        self.disagreement_rate = len(self.disagreements) / n if n > 0 else 0.0

    def _get_expected_verdict(self, item: dict) -> tuple[str, str]:
        """Determine expected verdict using dataset-appropriate comparison."""
        student_answer = item.get("student_answer", "")

        if not student_answer or student_answer == "UNABLE_TO_EXTRACT":
            student_answer = extract_answer(
                item.get("student_reasoning", ""), self.dataset_type,
            )

        ground_truth = item.get("ground_truth_answer", "")

        if answers_are_equivalent(student_answer, ground_truth, self.dataset_type):
            return "correct", student_answer
        return "incorrect", student_answer

    def _build_summary(self):
        """Build summary dict with all metrics."""
        self.summary = {
            "judge_model": self.judge_model,
            "max_tokens": self.max_tokens,
            "total_problems": len(self.judgements),
            "metrics": {
                "accuracy": round(self.accuracy, 4),
                "instruction_following_rate": round(self.ifr, 4),
                "disagreement_rate": round(self.disagreement_rate, 4),
            },
            "counts": {
                "correct_verdicts": int(self.accuracy * len(self.judgements)),
                "valid_verdicts": int(self.ifr * len(self.judgements)),
                "disagreements": len(self.disagreements),
            },
            "timestamp": datetime.now().isoformat(),
        }

    def _print_summary(self):
        """Print formatted evaluation summary to console."""
        print(f"\n\U0001f4ca Results for {self.judge_model} (tokens={self.max_tokens}):")
        print(f"   Dataset: {self.dataset} | Student: {self.student_model}")
        print(f"   Accuracy:      {self.accuracy:.2%}")
        print(f"   IFR:           {self.ifr:.2%}")
        print(f"   Disagreements: {len(self.disagreements)}/{len(self.judgements)}")

        if self.disagreements:
            print("\n   Sample disagreements (first 3):")
            for d in self.disagreements[:3]:
                print(
                    f"   - Problem {d['problem_id']}: "
                    f"expected={d['expected_verdict']}, got={d['judge_verdict']}"
                )

    def save_results(self, output_dir: Optional[Path] = None) -> Path:
        """Save evaluation results with disagreement details.

        Args:
            output_dir: Base directory for saving. Defaults to results/evaluations.

        Returns:
            Path to the saved JSON file.
        """
        base = output_dir or Path("results/evaluations")
        out_dir = base / self.judge_model
        out_dir.mkdir(parents=True, exist_ok=True)

        filename = f"{self.student_model}_{self.dataset}_t{self.max_tokens}.json"
        out_file = out_dir / filename

        full_results = {
            **self.summary,
            "student_model": self.student_model,
            "dataset": self.dataset,
            "dataset_type": self.dataset_type,
            "disagreements": self.disagreements,
        }

        with open(out_file, "w") as f:
            json.dump(full_results, f, indent=2)

        logger.info("Saved evaluation to %s", out_file)
        return out_file


def generate_judge_summary(
    judge_model: str,
    max_tokens: int,
    eval_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
) -> Optional[dict]:
    """Aggregate all evaluation results for a judge model + token setting.

    Args:
        judge_model: Judge model key.
        max_tokens: Token setting to aggregate.
        eval_dir: Directory containing per-file evaluation results.
        output_dir: Directory for saving the summary JSON.

    Returns:
        Summary dict with overall and per-dataset/student breakdowns,
        or None if no files found.
    """
    base_eval = eval_dir or Path("results/evaluations")
    model_dir = base_eval / judge_model

    pattern = f"*_t{max_tokens}.json"
    eval_files = list(model_dir.glob(pattern))

    if not eval_files:
        logger.warning(
            "No evaluation files for %s with tokens=%d", judge_model, max_tokens,
        )
        return None

    total_problems = 0
    total_correct = 0
    total_valid = 0
    by_dataset: dict[str, dict] = {}
    by_student: dict[str, dict] = {}

    for eval_file in eval_files:
        with open(eval_file) as f:
            data = json.load(f)

        dataset = data.get("dataset", "unknown")
        student = data.get("student_model", "unknown")
        n = data["total_problems"]
        correct = data["counts"]["correct_verdicts"]
        valid = data["counts"]["valid_verdicts"]

        total_problems += n
        total_correct += correct
        total_valid += valid

        for key, bucket in [(dataset, by_dataset), (student, by_student)]:
            if key not in bucket:
                bucket[key] = {"total": 0, "correct": 0, "valid": 0}
            bucket[key]["total"] += n
            bucket[key]["correct"] += correct
            bucket[key]["valid"] += valid

    # Compute rates
    for d in list(by_dataset.values()) + list(by_student.values()):
        d["accuracy"] = round(d["correct"] / d["total"], 4) if d["total"] > 0 else 0
        d["ifr"] = round(d["valid"] / d["total"], 4) if d["total"] > 0 else 0

    summary = {
        "judge_model": judge_model,
        "max_tokens": max_tokens,
        "total_problems": total_problems,
        "total_correct": total_correct,
        "total_valid": total_valid,
        "overall_accuracy": round(total_correct / total_problems, 4) if total_problems else 0,
        "overall_ifr": round(total_valid / total_problems, 4) if total_problems else 0,
        "by_dataset": by_dataset,
        "by_student_model": by_student,
        "num_files_aggregated": len(eval_files),
        "timestamp": datetime.now().isoformat(),
    }

    # Save summary
    summaries_dir = output_dir or Path("results/summaries/individual")
    summaries_dir.mkdir(parents=True, exist_ok=True)
    summary_file = summaries_dir / f"{judge_model}_t{max_tokens}.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("Summary saved to %s", summary_file)
    return summary


def parse_judgement_filename(filename: str) -> tuple[str, str, int]:
    """Extract components from a judgement filename.

    Expected format: {student}_{dataset}_t{n}.json

    Args:
        filename: Judgement filename or path.

    Returns:
        Tuple of (student_model, dataset, max_tokens).

    Raises:
        ValueError: If filename doesn't match expected format.
    """
    name = Path(filename).stem
    match = re.match(
        r"^(.+?)_(gsm8k|gsm_plus|math|arc_easy|arc_challenge|hellaswag|winogrande|truthfulqa)_t(\d+)$",
        name,
    )
    if match is None:
        raise ValueError(f"Cannot parse judgement filename: {filename}")
    return match.group(1), match.group(2), int(match.group(3))


def list_judge_models(
    judgements_dir: Path = Path("results/judgements"),
) -> list[str]:
    """Get list of all judge models with judgement files.

    Args:
        judgements_dir: Base directory containing per-model subdirectories.

    Returns:
        List of judge model directory names.
    """
    if not judgements_dir.exists():
        return []
    return [d.name for d in sorted(judgements_dir.iterdir()) if d.is_dir()]


def evaluate_judge_with_tokens(
    judge_model: str,
    max_tokens: int,
    judgements_dir: Path = Path("results/judgements"),
    output_dir: Path = Path("results/evaluations"),
):
    """Evaluate all judgement files for a judge/token combination.

    Args:
        judge_model: Judge model key.
        max_tokens: Token setting to filter files.
        judgements_dir: Directory containing per-model judgement subdirectories.
        output_dir: Base directory for saving evaluation results.
    """
    model_dir = judgements_dir / judge_model

    if not model_dir.exists():
        logger.warning("Judgement directory not found: %s", model_dir)
        return

    pattern = f"*_t{max_tokens}.json"
    judgement_files = list(model_dir.glob(pattern))

    if not judgement_files:
        logger.warning(
            "No judgement files for %s with tokens=%d", judge_model, max_tokens,
        )
        return

    for judgement_file in judgement_files:
        student, dataset, tokens = parse_judgement_filename(
            judgement_file.name,
        )

        with open(judgement_file) as f:
            judgements = json.load(f)

        evaluator = JudgeEvaluator(
            judge_model=judge_model,
            student_model=student,
            dataset=dataset,
            max_tokens=tokens,
            judgements=judgements,
        )
        evaluator.evaluate()
        evaluator.save_results(output_dir=output_dir)
