<p align="center">
  <img src="assets/banner.svg" alt="SLMJury Banner" width="100%">
</p>

<div align="center">

<!-- Badges -->
<a href="#-installation"><img src="https://img.shields.io/badge/📦_PyPI-Coming_Soon-lightgrey?style=for-the-badge&logo=pypi&logoColor=white" alt="PyPI"/></a>
<a href="#-citation"><img src="https://img.shields.io/badge/📄_Paper-Coming_Soon-lightgrey?style=for-the-badge&logo=arxiv" alt="Paper"/></a>
<a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10+-blue.svg?style=for-the-badge&logo=python&logoColor=white" alt="Python"/></a>
<a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-green.svg?style=for-the-badge" alt="License"/></a>
<a href="https://github.com/anishh15/SLMJury/stargazers"><img src="https://img.shields.io/github/stars/anishh15/SLMJury?style=for-the-badge&logo=github&color=yellow" alt="Stars"/></a>

*Can Small Language Models Judge as Well as Large Language Models?*

**🧑‍⚖️ 16 SLM Judges &bull; 📊 10 Datasets &bull; 🗳️ 3 Advanced Strategies &bull; 🎭 6 Persona Prompts**

[**🏆 Leaderboard**](https://anishh15.github.io/SLMJury/) | [**🚀 Get Started**](#-installation)

</div>

---

## 💡 What is SLMJury?

SLMJury is a comprehensive framework that investigates whether **Small Language Models (0.6B–14B parameters)** can serve as reliable judges across both **closed-ended** (accuracy-based) and **open-ended** (correlation-based) evaluation paradigms. The project explores six evaluation modes: individual judging, persona-based evaluation, majority-vote ensembles, multi-agent debate, human agreement scoring (SummEval), and LLM agreement scoring (MT-Bench).

<div align="center">
<a href="https://anishh15.github.io/SLMJury/">
<img src="https://img.shields.io/badge/🎯_Visit_Leaderboard-Live_Demo-brightgreen?style=for-the-badge&logo=rocket" alt="Visit Leaderboard">
</a>
</div>

### 🌟 Key Highlights

<table>
<tr>
<td width="33%">

#### 🧠 **Individual Judging**
- 16 SLM judges from 4 model families
- Quick verdict vs. reasoned response
- Accuracy & Instruction Following Rate

</td>
<td width="33%">

#### 🗳️ **Majority Voting**
- C(5,3) ensemble combinations
- Top-5 best individual judges
- Boosted accuracy via consensus

</td>
<td width="33%">

#### 🤝 **Multi-Agent Debate**
- RCR (Reflect-Critique-Refine) prompting
- Cross-architecture & same-model variants
- Up to 5 rounds with consensus fallback

</td>
</tr>
</table>

---

## ⚡ Installation

### 🔧 From Source (Recommended)

```bash
git clone https://github.com/anishh15/SLMJury.git
cd SLMJury
pip install -e .
```

---

## 🚀 Quick Start

### 💻 CLI Scripts

```bash
# Step 1: Run student model inference
python scripts/run_student.py --model qwen2.5-32b --datasets gsm8k math

# Step 2: Run judge evaluations
python scripts/run_judge.py --judge qwen3-4b --max-tokens 10 8192

# Step 3: Evaluate all judgements and generate summaries
python scripts/run_evaluation.py
```

### 🐍 Python API

```python
from slmjury.core.solver import StudentSolver
from slmjury.core.judge import JudgeModel
from slmjury.core.evaluator import JudgeEvaluator

# Step 1: Solve problems with a student model
solver = StudentSolver("qwen2.5-32b")
results = solver.solve_batch(problems, "gsm8k")
solver.save_results(results, "gsm8k")
solver.cleanup()

# Step 2: Judge the solutions
judge = JudgeModel("qwen3-4b")
judgements = judge.evaluate_batch(results, max_tokens=10)
judge.save_results(judgements, "qwen2.5-32b", "gsm8k", 10)
judge.cleanup()

# Step 3: Evaluate judge accuracy
evaluator = JudgeEvaluator("qwen3-4b", "qwen2.5-32b", "gsm8k", 10, judgements)
summary = evaluator.evaluate()
```

<details>
<summary><b>🧩 Advanced: Multi-Agent Strategies</b></summary>

```python
# Majority voting ensemble
from slmjury.strategies.ensemble import run_majority_voting
run_majority_voting(
    judge_keys=["qwen3-4b", "phi4mi-3.8b", "qwen2.5-3b"],
    student_results=results,
    max_tokens=10,
)

# Multi-agent debate (3 judges, RCR prompting)
from slmjury.strategies.debate import run_debate
run_debate(
    combo_models=["qwen3-4b", "phi4mi-3.8b", "qwen2.5-3b"],
    combo_temps=[0, 0, 0],
    student_results=results,
    dataset_name="gsm8k",
)

# Persona effects (6 system prompts × all judges)
from slmjury.strategies.persona import run_persona_evaluation
run_persona_evaluation("qwen3-4b", results, max_tokens=10)
```

</details>

<details>
<summary><b>🔬 Open-Ended Scoring (SummEval / MT-Bench)</b></summary>

```bash
# Score SummEval with a single judge
python scripts/run_scoring_judge.py \
  --judge qwen3-4b --dataset summeval

# Score MT-Bench with a single judge
python scripts/run_scoring_judge.py \
  --judge qwen3-4b --dataset mtbench \
  --oracle-scores results/mtbench_oracle/
```

```python
from slmjury.core.scoring_judge import ScoringJudge

judge = ScoringJudge("qwen3-4b", output_dir="results/scoring")

# Score SummEval (4-dimension scoring)
summeval_data = load_dataset("summeval")
results = judge.score_summeval(summeval_data, max_tokens=8192)
judge.save_results(results, "summeval")
judge.cleanup()
```

</details>

---

## 🤖 Supported Models

<table>
<thead>
<tr>
<th align="left">Family</th>
<th align="left">Models</th>
<th align="center">Parameters</th>
<th align="center">Thinking</th>
</tr>
</thead>
<tbody>
<tr><td><b>Qwen 2.5</b></td><td>1.5B, 3B, 7B</td><td align="center">1.5B – 7B</td><td align="center">—</td></tr>
<tr><td><b>Qwen 3</b></td><td>0.6B, 1.7B, 4B, 8B, 14B</td><td align="center">0.6B – 14B</td><td align="center">✅</td></tr>
<tr><td><b>Llama 3.x</b></td><td>3.2-1B, 3.2-3B, 3.1-8B</td><td align="center">1B – 8B</td><td align="center">—</td></tr>
<tr><td><b>Phi-4</b></td><td>14B, Reasoning, R-Plus, Mini, Mini-Reasoning</td><td align="center">3.8B – 14B</td><td align="center">✅*</td></tr>
</tbody>
</table>

<sub>*Phi-4 Reasoning/Plus/Mini-Reasoning always use thinking mode and skip quick verdict (t=10) evaluation.</sub>

### 📊 Datasets

**Closed-ended** (verdict: Correct/Incorrect):

| Dataset | Type | Domain | Size |
|---------|------|--------|------|
| **GSM8K** | Numeric | Math | 1,319 |
| **GSM-Plus** | Numeric | Math | 10,552 |
| **MATH** | LaTeX | Math | 5,000 |
| **ARC-Easy** | Multiple Choice | Science | 2,376 |
| **ARC-Challenge** | Multiple Choice | Science | 1,172 |
| **HellaSwag** | Multiple Choice | General | 10,042 |
| **WinoGrande** | Multiple Choice | General | 1,267 |
| **TruthfulQA** | Multiple Choice | General | 684 |

**Open-ended** (scoring: 1–5):

| Dataset | Type | Turns | Size | Oracle |
|---------|------|-------|------|--------|
| **SummEval** | Summarization | — | 1,600 pairs | Human annotations |
| **MT-Bench** | Multi-turn chat | 2 | 80 questions | GPT-OSS-120B, Qwen3.5-397B (Together API) |

---

## 🏗️ Project Structure

```
SLMJury/
├── slmjury/                  # Python package
│   ├── configs/              # Centralized YAML model configurations
│   ├── data/                 # Dataset loaders (HuggingFace → local JSON)
│   ├── parsers/              # Answer extraction, normalization, verdict/score parsing
│   ├── core/                 # Pipeline: solver → judge → evaluator + scoring
│   └── strategies/           # Ensemble voting, multi-agent debate, personas
├── scripts/                  # CLI entry-points (student, judge, oracle, scoring)
├── bash/                     # Bash wrappers for full experiment runs
├── tests/                    # Unit & integration tests (pytest)
├── website/                  # React leaderboard (Vite + Tailwind)
├── assets/                   # SVG banner and logo
├── pyproject.toml            # Package config (pip install -e .)
└── README.md
```

---

## 🏆 Leaderboard

Explore full results on the interactive leaderboard:

<div align="center">
<a href="https://anishh15.github.io/SLMJury/">
<img src="https://img.shields.io/badge/🏆_Explore_Leaderboard-Visit_Now-brightgreen?style=for-the-badge&logo=rocket" alt="Visit Leaderboard">
</a>
</div>

---

## 📖 Citation

If you use SLMJury in your research, please cite:

```bibtex
@misc{laddha2026slmjury,
      title={SLMJury: Can Small Language Models Judge as Well as Large Language Models?},
      author={Anish Laddha and Nitesh Pradhan and Gaurav Srivastava},
      year={2026},
}
```

---

## 📄 License

Apache License 2.0 — see [LICENSE](LICENSE) for details.

---

<div align="center">

<a href="#-installation"><img src="https://img.shields.io/badge/🚀_Get_Started-FF6B6B?style=for-the-badge" alt="Get Started"/></a>
<a href="https://anishh15.github.io/SLMJury/"><img src="https://img.shields.io/badge/🏆_Leaderboard-4ECDC4?style=for-the-badge" alt="Leaderboard"/></a>
<a href="https://github.com/anishh15/SLMJury"><img src="https://img.shields.io/badge/⭐_Star_on_GitHub-yellow?style=for-the-badge" alt="GitHub"/></a>

**Made with ❤️ by [Anish Laddha](https://github.com/anishh15), [Nitesh Pradhan](https://github.com), and [Gaurav Srivastava](https://github.com/ctrl-gaurav)**

</div>

<p align="center">
  <a href="https://github.com/anishh15/SLMJury">
    <img src="assets/logo.svg" alt="SLMJury Logo" width="100">
  </a>
</p>
