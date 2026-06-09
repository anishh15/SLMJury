<p align="center">
  <img src="assets/banner.svg" alt="SLMJury Banner" width="100%">
</p>

<div align="center">

<!-- Badges -->
<a href="https://pypi.org/project/slmjury/"><img src="https://img.shields.io/pypi/v/slmjury?style=for-the-badge&logo=pypi&logoColor=white&label=PyPI" alt="PyPI"/></a>
<a href="https://arxiv.org/abs/2606.07810"><img src="https://img.shields.io/badge/📄_Paper-ArXiv%3A2606.07810-red?style=for-the-badge&logo=arxiv" alt="Paper"/></a>
<a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10+-blue.svg?style=for-the-badge&logo=python&logoColor=white" alt="Python"/></a>
<a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-green.svg?style=for-the-badge" alt="License"/></a>
<a href="https://github.com/anishh15/SLMJury/stargazers"><img src="https://img.shields.io/github/stars/anishh15/SLMJury?style=for-the-badge&logo=github&color=yellow" alt="Stars"/></a>

*Can Small Language Models Judge as Well as Large Language Models?*

**🧑‍⚖️ 16 SLM Judges &bull; 📊 10 Datasets &bull; 🗳️ 3 Advanced Strategies &bull; 🎭 6 Persona Prompts**

[**🏆 Leaderboard**](https://anishh15.github.io/SLMJury/) | [**📖 Read Paper**](https://arxiv.org/abs/2606.07810) | [**🚀 Get Started**](#-installation)

</div>

---

## 📢 Latest News

| Date | Update |
|------|--------|
| **Jun 2026** | Paper submitted: [arXiv:2606.07810](https://arxiv.org/abs/2606.07810) |
| **May 2026** | Interactive leaderboard website launched: [anishh15.github.io/SLMJury](https://anishh15.github.io/SLMJury/) |
| **May 2026** | v0.1.0 released on [PyPI](https://pypi.org/project/slmjury/) -- first public release |

---

## 💡 What is SLMJury?

SLMJury is a comprehensive framework that investigates whether **Small Language Models (0.6B-14B parameters)** can serve as reliable judges across both **closed-ended** (accuracy-based) and **open-ended** (correlation-based) evaluation paradigms. The project explores six evaluation modes: individual judging, persona-based evaluation, majority-vote ensembles, multi-agent debate, human agreement scoring (SummEval), and LLM agreement scoring (MT-Bench).

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

### 📦 From PyPI

```bash
pip install slmjury
```

Optional extras:
```bash
pip install slmjury[vllm]       # GPU inference with vLLM
pip install slmjury[together]   # Together API for oracle scoring
pip install slmjury[full]       # Everything (vllm + together + dev tools)
```

### 🔧 From Source (Development)

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
# Majority voting on individual verdicts
from slmjury.strategies.ensemble import majority_vote
verdict = majority_vote(["Correct", "Incorrect", "Correct"])  # → "Correct"

# Generate all C(5,3)=10 ensemble combinations from pre-computed judgements
from slmjury.strategies.ensemble import generate_all_ensembles
generate_all_ensembles(
    judgements_dir="results/judgements",
    output_dir="results/majority_voting",
)

# Multi-agent debate (3 judges, RCR prompting)
from slmjury.strategies.debate import run_debate
run_debate(
    combo_models=["qwen3-4b", "phi4mi-3.8b", "qwen2.5-3b"],
    combo_temps=[0, 0, 0],
    student_results=results,
    dataset_name="gsm8k",
)

# Persona effects
from slmjury.strategies.persona import run_persona_evaluation, get_personas
personas = get_personas()
run_persona_evaluation(
    judge_model_key="qwen3-4b",
    student_results=results,
    student_model="qwen2.5-32b",
    dataset="gsm8k",
    max_tokens=10,
    persona_name="strict",
    persona_prompt=personas["strict"],
)
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
<tr><td><b>Qwen 2.5</b></td><td>1.5B, 3B, 7B</td><td align="center">1.5B - 7B</td><td align="center">-</td></tr>
<tr><td><b>Qwen 3</b></td><td>0.6B, 1.7B, 4B, 8B, 14B</td><td align="center">0.6B - 14B</td><td align="center">✅</td></tr>
<tr><td><b>Llama 3.x</b></td><td>3.2-1B, 3.2-3B, 3.1-8B</td><td align="center">1B - 8B</td><td align="center">-</td></tr>
<tr><td><b>Phi-4</b></td><td>14B, Reasoning, R-Plus, Mini, Mini-Reasoning</td><td align="center">3.8B - 14B</td><td align="center">✅*</td></tr>
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

**Open-ended** (scoring: 1-5):

| Dataset | Type | Turns | Size | Oracle |
|---------|------|-------|------|--------|
| **SummEval** | Summarization | - | 1,600 pairs | Human annotations |
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
├── pyproject.toml            # Package config (pip install slmjury)
└── README.md
```

---

## 📊 Results

### 🏆 Leaderboard (Top Judges - Closed-Ended)

<table>
<thead>
<tr>
<th align="center">🏅 Rank</th>
<th align="left">🤖 Judge Model</th>
<th align="center">Params</th>
<th align="center">Max Tokens</th>
<th align="center">📊 Accuracy</th>
<th align="center">🎯 IFR</th>
</tr>
</thead>
<tbody>
<tr><td align="center">🥇</td><td><strong>Phi-4</strong></td><td align="center">14B</td><td align="center">10</td><td align="center"><strong>89.55%</strong></td><td align="center">99.98%</td></tr>
<tr><td align="center">🥈</td><td><strong>Qwen3-14B</strong></td><td align="center">14B</td><td align="center">10</td><td align="center"><strong>89.51%</strong></td><td align="center">100.0%</td></tr>
<tr><td align="center">🥉</td><td><strong>Qwen3-8B</strong></td><td align="center">8B</td><td align="center">10</td><td align="center"><strong>88.96%</strong></td><td align="center">100.0%</td></tr>
<tr><td align="center">4</td><td><strong>Phi-4-Reasoning-Plus</strong></td><td align="center">14B</td><td align="center">8192</td><td align="center"><strong>88.75%</strong></td><td align="center">100.0%</td></tr>
<tr><td align="center">5</td><td><strong>Phi-4-Reasoning</strong></td><td align="center">14B</td><td align="center">8192</td><td align="center"><strong>88.24%</strong></td><td align="center">100.0%</td></tr>
</tbody>
</table>

<sub>Top judges ranked by overall accuracy across 8 closed-ended benchmarks (N=64,824 judgments per configuration). Full results for all 16 judges available on the [leaderboard](https://anishh15.github.io/SLMJury/) and in the [paper](https://arxiv.org/abs/2606.07810).</sub>

Explore full results on the interactive leaderboard:

<div align="center">
<a href="https://anishh15.github.io/SLMJury/">
<img src="https://img.shields.io/badge/🏆_Explore_Leaderboard-Visit_Now-brightgreen?style=for-the-badge&logo=rocket" alt="Visit Leaderboard">
</a>
</div>

### 🔍 Key Findings

- **Overthinking is domain-dependent**: Quick 10-token verdicts match or beat extended reasoning on math judging, while reasoning wins on general tasks by up to 23%
- **Domain generalization separates families**: Math-to-general accuracy gaps range from under 10% to nearly 40% across model families
- **Closed vs. open-ended judging differ**: The best binary judge (Phi-4) drops to rank 9 on MT-Bench; reasoning-trained models invert this ordering
- **Multi-agent debate degrades accuracy**: Under the RCR protocol, debate hurts performance across all tested configurations, while top judges resist six adversarial personas with ≤0.55% variance

---

## 📖 Citation

If you use SLMJury in your research, please cite:

```bibtex
@misc{laddha2026slmjurysmalllanguagemodels,
      title={SLMJury: Can Small Language Models Judge as Well as Large Ones?},
      author={Anish Laddha and Nitesh Pradhan and Gaurav Srivastava},
      year={2026},
      eprint={2606.07810},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2606.07810},
}
```

---

## 🤝 Contributing

We welcome contributions! Here's how to get started:

```bash
git clone https://github.com/anishh15/SLMJury.git
cd SLMJury
pip install -e ".[dev]"
pytest tests/ -v
```

### 🛠️ Ways to Contribute
- **🐛 Bug Reports**: Found an issue? [Report it here](https://github.com/anishh15/SLMJury/issues)
- **✨ Feature Requests**: Have ideas? [Share them here](https://github.com/anishh15/SLMJury/issues)
- **🔧 Code Contributions**: Submit PRs for improvements
- **📚 Documentation**: Help improve our docs
- **🤖 Model Submissions**: Suggest new judge models for evaluation

---

## 📞 Contact & Support

- **📧 Email**: [anshladdha15@gmail.com](mailto:anshladdha15@gmail.com), [nitesh.pradhan@lnmiit.ac.in](mailto:nitesh.pradhan@lnmiit.ac.in), [gks@vt.edu](mailto:gks@vt.edu)
- **🐛 Issues**: [GitHub Issues](https://github.com/anishh15/SLMJury/issues)

---

## 📄 License

Apache License 2.0 -- see [LICENSE](LICENSE) for details.

---

<div align="center">

<a href="#-installation"><img src="https://img.shields.io/badge/🚀_Get_Started-FF6B6B?style=for-the-badge" alt="Get Started"/></a>
<a href="https://anishh15.github.io/SLMJury/"><img src="https://img.shields.io/badge/🏆_Leaderboard-4ECDC4?style=for-the-badge" alt="Leaderboard"/></a>
<a href="https://arxiv.org/abs/2606.07810"><img src="https://img.shields.io/badge/📖_Read_Paper-red?style=for-the-badge" alt="Paper"/></a>
<a href="https://github.com/anishh15/SLMJury"><img src="https://img.shields.io/badge/⭐_Star_on_GitHub-yellow?style=for-the-badge" alt="GitHub"/></a>

**Made with ❤️ by [Anish Laddha](https://github.com/anishh15), [Nitesh Pradhan](https://github.com), and [Gaurav Srivastava](https://github.com/ctrl-gaurav)**

</div>

<p align="center">
  <a href="https://github.com/anishh15/SLMJury">
    <img src="assets/logo.svg" alt="SLMJury Logo" width="100">
  </a>
</p>
