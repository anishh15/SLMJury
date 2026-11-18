# Reproducing the paper

**TL;DR.** `python experiments/run_all.py` rebuilds every table and figure from `results/`. The GPU runs below produce `results/`.

This page lists every run behind the paper's results, in the order they depend on each other, and where each output goes. The [README](../README.md#reproduce-the-paper) has the short version.

Each experiment has runs and analyses:

- **Runs** (GPU). Students answer, judges give verdicts and scores. The core runners are in `scripts/`; each experiment's own runners are `experiments/<experiment>/run_*.py`. They write raw JSON to `results/`.
- **Analyses** (CPU). `experiments/<experiment>/analyze*.py` read the raw runs and write tables to `results/analysis/<experiment>/`. `python experiments/run_all.py` runs all of them in dependency order and then draws the figures.

## Before you start

```bash
git clone https://github.com/anishh15/SLMJury.git
cd SLMJury
pip install -e ".[full]"
```

- Run everything from the repository root. Results are read from and written to `./results`, or to `$SLMJURY_RESULTS` when it is set.
- Every judge, student and reference scorer is a key in [`slmjury/configs/models.yaml`](../slmjury/configs/models.yaml), with its weights and serving settings (GPU count, memory share, batch size). `--tensor-parallel-size`, `--gpu-memory-utilization` and `--max-num-seqs` override those for your hardware; they change speed and memory use, not sampling.
- Runners with a `--force` flag skip output files that already exist, so a rerun after an interruption does only the missing files. `--force` redoes them.
- Judges marked `always_thinks` in `models.yaml` reason before every verdict, so they skip the 10-token budget on their own.
- MT-Bench reference scores come through the Together API and need `TOGETHER_API_KEY`.
- MATH answers are compared with SymPy's LaTeX parser (`sympy==1.13.1`, `antlr4-python3-runtime==4.11`), and the package ships the equivalence decision for every MATH answer in the study in `slmjury/parsers/math_labels.json`, so the analyses never depend on the parser's version. `scripts/build_math_labels.py` is the only script that writes that file. If your own runs produce MATH answers the table lacks, run it to rebuild the table from your `results/`.

## The runs, experiment by experiment

### Main results: individual judges, adversarial personas, majority voting, multi-agent debate and open-ended scoring

```bash
# students, then every small judge at both budgets
python scripts/run_student.py --student qwen2.5-32b
python scripts/run_student.py --student llama3.1-8b
python scripts/run_judge.py --judge <judge>          # once per key under judge_models
python scripts/run_evaluation.py --summary           # per-file accuracy and IFR, per-judge summaries

# personas (qwen3-4b, phi4mi-3.8b, llama3.1-8b, qwen3-14b, phi4-14b), juries, debate
python scripts/run_persona.py --judge <judge>
python -c "from slmjury.strategies.ensemble import generate_all_ensembles; generate_all_ensembles()"
python scripts/run_debate.py --configs 0,3,5,6,7,8,9                    # debate budgets: see below
python scripts/run_debate.py --configs 1,2,4 --students qwen2.5-32b --datasets gsm8k
python scripts/run_debate.py --configs 2 --students qwen2.5-32b --datasets gsm_plus
python scripts/run_debate.py --configs 1,2,4 --max-tokens 4096           # the rest of 1, 2 and 4
python scripts/run_strategy_evaluation.py            # per-file accuracy and IFR for personas, juries and debate

# open-ended scoring
python scripts/run_mtbench_student.py --student qwen2.5-32b
python scripts/run_mtbench_student.py --student llama3.1-8b
python scripts/run_mtbench_reference.py --reference gpt-oss-120b --solutions-dir results/core/mtbench_responses/
python scripts/run_mtbench_reference.py --reference qwen3.5-397b --solutions-dir results/core/mtbench_responses/
python scripts/run_scoring_judge.py --judge <judge> --datasets summeval
python scripts/run_scoring_judge.py --judge <judge> --datasets mtbench --reference-scores results/core/mtbench_references/

# analyses
python experiments/main_results/analyze.py
python experiments/main_results/debate_persuasion.py
python experiments/main_results/analyze_variance.py  # also reads the token-budget single-factor baselines
```

`run_persona.py --personas Strict Lenient --max-tokens 10` runs a subset of the personas, and `run_debate.py --configs 3` (or `--configs 0,4`) runs only those debate configurations.

**Debate budgets.** `run_debate.py` gives each agent turn 8,192 tokens unless `--max-tokens` sets another budget. Configurations 0, 3 and 5 to 9 ran at 8,192. Configurations 1, 2 and 4 ran at 4,096 (`--max-tokens 4096`), apart from four runs in Appendix J that used 8,192:

| Configuration | Agents | Runs at 8,192 |
|:--|:--|:--|
| 1 | qwen2.5-1.5b, qwen3-1.7b, phi4mi-3.8b | qwen2.5-32b on gsm8k |
| 2 | qwen3-4b, phi4mi-3.8b, qwen2.5-3b | qwen2.5-32b on gsm8k and on gsm_plus |
| 4 | llama3.2-3b, qwen3-4b, phi4mi-3.8b | qwen2.5-32b on gsm8k |

The commands above run those four first, at the default. The last debate command then skips them, because existing files are kept, and runs the rest of configurations 1, 2 and 4 at 4,096.

### Large judges, and scaling and cost

```bash
# verdicts; judges that always reason run at 8,192 tokens only
python scripts/run_judge.py --judge llama3.3-70b --output-dir results/large_judges
python scripts/run_judge.py --judge qwen2.5-72b --output-dir results/large_judges
python scripts/run_judge.py --judge qwen3.5-122b --output-dir results/large_judges
python scripts/run_judge.py --judge qwen3-32b --output-dir results/large_judges
python scripts/run_judge.py --judge gpt-oss-120b --max-tokens 8192 --output-dir results/large_judges
python scripts/run_judge.py --judge deepseek-r1-distill-llama-70b --max-tokens 8192 --output-dir results/large_judges
python scripts/run_judge.py --judge glm-4.5-air --max-tokens 8192 --output-dir results/large_judges

# open-ended scoring, for each judge key above
python scripts/run_scoring_judge.py --judge <key> --datasets summeval \
    --output-dir results/large_judges --eval-dir results/large_judges/scoring_evaluations
python scripts/run_scoring_judge.py --judge <key> --datasets mtbench --reference-scores results/core/mtbench_references/ \
    --output-dir results/large_judges --eval-dir results/large_judges/scoring_evaluations

python experiments/large_judges/analyze.py
python experiments/scaling_cost/analyze.py           # estimated serving cost per 1,000 verdicts
```

`qwen3-32b` is the one Qwen3 size between the small and the large judges (the ladder judge). Serving cost is an estimate based on an approximate decode rate for each judge's size (one rule for every judge), the GPUs in its serving configuration and one price per GPU-hour. The rule and its constants are in `experiments/scaling_cost/analyze.py`.

### Token budget and adaptive budget

```bash
JUDGES="qwen2.5-1.5b qwen2.5-3b qwen2.5-7b qwen3-0.6b qwen3-1.7b qwen3-4b qwen3-8b qwen3-14b \
        llama3.2-1b llama3.2-3b llama3.1-8b phi4-14b phi4mi-3.8b phi4mr-3.8b"

python experiments/token_budget/run_budget_sweep.py --judges $JUDGES                # budgets from 10 to 8,192 tokens
python experiments/token_budget/run_single_factor.py --judges $JUDGES --factor all  # two baselines, four single-factor changes

python experiments/token_budget/analyze.py
python experiments/adaptive_budget/analyze.py        # budget policies on the same sweep
```

The single-factor runs change one thing at a time from the quick verdict: the budget, the prompt, the thinking mode or the temperature.

### Prompt sensitivity

The variants are in [`slmjury/configs/prompt_variants.yaml`](../slmjury/configs/prompt_variants.yaml): `original`, `minimal`, `gpt_style`, `rubric_based` and `cot_style`, plus the stress variants `adversarial` and `few_shot`.

```bash
# for each small judge and each variant
python experiments/prompt_sensitivity/run_prompt_variants.py --judge <judge> --variant <variant> --max-tokens 10 8192
python experiments/prompt_sensitivity/run_prompt_variants.py --judge <judge> --variant <variant> \
    --prompt-type scoring --max-tokens 8192

python experiments/prompt_sensitivity/analyze.py
python experiments/prompt_sensitivity/analyze_open_ended.py
```

### Judging protocols: reference-free, G-Eval and pairwise

```bash
# for each small judge, and for qwen2.5-72b and gpt-oss-120b
python experiments/judging_protocols/run_reference_free.py --judge <judge> --max-tokens 10 8192
python experiments/judging_protocols/run_geval.py --judge <judge> --max-tokens 8192
python experiments/judging_protocols/run_pairwise.py --judge <judge> --max-tokens 10 8192

python experiments/judging_protocols/analyze.py
```

### Family bias: a wider pool of students

```bash
# students vLLM can serve
python experiments/family_bias/run_students.py --student deepseek-r1-0528-qwen3-8b
python experiments/family_bias/run_students.py --student mistral-7b-v0.3
python experiments/family_bias/run_students.py --student qwen3-14b-student

# newer architectures, through Transformers
python experiments/family_bias/run_students_hf.py --student gemma4-31b-student
python experiments/family_bias/run_students_hf.py --student glm-4.7-flash \
    --datasets arc_easy arc_challenge gsm8k winogrande truthfulqa
python experiments/family_bias/run_students_hf.py --student qwen3.5-35b-student \
    --datasets arc_easy arc_challenge gsm8k winogrande truthfulqa

# all 16 small judges grade five of the new students at both budgets
python scripts/run_judge.py --judge <judge> --max-tokens 10 8192 \
    --students deepseek-r1-0528-qwen3-8b mistral-7b-v0.3 gemma4-31b-student glm-4.7-flash qwen3.5-35b-student \
    --solutions-dir results/family_bias/student_solutions --output-dir results/family_bias/judgements

# the 11 judges outside the Phi-4 family also grade qwen3-14b-student
python scripts/run_judge.py --judge <judge> --max-tokens 10 8192 --students qwen3-14b-student \
    --solutions-dir results/family_bias/student_solutions --output-dir results/family_bias/judgements

python experiments/family_bias/analyze.py
python experiments/family_bias/student_ordering.py
```

The strong Qwen3 student (`qwen3-14b-student`) is graded only by the 11 judges outside the Phi-4 family: `qwen3-0.6b`, `qwen3-1.7b`, `qwen3-4b`, `qwen3-8b`, `qwen3-14b`, `qwen2.5-1.5b`, `qwen2.5-3b`, `qwen2.5-7b`, `llama3.2-1b`, `llama3.2-3b` and `llama3.1-8b`.

### Error analysis and significance

```bash
python experiments/error_analysis/analyze.py         # disagreements sorted into categories by rules
python experiments/significance/analyze.py           # paired tests, TOST, BH correction, bootstrap CIs
                                                     # (--part small-vs-large, all-pairs or open-ended runs one part)
python experiments/significance/open_ended.py        # intervals on the reported SummEval and MT-Bench correlations
python experiments/significance/supplement.py        # lineage difference-in-differences, paired steps of the Qwen3 ladder
```

The significance scripts cache aligned per-item correctness vectors in `results/cache/significance/` the first time they run.

## From the tables to the figures

```bash
python experiments/run_all.py                  # every analysis above, in dependency order, then the figures
python figures/make_figures.py --list          # the figures; pass names to draw only those
```

`make_figures.py` writes the print figures to `results/figures/` and the dark README versions of the data figures to `assets/figures/`. The overview figure (print only) needs the Source Sans Pro and Source Code Pro fonts from a TeX installation, and is skipped if they are missing.

## Checks

```bash
python scripts/build_math_labels.py --check    # the shipped MATH labels cover every MATH answer in results/
python -m pytest tests                         # unit tests
```

## Results layout

```
results/                                  git-ignored; set $SLMJURY_RESULTS to read it from elsewhere
├── core/                                 written by the scripts in scripts/
│   ├── student_solutions/<student>/<dataset>.json
│   ├── judgements/<judge>/<student>_<dataset>_t<budget>.json
│   ├── evaluations/<judge>/              per-file accuracy and IFR
│   ├── summaries/                        individual/, personas/, juries/, debate/ and student_accuracy.json
│   ├── personas/  juries/  debate/       strategy runs, each with an evaluations/ folder
│   ├── scoring/<judge>/  scoring_evaluations/<judge>/
│   ├── mtbench_responses/<student>.json
│   └── mtbench_references/<reference>/
├── large_judges/<judge>/                 verdicts and scores of the large and ladder judges, plus scoring_evaluations/
├── family_bias/                          student_solutions/<student>/ and judgements/<judge>/
├── prompt_sensitivity/<variant>/<judge>/ one folder per prompt variant
├── token_budget/                         budget_sweep/ and single_factor/
├── judging_protocols/                    referencefree/, geval/ and pairwise/
├── analysis/<experiment>/*.csv, *.json   the analysis tables, which the figures are drawn from
├── figures/                              print figures (PDF and PNG); analysis/<experiment>/ holds the analysis plots
└── cache/                                files the scripts rebuild when deleted (significance vectors, figure builds)
```
