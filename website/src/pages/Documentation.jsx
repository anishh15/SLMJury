import { useState } from 'react'
import { BookOpen, Terminal, Code, Settings, Layers, Cpu, Package, FileText, Hexagon, Copy, Check, Zap, Database, GitBranch, Shield, BarChart3, Wrench, AlertTriangle, Lightbulb, Users, MessageSquare, UserCheck } from 'lucide-react'
import { useTheme } from '../context/ThemeContext'

function CodeBlock({ code, language = 'bash' }) {
  const [copied, setCopied] = useState(false)
  const { isDark } = useTheme()
  function handleCopy() {
    navigator.clipboard.writeText(code)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }
  return (
    <div className="relative group my-3">
      <div className={`flex items-center justify-between px-4 py-2 rounded-t-lg border border-b-0 ${
        isDark ? 'bg-bb-dark-600/80 border-bb-dark-50/20' : 'bg-gray-100 border-gray-200'
      }`}>
        <span className={`text-[10px] uppercase tracking-wider font-mono ${isDark ? 'text-gray-600' : 'text-gray-400'}`}>{language}</span>
        <button onClick={handleCopy} className={`transition-colors ${isDark ? 'text-gray-600 hover:text-bb-accent' : 'text-gray-400 hover:text-bb-accent-dark'}`}>
          {copied ? <Check className={`w-3.5 h-3.5 ${isDark ? 'text-bb-accent' : 'text-bb-accent-dark'}`} /> : <Copy className="w-3.5 h-3.5" />}
        </button>
      </div>
      <pre className={`rounded-b-lg p-4 overflow-x-auto text-sm font-mono leading-relaxed border ${
        isDark ? 'bg-bb-dark-500/80 border-bb-dark-50/20 text-gray-300' : 'bg-gray-50 border-gray-200 text-gray-700'
      }`}>
        <code>{code}</code>
      </pre>
    </div>
  )
}

function Section({ id, icon: Icon, title, children }) {
  const { isDark } = useTheme()
  return (
    <section id={id} className="mb-12 scroll-mt-24">
      <div className="flex items-center gap-3 mb-4">
        <div className={`w-8 h-8 rounded-lg flex items-center justify-center ${isDark ? 'bg-bb-accent/10' : 'bg-bb-accent-dark/10'}`}>
          <Icon className={`w-4 h-4 ${isDark ? 'text-bb-accent' : 'text-bb-accent-dark'}`} />
        </div>
        <h2 className={`text-xl font-bold ${isDark ? 'text-white' : 'text-gray-900'}`}>{title}</h2>
      </div>
      <div className={`leading-relaxed text-sm space-y-4 ${isDark ? 'text-gray-400' : 'text-gray-600'}`}>{children}</div>
    </section>
  )
}

function SubSection({ title, children }) {
  const { isDark } = useTheme()
  return (
    <div className="mt-6">
      <h3 className={`text-base font-semibold mb-3 ${isDark ? 'text-gray-200' : 'text-gray-800'}`}>{title}</h3>
      <div className={`text-sm space-y-3 ${isDark ? 'text-gray-400' : 'text-gray-600'}`}>{children}</div>
    </div>
  )
}

function Callout({ type = 'info', children }) {
  const { isDark } = useTheme()
  const styles = {
    info: isDark ? 'border-bb-accent/40 bg-bb-accent/5' : 'border-bb-accent-dark/40 bg-bb-accent-dark/5',
    warning: 'border-yellow-500/40 bg-yellow-500/5',
    tip: 'border-green-400/40 bg-green-400/5',
  }
  const icons = {
    info: <Shield className={`w-4 h-4 shrink-0 mt-0.5 ${isDark ? 'text-bb-accent' : 'text-bb-accent-dark'}`} />,
    warning: <AlertTriangle className="w-4 h-4 text-yellow-500 shrink-0 mt-0.5" />,
    tip: <Lightbulb className="w-4 h-4 text-green-400 shrink-0 mt-0.5" />,
  }
  return (
    <div className={`border-l-2 ${styles[type]} rounded-r-lg p-3 flex gap-2 text-xs ${isDark ? 'text-gray-400' : 'text-gray-600'}`}>
      {icons[type]}
      <div>{children}</div>
    </div>
  )
}

const NAV_ITEMS = [
  { id: 'overview', icon: BookOpen, label: 'Overview' },
  { id: 'installation', icon: Package, label: 'Installation' },
  { id: 'quickstart', icon: Zap, label: 'Quick Start' },
  { id: 'cli', icon: Terminal, label: 'CLI Reference' },
  { id: 'python-api', icon: Code, label: 'Python API' },
  { id: 'datasets', icon: Database, label: 'Datasets' },
  { id: 'phase1', icon: Cpu, label: 'Phase 1: Individual' },
  { id: 'phase2-persona', icon: UserCheck, label: 'Persona Effect' },
  { id: 'phase2-ensemble', icon: Users, label: 'Majority Voting' },
  { id: 'phase2-debate', icon: MessageSquare, label: 'Multi-Agent Debate' },
  { id: 'prompts', icon: FileText, label: 'Experimental Prompts' },
  { id: 'configuration', icon: Settings, label: 'Configuration' },
  { id: 'structure', icon: Layers, label: 'Project Structure' },
  { id: 'extending', icon: GitBranch, label: 'Extending' },
  { id: 'troubleshooting', icon: Wrench, label: 'Troubleshooting' },
  { id: 'citation', icon: FileText, label: 'Citation' },
]

export default function Documentation() {
  const [activeSection, setActiveSection] = useState('overview')
  const { isDark } = useTheme()

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 lg:pl-64">
      {/* Sidebar */}
      <nav className={`hidden lg:flex flex-col fixed left-0 top-16 bottom-0 w-56 z-40 border-r px-4 pt-8 pb-6 overflow-y-auto ${
        isDark
          ? 'bg-bb-dark-500/90 backdrop-blur-xl border-bb-dark-50/20'
          : 'bg-white/90 backdrop-blur-xl border-bb-light-300/50'
      }`}>
        <div className="flex items-center gap-2 mb-6">
          <Hexagon className={`w-5 h-5 ${isDark ? 'text-bb-accent' : 'text-bb-accent-dark'}`} />
          <span className={`text-sm font-bold ${isDark ? 'text-gray-300' : 'text-gray-700'}`}>Documentation</span>
        </div>
        <ul className="space-y-1 flex-1">
          {NAV_ITEMS.map(item => (
            <li key={item.id}>
              <button
                onClick={() => {
                  setActiveSection(item.id)
                  const el = document.getElementById(item.id)
                  if (el) el.scrollIntoView({ behavior: 'smooth', block: 'start' })
                }}
                className={`w-full flex items-center gap-2 px-3 py-2 rounded-lg text-xs font-medium transition-all text-left ${
                  activeSection === item.id
                    ? isDark ? 'bg-bb-accent/10 text-bb-accent' : 'bg-bb-accent-dark/10 text-bb-accent-dark'
                    : isDark ? 'text-gray-500 hover:text-gray-300 hover:bg-bb-dark-300/30' : 'text-gray-500 hover:text-gray-700 hover:bg-bb-light-200'
                }`}
              >
                <item.icon className="w-3.5 h-3.5 shrink-0" />
                {item.label}
              </button>
            </li>
          ))}
        </ul>
      </nav>

      {/* Content */}
      <div className="min-w-0">
          {/* Header */}
          <div className="glass-card p-6 mb-8">
            <div className="flex items-center gap-3 mb-3">
              <div className={`w-10 h-10 rounded-xl flex items-center justify-center ${
                isDark ? 'bg-gradient-to-br from-bb-accent/20 to-bb-teal/20' : 'bg-gradient-to-br from-bb-accent-dark/15 to-bb-teal/15'
              }`}>
                <BookOpen className={`w-5 h-5 ${isDark ? 'text-bb-accent' : 'text-bb-accent-dark'}`} />
              </div>
              <div>
                <h1 className={`text-2xl font-bold ${isDark ? 'text-white' : 'text-gray-900'}`}>SLMJury Documentation</h1>
                <p className={`text-xs ${isDark ? 'text-gray-500' : 'text-gray-400'}`}>v0.1.0 &bull; pip install slmjury</p>
              </div>
            </div>
            <p className={`text-sm leading-relaxed ${isDark ? 'text-gray-400' : 'text-gray-600'}`}>
              Complete documentation for the SLMJury evaluation framework. SLMJury investigates whether Small Language Models (0.6B–14B parameters) can serve as reliable judges of mathematical and scientific reasoning, matching or exceeding the performance of much larger proprietary models.
            </p>
          </div>

          {/* ── Overview ── */}
          <Section id="overview" icon={BookOpen} title="Overview">
            <p>
              <strong className={isDark ? 'text-gray-200' : 'text-gray-800'}>SLMJury</strong> is a comprehensive framework to evaluate whether Small Language Models can serve as accurate, cost-effective judges of student solutions across mathematical and scientific benchmarks. The project explores <strong className={isDark ? 'text-bb-accent' : 'text-bb-accent-dark'}>four evaluation paradigms</strong>: individual judging, persona-based evaluation, majority-vote ensembles, and multi-agent debate.
            </p>
            <div className="grid grid-cols-1 md:grid-cols-4 gap-3 mt-4">
              <div className="glass-card p-4">
                <div className={`text-sm font-semibold mb-1 ${isDark ? 'text-gray-200' : 'text-gray-800'}`}>16 Judges</div>
                <div className="text-xs text-gray-500">0.6B to 14B parameters from Qwen, Microsoft, and Meta model families.</div>
              </div>
              <div className="glass-card p-4">
                <div className={`text-sm font-semibold mb-1 ${isDark ? 'text-gray-200' : 'text-gray-800'}`}>5 Benchmarks</div>
                <div className="text-xs text-gray-500">GSM8K, GSM-Plus, MATH, ARC-Easy, and ARC-Challenge evaluation datasets.</div>
              </div>
              <div className="glass-card p-4">
                <div className={`text-sm font-semibold mb-1 ${isDark ? 'text-gray-200' : 'text-gray-800'}`}>2,400+ Experiments</div>
                <div className="text-xs text-gray-500">Cross-product of judges × students × token settings × personas × ensembles.</div>
              </div>
              <div className="glass-card p-4">
                <div className={`text-sm font-semibold mb-1 ${isDark ? 'text-gray-200' : 'text-gray-800'}`}>4 Paradigms</div>
                <div className="text-xs text-gray-500">Individual, Persona Effect, Majority Voting, and Multi-Agent Debate evaluation.</div>
              </div>
            </div>
          </Section>

          {/* ── Installation ── */}
          <Section id="installation" icon={Package} title="Installation">
            <SubSection title="From Source (Recommended)">
              <CodeBlock code={`git clone https://github.com/anishh15/SLMJury.git
cd SLMJury
pip install -e .`} />
            </SubSection>

            <SubSection title="Requirements">
              <ul className="list-disc list-inside text-gray-500 space-y-1 text-xs">
                <li>Python 3.9+</li>
                <li>PyTorch 2.0+ with CUDA support</li>
                <li>vLLM (for running model inference)</li>
                <li>Transformers &amp; Tokenizers</li>
                <li>NVIDIA GPU with sufficient VRAM (see models.yaml for memory settings)</li>
              </ul>
            </SubSection>

            <SubSection title="Verify Installation">
              <CodeBlock code={`# Verify all imports work
python -c "from slmjury import load_models_config, JudgeModel; print('OK')"

# Check available judge models
python -c "from slmjury.configs import load_models_config; print(list(load_models_config()['judge_models'].keys()))"`} />
            </SubSection>
          </Section>

          {/* ── Quick Start ── */}
          <Section id="quickstart" icon={Zap} title="Quick Start">
            <SubSection title="1. Generate Student Solutions">
              <CodeBlock code={`python scripts/run_students.py \\
  --model qwen3-4b \\
  --datasets gsm8k gsm_plus math arc_easy arc_challenge \\
  --output-dir results/phase1/student_solutions`} />
            </SubSection>

            <SubSection title="2. Run Judge Evaluations">
              <CodeBlock code={`python scripts/run_judges.py \\
  --judge qwen3-4b \\
  --max-tokens 10 8192 \\
  --solutions-dir results/phase1/student_solutions \\
  --output-dir results/phase1/judgements`} />
            </SubSection>

            <SubSection title="3. Compute Accuracy Summaries">
              <CodeBlock code={`python scripts/run_evaluations.py \\
  --judgements-dir results/phase1/judgements \\
  --output-dir results/phase1/summaries`} />
            </SubSection>

            <SubSection title="Python API">
              <CodeBlock language="python" code={`from slmjury.core.judge import JudgeModel
from slmjury.core.evaluator import JudgeEvaluator

# Initialize a judge model
judge = JudgeModel("qwen3-4b", output_dir="results/judgements")

# Evaluate student solutions
results = judge.evaluate_batch(student_results, max_tokens=10)
judge.save_results(results, student_model="phi4-14b", dataset="gsm8k", max_tokens=10)

# Compute metrics
evaluator = JudgeEvaluator("qwen3-4b", "phi4-14b", "gsm8k", 10, results)
summary = evaluator.evaluate()
print(f"Accuracy: {evaluator.accuracy:.2%}, IFR: {evaluator.ifr:.2%}")`} />
            </SubSection>
          </Section>

          {/* ── CLI Reference ── */}
          <Section id="cli" icon={Terminal} title="CLI Reference">
            <SubSection title="run_students.py">
              <p>Run student model inference across specified datasets.</p>
              <div className="glass-card p-4 font-mono text-xs space-y-2 text-gray-400">
                <div><span className={isDark ? 'text-bb-accent' : 'text-bb-accent-dark'}>--model</span> TEXT &nbsp;&nbsp; Student model key from models.yaml (required)</div>
                <div><span className={isDark ? 'text-bb-accent' : 'text-bb-accent-dark'}>--datasets</span> TEXT... &nbsp;&nbsp; Datasets to solve (default: gsm8k gsm_plus math arc_easy arc_challenge)</div>
                <div><span className={isDark ? 'text-bb-accent' : 'text-bb-accent-dark'}>--num-samples</span> INT &nbsp;&nbsp; Limit number of problems per dataset</div>
                <div><span className={isDark ? 'text-bb-accent' : 'text-bb-accent-dark'}>--output-dir</span> PATH &nbsp;&nbsp; Output directory (default: results/phase1/student_solutions)</div>
              </div>
            </SubSection>

            <SubSection title="run_judges.py">
              <p>Run judge model evaluation on student solutions.</p>
              <div className="glass-card p-4 font-mono text-xs space-y-2 text-gray-400">
                <div><span className={isDark ? 'text-bb-accent' : 'text-bb-accent-dark'}>--judge</span> TEXT &nbsp;&nbsp; Judge model key from models.yaml (required)</div>
                <div><span className={isDark ? 'text-bb-accent' : 'text-bb-accent-dark'}>--max-tokens</span> INT... &nbsp;&nbsp; Token settings to evaluate (default: 10 8192)</div>
                <div><span className={isDark ? 'text-bb-accent' : 'text-bb-accent-dark'}>--solutions-dir</span> PATH &nbsp;&nbsp; Directory containing student solutions (default: results/phase1/student_solutions)</div>
                <div><span className={isDark ? 'text-bb-accent' : 'text-bb-accent-dark'}>--output-dir</span> PATH &nbsp;&nbsp; Output directory (default: results/phase1/judgements)</div>
              </div>
              <Callout type="tip">
                Models marked as <code className={`font-mono ${isDark ? 'text-bb-accent' : 'text-bb-accent-dark'}`}>always_thinks: true</code> in models.yaml automatically skip <code className={`font-mono ${isDark ? 'text-bb-accent' : 'text-bb-accent-dark'}`}>max_tokens=10</code> since they cannot suppress chain-of-thought.
              </Callout>
            </SubSection>

            <SubSection title="run_evaluations.py">
              <p>Evaluate judge accuracy and generate summary reports.</p>
              <div className="glass-card p-4 font-mono text-xs space-y-2 text-gray-400">
                <div><span className={isDark ? 'text-bb-accent' : 'text-bb-accent-dark'}>--judgements-dir</span> PATH &nbsp;&nbsp; Directory containing judgement files (default: results/phase1/judgements)</div>
                <div><span className={isDark ? 'text-bb-accent' : 'text-bb-accent-dark'}>--output-dir</span> PATH &nbsp;&nbsp; Output directory (default: results/phase1/summaries)</div>
              </div>
            </SubSection>
          </Section>

          {/* ── Python API ── */}
          <Section id="python-api" icon={Code} title="Python API">
            <SubSection title="Core Classes">
              <div className="space-y-4">
                <div className="glass-card p-4">
                  <div className={`text-sm font-semibold font-mono mb-2 ${isDark ? 'text-bb-accent' : 'text-bb-accent-dark'}`}>JudgeModel</div>
                  <p className="text-xs text-gray-500 mb-2">SLM judge for evaluating student solutions via vLLM. Loads a model once and evaluates multiple student-dataset combinations.</p>
                  <CodeBlock language="python" code={`from slmjury.core.judge import JudgeModel

# Initialize with model key from models.yaml
judge = JudgeModel(
    model_key="qwen3-4b",
    output_dir="results/judgements",
    model_config=None  # optional config override
)

# Evaluate a batch of student solutions
results = judge.evaluate_batch(
    student_results,          # list of student solution dicts
    max_tokens=10,            # 10 = quick verdict, 8192 = reasoned
    system_prompt=None        # optional persona prompt
)

# Save results to disk
judge.save_results(results, "phi4-14b", "gsm8k", max_tokens=10)

# Cleanup GPU memory
judge.cleanup()`} />
                </div>

                <div className="glass-card p-4">
                  <div className={`text-sm font-semibold font-mono mb-2 ${isDark ? 'text-bb-accent' : 'text-bb-accent-dark'}`}>StudentSolver</div>
                  <p className="text-xs text-gray-500 mb-2">Runs student model inference to generate solutions for benchmark problems.</p>
                  <CodeBlock language="python" code={`from slmjury.core.solver import StudentSolver
from slmjury.data import load_dataset

solver = StudentSolver("qwen3-4b", output_dir="results/solutions")
problems = load_dataset("gsm8k")
results = solver.solve_batch(problems, "gsm8k", num_samples=100)
solver.save_results(results, "gsm8k")
solver.cleanup()`} />
                </div>

                <div className="glass-card p-4">
                  <div className={`text-sm font-semibold font-mono mb-2 ${isDark ? 'text-bb-accent' : 'text-bb-accent-dark'}`}>JudgeEvaluator</div>
                  <p className="text-xs text-gray-500 mb-2">Computes accuracy, instruction-following rate, and per-dataset breakdowns from judgement files.</p>
                  <CodeBlock language="python" code={`from slmjury.core.evaluator import JudgeEvaluator, generate_judge_summary

evaluator = JudgeEvaluator(
    judge="qwen3-4b",
    student="phi4-14b",
    dataset="gsm8k",
    tokens=10,
    judgements=results  # list of judgement dicts
)
summary = evaluator.evaluate()
print(f"Accuracy: {evaluator.accuracy:.2%}")
print(f"Instruction Following Rate: {evaluator.ifr:.2%}")`} />
                </div>
              </div>
            </SubSection>

            <SubSection title="Phase 2 Strategies">
              <CodeBlock language="python" code={`# Persona Effect
from slmjury.strategies.persona import run_persona_evaluation, get_personas
personas = get_personas()  # dict of 6 persona prompts
run_persona_evaluation("qwen3-4b", student_results, max_tokens=10)

# Majority Voting Ensemble
from slmjury.strategies.ensemble import run_majority_voting
run_majority_voting(judge_keys=["qwen3-4b", "phi4mi-3.8b", "qwen2.5-3b"],
                    student_results=student_results, max_tokens=10)

# Multi-Agent Debate
from slmjury.strategies.debate import run_debate
run_debate(models=["qwen3-4b", "phi4mi-3.8b", "qwen2.5-3b"],
           temperatures=[0.0, 0.0, 0.0],
           student_results=student_results, dataset="gsm8k")`} />
            </SubSection>
          </Section>

          {/* ── Datasets ── */}
          <Section id="datasets" icon={Database} title="Supported Datasets">
            <div className="glass-card p-4 text-xs">
              <table className="w-full">
                <thead>
                  <tr className={`border-b ${isDark ? 'border-bb-dark-50/20' : 'border-gray-200'}`}>
                    <th className={`text-left py-2 ${isDark ? 'text-gray-500' : 'text-gray-400'}`}>Dataset</th>
                    <th className={`text-left py-2 ${isDark ? 'text-gray-500' : 'text-gray-400'}`}>Domain</th>
                    <th className={`text-left py-2 ${isDark ? 'text-gray-500' : 'text-gray-400'}`}>Problems</th>
                    <th className={`text-left py-2 ${isDark ? 'text-gray-500' : 'text-gray-400'}`}>Key</th>
                  </tr>
                </thead>
                <tbody className={isDark ? 'text-gray-400' : 'text-gray-600'}>
                  <tr className={`border-b ${isDark ? 'border-bb-dark-50/10' : 'border-gray-100'}`}><td className="py-1.5">GSM8K</td><td className="py-1.5">Grade School Math</td><td className="py-1.5">8,792</td><td className="py-1.5 font-mono">gsm8k</td></tr>
                  <tr className={`border-b ${isDark ? 'border-bb-dark-50/10' : 'border-gray-100'}`}><td className="py-1.5">GSM-Plus</td><td className="py-1.5">Extended Math</td><td className="py-1.5">8,792</td><td className="py-1.5 font-mono">gsm_plus</td></tr>
                  <tr className={`border-b ${isDark ? 'border-bb-dark-50/10' : 'border-gray-100'}`}><td className="py-1.5">MATH</td><td className="py-1.5">Competition Math</td><td className="py-1.5">5,000</td><td className="py-1.5 font-mono">math</td></tr>
                  <tr className={`border-b ${isDark ? 'border-bb-dark-50/10' : 'border-gray-100'}`}><td className="py-1.5">ARC-Easy</td><td className="py-1.5">Science (Easy)</td><td className="py-1.5">9,627</td><td className="py-1.5 font-mono">arc_easy</td></tr>
                  <tr><td className="py-1.5">ARC-Challenge</td><td className="py-1.5">Science (Hard)</td><td className="py-1.5">8,627</td><td className="py-1.5 font-mono">arc_challenge</td></tr>
                </tbody>
              </table>
            </div>
          </Section>

          {/* ── Phase 1 ── */}
          <Section id="phase1" icon={Cpu} title="Phase 1: Individual Judge Evaluation">
            <p>Each judge model independently evaluates student solutions under two <strong className={isDark ? 'text-gray-200' : 'text-gray-800'}>token budget settings</strong>:</p>
            <ul className="list-disc list-inside text-gray-500 space-y-1 text-xs mt-2">
              <li><strong>Concise (max_tokens=10)</strong> — forces a one-word verdict: "Correct" or "Incorrect"</li>
              <li><strong>Reasoned (max_tokens=8192)</strong> — allows chain-of-thought reasoning before a \boxed verdict</li>
            </ul>

            <SubSection title="Running Phase 1">
              <CodeBlock code={`# Evaluate a single judge across all students & datasets
python scripts/run_judges.py --judge qwen3-4b --max-tokens 10 8192

# Evaluate with only the quick verdict mode
python scripts/run_judges.py --judge phi4mi-3.8b --max-tokens 10`} />
            </SubSection>

            <SubSection title="Supported Judge Models">
              <div className="glass-card p-4 text-xs">
                <div className={`grid grid-cols-2 md:grid-cols-3 gap-2 font-mono ${isDark ? 'text-gray-500' : 'text-gray-400'}`}>
                  <div>Qwen3-14B</div>
                  <div>Qwen3-8B</div>
                  <div>Qwen3-4B</div>
                  <div>Qwen3-1.7B</div>
                  <div>Qwen3-0.6B</div>
                  <div>Qwen2.5-7B-Instruct</div>
                  <div>Qwen2.5-3B-Instruct</div>
                  <div>Qwen2.5-1.5B-Instruct</div>
                  <div>Phi-4 (14B)</div>
                  <div>Phi-4-mini-instruct (3.8B)</div>
                  <div>Phi-4-reasoning (14B)</div>
                  <div>Phi-4-reasoning-plus (14B)</div>
                  <div>Phi-4-mini-reasoning (3.8B)</div>
                  <div>Llama-3.1-8B-Instruct</div>
                  <div>Llama-3.2-3B-Instruct</div>
                  <div>Llama-3.2-1B-Instruct</div>
                </div>
              </div>
            </SubSection>
          </Section>

          {/* ── Persona Effect ── */}
          <Section id="phase2-persona" icon={UserCheck} title="Phase 2: Persona Effect">
            <p>
              Measures how <strong className={isDark ? 'text-gray-200' : 'text-gray-800'}>system prompts (personas)</strong> affect judge quality. Each judge adopts 6 distinct evaluation personas and runs the same evaluation pipeline to study the impact of role-prompting on judging accuracy.
            </p>
            <SubSection title="6 Personas">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                {[
                  { name: 'Strict', desc: 'Critical academic reviewer — penalizes vagueness, rewards rigor' },
                  { name: 'Lenient', desc: 'Supportive teacher — gives partial credit, rewards effort' },
                  { name: 'Industry', desc: 'Industry professional — prioritizes practicality and actionability' },
                  { name: 'Logic', desc: 'Logic evaluator — judges only internal reasoning quality' },
                  { name: 'Safety', desc: 'Safety auditor — penalizes hallucinations and unsupported claims' },
                  { name: 'Helpfulness', desc: 'Helpfulness judge — rewards clarity, directness, and relevance' },
                ].map(p => (
                  <div key={p.name} className="glass-card p-3">
                    <div className={`text-sm font-semibold mb-1 ${isDark ? 'text-gray-200' : 'text-gray-800'}`}>{p.name}</div>
                    <div className="text-xs text-gray-500">{p.desc}</div>
                  </div>
                ))}
              </div>
            </SubSection>
            <Callout type="info">
              See the <a href="#prompts" className={`underline ${isDark ? 'text-bb-accent' : 'text-bb-accent-dark'}`}>Experimental Prompts</a> section for the exact persona prompt text used in all runs.
            </Callout>
          </Section>

          {/* ── Majority Voting ── */}
          <Section id="phase2-ensemble" icon={Users} title="Phase 2: Majority Voting Ensemble">
            <p>
              <strong className={isDark ? 'text-gray-200' : 'text-gray-800'}>C(5,3) = 10 combinations</strong> of the top 5 individual judges. Each problem is evaluated by 3 judge models and the majority verdict wins. This demonstrates that ensembles of small judges can exceed any individual judge's accuracy.
            </p>
            <SubSection title="Running Majority Voting">
              <CodeBlock language="python" code={`from slmjury.strategies.ensemble import run_majority_voting, majority_vote

# Run a 3-judge ensemble
run_majority_voting(
    judge_keys=["qwen3-4b", "phi4mi-3.8b", "qwen2.5-3b"],
    student_results=student_results,
    max_tokens=10,
    output_dir="results/phase2/ensemble"
)

# Or apply majority vote to existing judgements
verdict = majority_vote(["Correct", "Incorrect", "Correct"])  # → "Correct"`} />
            </SubSection>
          </Section>

          {/* ── Multi-Agent Debate ── */}
          <Section id="phase2-debate" icon={MessageSquare} title="Phase 2: Multi-Agent Debate">
            <p>
              Three SLM agents engage in <strong className={isDark ? 'text-gray-200' : 'text-gray-800'}>RCR (Reflect-Critique-Refine)</strong> debate. Each agent independently judges a problem, then they exchange judgements and debate for up to {'{'}MAX_DEBATE_ROUNDS{'}'} = 5 rounds until consensus. Unresolved problems fall back to majority vote.
            </p>
            <SubSection title="Two Variants">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                <div className="glass-card p-4">
                  <div className={`text-sm font-semibold mb-1 ${isDark ? 'text-gray-200' : 'text-gray-800'}`}>Variant A: Cross-Architecture</div>
                  <div className="text-xs text-gray-500">3 different models at temperature 0.0. Diversity from architectural differences.</div>
                </div>
                <div className="glass-card p-4">
                  <div className={`text-sm font-semibold mb-1 ${isDark ? 'text-gray-200' : 'text-gray-800'}`}>Variant B: Same Model</div>
                  <div className="text-xs text-gray-500">Same model at temperatures 0.0, 0.4, 0.9. Diversity from sampling randomness.</div>
                </div>
              </div>
            </SubSection>
            <Callout type="warning">
              Multi-agent debate requires significantly more GPU memory than individual evaluation because multiple model contexts are maintained simultaneously. See <code className={`font-mono ${isDark ? 'text-bb-accent' : 'text-bb-accent-dark'}`}>_DEBATE_MAX_NUM_SEQS</code> in debate.py for memory-tuned batch sizes.
            </Callout>
          </Section>

          {/* ── Experimental Prompts ── */}
          <Section id="prompts" icon={FileText} title="Experimental Prompts">
            <p>
              The exact prompts used in all experiments. These are preserved verbatim to ensure reproducibility. <strong className={isDark ? 'text-gray-200' : 'text-gray-800'}>Do not modify</strong> — any change would alter experimental results.
            </p>

            <SubSection title="Quick Verdict Prompt (max_tokens=10)">
              <CodeBlock language="text" code={`Your role is to compare the student's answer to the ground truth and determine correctness.

Respond with exactly one word:
- 'Correct': If the student's final answer matches the ground truth.
- 'Incorrect': If the student's final answer does not match.

Focus only on the final answer, ignore reasoning steps.

Question: {question}

Ground truth answer: {ground_truth}

Student answer: {student_answer}`} />
            </SubSection>

            <SubSection title="Reasoned Verdict Prompt (max_tokens=8192)">
              <CodeBlock language="text" code={`Your role is to compare the student's answer to the ground truth and determine correctness.

First, explain your reasoning for the judgment.
Then, on a new line, give the final verdict in the exact format:
\\boxed{CORRECT} or \\boxed{INCORRECT}

Focus only on the final answer when judging correctness. Ignore reasoning steps in the student's solution.

Question: {question}

Ground truth answer: {ground_truth}

Student answer: {student_answer}`} />
            </SubSection>

            <SubSection title="Persona: Strict">
              <CodeBlock language="text" code={`You are a highly critical academic reviewer.
Evaluate the answer for factual correctness, logical consistency, completeness, and absence of unsupported claims.
Penalize vagueness, missing steps, and incorrect terminology.
Do not reward verbosity unless it improves rigor.
Prefer precise, structured, and well-justified answers.
If uncertain, lower the score.`} />
            </SubSection>

            <SubSection title="Persona: Lenient">
              <CodeBlock language="text" code={`You are a supportive teacher grading student work.
Focus on whether the core idea is correct even if wording is imperfect.
Give partial credit for partially correct reasoning.
Do not heavily penalize minor factual or grammatical errors.
Reward effort, clarity of intent, and basic understanding.`} />
            </SubSection>

            <SubSection title="Persona: Industry">
              <CodeBlock language="text" code={`You are an industry professional evaluating usefulness for real-world application.
Prioritize correctness, practicality, and actionability.
Penalize unnecessary theory, long explanations, and fluff.
Prefer short, clear, implementable answers.
An answer that can be used immediately should score higher.`} />
            </SubSection>

            <SubSection title="Persona: Logic">
              <CodeBlock language="text" code={`You are a logic-focused evaluator.
Judge only based on internal reasoning quality.
Ignore writing style, tone, and length.
Check whether conclusions follow from premises.
Penalize contradictions, missing steps, and invalid inferences.
Factual correctness matters only if it affects logical validity.`} />
            </SubSection>

            <SubSection title="Persona: Safety">
              <CodeBlock language="text" code={`You are a safety auditor checking for hallucinations and unsupported claims.
Penalize confident statements without evidence.
Reward uncertainty when appropriate.
If the answer invents facts, names, or numbers, assign a low score.
Factual grounding and cautious language should score higher than fluency.`} />
            </SubSection>

            <SubSection title="Persona: Helpfulness">
              <CodeBlock language="text" code={`You are judging how helpful the answer is to the user.
Focus on whether the response solves the user's problem.
Reward clarity, directness, and relevance.
Penalize tangents, missing steps, and unclear instructions.
An answer that enables the user to act should receive a higher score.`} />
            </SubSection>

            <SubSection title="Debate Prompt (Math Datasets)">
              <CodeBlock language="text" code={`You are Agent {agent_id} in a multi-agent debate to judge whether a student's math answer is correct.

Question: {question}

Ground truth answer: {ground_truth}

Student answer: {student_answer}

{own_previous}

Here are the judgements from other agents:
{context}

This is debate round {round_num}. Please carefully analyze all judgements—including your own—identify any errors in reasoning, and provide your revised judgement.
- If you believe your previous verdict is correct, explain why and defend it.
- If you believe you made an error, explain the error and provide a corrected judgement.
- If you believe another agent's verdict is correct, explain why you agree with it.

Your final verdict must be: \\boxed{CORRECT} or \\boxed{INCORRECT}`} />
            </SubSection>

            <SubSection title="Debate Prompt (Science Datasets)">
              <CodeBlock language="text" code={`You are Agent {agent_id} in a multi-agent debate to judge whether a student's science answer is correct.

Question: {question}

Ground truth answer: {ground_truth}

Student answer: {student_answer}

{own_previous}

Here are the judgements from other agents:
{context}

This is debate round {round_num}. Please carefully analyze all judgements—including your own—identify any misconceptions or flawed scientific reasoning, and provide your revised judgement.
- If you believe your previous verdict is correct, explain the scientific principles supporting your verdict.
- If you believe you made an error, explain the scientific misconception and provide a corrected judgement.
- If you believe another agent's verdict is correct, explain why their scientific reasoning is sound.

Your final verdict must be: \\boxed{CORRECT} or \\boxed{INCORRECT}`} />
            </SubSection>
          </Section>

          {/* ── Configuration ── */}
          <Section id="configuration" icon={Settings} title="Configuration">
            <SubSection title="models.yaml">
              <p>Central configuration file for all judge and student models. Located at <code className={`font-mono ${isDark ? 'text-bb-accent' : 'text-bb-accent-dark'}`}>slmjury/configs/models.yaml</code>.</p>
              <CodeBlock language="yaml" code={`judge_models:
  qwen3-4b:
    model_name: "Qwen/Qwen3-4B"
    enable_thinking: true
    always_thinks: false
    max_num_seqs: 64
    gpu_memory_utilization: 0.95
    tensor_parallel_size: 1

  phi4-14b:
    model_name: "microsoft/phi-4"
    enable_thinking: false
    always_thinks: false
    max_num_seqs: 64
    gpu_memory_utilization: 0.95
    tensor_parallel_size: 1

student_models:
  qwen3-4b:
    model_name: "Qwen/Qwen3-4B"
    gpu_memory_utilization: 0.95
    tensor_parallel_size: 1`} />
            </SubSection>

            <SubSection title="Key Configuration Fields">
              <div className="glass-card p-4 text-xs space-y-2">
                <div><span className={`font-mono font-semibold ${isDark ? 'text-bb-accent' : 'text-bb-accent-dark'}`}>model_name</span> — HuggingFace model identifier</div>
                <div><span className={`font-mono font-semibold ${isDark ? 'text-bb-accent' : 'text-bb-accent-dark'}`}>enable_thinking</span> — Enable Qwen3 thinking mode for reasoned verdicts</div>
                <div><span className={`font-mono font-semibold ${isDark ? 'text-bb-accent' : 'text-bb-accent-dark'}`}>always_thinks</span> — If true, skip max_tokens=10 (model can't suppress CoT)</div>
                <div><span className={`font-mono font-semibold ${isDark ? 'text-bb-accent' : 'text-bb-accent-dark'}`}>max_num_seqs</span> — vLLM batch size (lower for memory-constrained models)</div>
                <div><span className={`font-mono font-semibold ${isDark ? 'text-bb-accent' : 'text-bb-accent-dark'}`}>gpu_memory_utilization</span> — Fraction of GPU memory for vLLM (0.0–1.0)</div>
                <div><span className={`font-mono font-semibold ${isDark ? 'text-bb-accent' : 'text-bb-accent-dark'}`}>tensor_parallel_size</span> — Number of GPUs for tensor parallelism</div>
              </div>
            </SubSection>
          </Section>

          {/* ── Project Structure ── */}
          <Section id="structure" icon={Layers} title="Project Structure">
            <CodeBlock language="text" code={`SLMJury/
├── slmjury/                  # Core framework package
│   ├── __init__.py           # Public API exports
│   ├── configs/              # Model + prompt configuration
│   │   ├── __init__.py       # load_models_config()
│   │   └── models.yaml       # Judge & student model definitions
│   ├── core/                 # Core evaluation logic
│   │   ├── evaluator.py      # JudgeEvaluator + accuracy metrics
│   │   ├── judge.py          # JudgeModel (vLLM-based inference)
│   │   └── solver.py         # StudentSolver (student inference)
│   ├── data/                 # Dataset loaders
│   │   └── __init__.py       # load_dataset() for all benchmarks
│   ├── parsers/              # Output parsing
│   │   ├── answer.py         # Student answer parsing
│   │   └── judgement.py      # Judge verdict parsing
│   └── strategies/           # Phase 2 strategies
│       ├── debate.py         # Multi-Agent Debate (MAD)
│       ├── ensemble.py       # Majority Voting
│       └── persona.py        # Persona Effect evaluation
├── scripts/                  # CLI entry-points
│   ├── run_students.py       # Generate student solutions
│   ├── run_judges.py         # Run judge evaluations
│   └── run_evaluations.py    # Compute accuracy summaries
├── website/                  # React leaderboard + docs
│   └── src/
│       ├── pages/            # Leaderboard + Documentation
│       ├── components/       # Navbar, Footer
│       └── data/             # modelData.js (results data)
├── assets/                   # SVG banner + logo
├── pyproject.toml            # Python package metadata
└── README.md                 # Project documentation`} />
          </Section>

          {/* ── Extending ── */}
          <Section id="extending" icon={GitBranch} title="Extending SLMJury">
            <SubSection title="Adding a New Judge Model">
              <ol className="list-decimal list-inside text-gray-500 space-y-2 text-xs">
                <li>Add the model entry to <code className={`font-mono ${isDark ? 'text-bb-accent' : 'text-bb-accent-dark'}`}>slmjury/configs/models.yaml</code> under <code className="font-mono">judge_models</code>.</li>
                <li>If the model has special tokenizer requirements, update <code className={`font-mono ${isDark ? 'text-bb-accent' : 'text-bb-accent-dark'}`}>slmjury/core/judge.py</code> (chat template handling).</li>
                <li>Run <code className={`font-mono ${isDark ? 'text-bb-accent' : 'text-bb-accent-dark'}`}>python scripts/run_judges.py --judge your-model-key</code> to evaluate.</li>
              </ol>
            </SubSection>

            <SubSection title="Adding a New Dataset">
              <ol className="list-decimal list-inside text-gray-500 space-y-2 text-xs">
                <li>Add a loader function in <code className={`font-mono ${isDark ? 'text-bb-accent' : 'text-bb-accent-dark'}`}>slmjury/data/__init__.py</code>.</li>
                <li>Add an answer parser in <code className={`font-mono ${isDark ? 'text-bb-accent' : 'text-bb-accent-dark'}`}>slmjury/parsers/answer.py</code>.</li>
                <li>Register the dataset key in the loader's dispatch map.</li>
                <li>Run <code className={`font-mono ${isDark ? 'text-bb-accent' : 'text-bb-accent-dark'}`}>python scripts/run_students.py --model your-model --datasets your-dataset</code>.</li>
              </ol>
            </SubSection>

            <SubSection title="Adding a New Persona">
              <p>Add your persona system prompt to the <code className={`font-mono ${isDark ? 'text-bb-accent' : 'text-bb-accent-dark'}`}>PERSONAS</code> dictionary in <code className="font-mono">slmjury/strategies/persona.py</code>. The persona evaluation pipeline will automatically pick it up.</p>
            </SubSection>
          </Section>

          {/* ── Troubleshooting ── */}
          <Section id="troubleshooting" icon={Wrench} title="Troubleshooting">
            <SubSection title="Out of GPU Memory">
              <p>Reduce <code className={`font-mono ${isDark ? 'text-bb-accent' : 'text-bb-accent-dark'}`}>max_num_seqs</code> in models.yaml, lower <code className={`font-mono ${isDark ? 'text-bb-accent' : 'text-bb-accent-dark'}`}>gpu_memory_utilization</code>, or increase <code className={`font-mono ${isDark ? 'text-bb-accent' : 'text-bb-accent-dark'}`}>tensor_parallel_size</code> to distribute across GPUs.</p>
            </SubSection>

            <SubSection title="Disk Full Errors">
              <p>vLLM writes temporary files to <code className="font-mono">/tmp</code>. If the root partition is full, set environment variables to redirect:</p>
              <CodeBlock code={`export TMPDIR=/path/to/large/tmp
export VLLM_WORKER_MULTIPROC_METHOD=spawn`} />
            </SubSection>

            <SubSection title="Model Skips max_tokens=10">
              <p>Models with <code className={`font-mono ${isDark ? 'text-bb-accent' : 'text-bb-accent-dark'}`}>always_thinks: true</code> cannot suppress chain-of-thought reasoning. They automatically skip the quick verdict setting and only run with max_tokens=8192.</p>
            </SubSection>

            <SubSection title="Debate Memory Issues">
              <p>Multi-agent debate maintains multiple model contexts simultaneously. Use the <code className={`font-mono ${isDark ? 'text-bb-accent' : 'text-bb-accent-dark'}`}>_DEBATE_MAX_NUM_SEQS</code> overrides in debate.py to tune batch sizes per model size.</p>
            </SubSection>
          </Section>

          {/* ── Citation ── */}
          <Section id="citation" icon={BarChart3} title="Citation">
            <CodeBlock language="bibtex" code={`@article{laddha2025slmjury,
  title   = {SLMJury: Can Small Language Models Judge as Well as Large Language Models?},
  author  = {Anish Laddha and Nitesh Pradhan and Gaurav Srivastava},
  year    = {2025},
  note    = {Manuscript in preparation}
}`} />
          </Section>
      </div>
    </div>
  )
}
