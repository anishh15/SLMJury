import { useState, useMemo } from 'react'
import { BookOpen, Terminal, Code, Settings, Layers, Cpu, Package, FileText, Copy, Check, Zap, Database, GitBranch, Shield, BarChart3, Wrench, AlertTriangle, Lightbulb, Users, MessageSquare, UserCheck, BookOpenText, FlaskConical, Search, ChevronDown } from 'lucide-react'
import { useTheme } from '../context/ThemeContext'

/* ============ SYNTAX HIGHLIGHTING ============ */

const darkColors = {
  comment: '#6b7280', keyword: '#c084fc', string: '#34d399', number: '#f59e0b',
  function: '#60a5fa', operator: '#9ca3af', punctuation: '#9ca3af', key: '#60a5fa',
  boolean: '#f59e0b', decorator: '#c084fc', flag: '#60a5fa', text: '#d1d5db',
}
const lightColors = {
  comment: '#9ca3af', keyword: '#7c3aed', string: '#059669', number: '#d97706',
  function: '#2563eb', operator: '#64748b', punctuation: '#64748b', key: '#2563eb',
  boolean: '#d97706', decorator: '#7c3aed', flag: '#2563eb', text: '#374151',
}

function tokenizePython(code) {
  const tokens = []
  const keywords = new Set(['import','from','as','def','return','class','if','else','elif','for','while','with','try','except','raise','True','False','None','print','self','in','not','and','or','is','lambda','yield','async','await','pass','break','continue','del','global','nonlocal','assert','finally'])
  let i = 0
  while (i < code.length) {
    if (code[i] === '#') { let e = code.indexOf('\n', i); if (e === -1) e = code.length; tokens.push({ type: 'comment', value: code.slice(i, e) }); i = e; continue }
    if (code.slice(i, i+3) === '"""' || code.slice(i, i+3) === "'''") { const q = code.slice(i, i+3); let e = code.indexOf(q, i+3); if (e === -1) e = code.length-3; tokens.push({ type: 'string', value: code.slice(i, e+3) }); i = e+3; continue }
    if (code[i] === '"' || code[i] === "'") { const q = code[i]; let j = i+1; while (j < code.length && code[j] !== q) { if (code[j] === '\\') j++; j++ } tokens.push({ type: 'string', value: code.slice(i, j+1) }); i = j+1; continue }
    if (code[i] === '@' && (i === 0 || code[i-1] === '\n' || /\s/.test(code[i-1]))) { let j = i+1; while (j < code.length && /[\w.]/.test(code[j])) j++; tokens.push({ type: 'decorator', value: code.slice(i, j) }); i = j; continue }
    if (/\d/.test(code[i]) && (i === 0 || !/[\w.]/.test(code[i-1]))) { let j = i; while (j < code.length && /[\d.eE_xXa-fA-F]/.test(code[j])) j++; tokens.push({ type: 'number', value: code.slice(i, j) }); i = j; continue }
    if (/[a-zA-Z_]/.test(code[i])) { let j = i; while (j < code.length && /[\w]/.test(code[j])) j++; const w = code.slice(i, j); if (keywords.has(w)) tokens.push({ type: 'keyword', value: w }); else if (j < code.length && code[j] === '(') tokens.push({ type: 'function', value: w }); else tokens.push({ type: 'text', value: w }); i = j; continue }
    if (/[=+\-*/<>!&|^~%]/.test(code[i])) { tokens.push({ type: 'operator', value: code[i] }); i++; continue }
    if (/[()[\]{},;:.]/.test(code[i])) { tokens.push({ type: 'punctuation', value: code[i] }); i++; continue }
    tokens.push({ type: 'text', value: code[i] }); i++
  }
  return tokens
}

function tokenizeBash(code) {
  const tokens = []
  const keywords = new Set(['python','pip','git','cd','source','export','npm','slmjury','CUDA_VISIBLE_DEVICES','rm','tail','mkdir','echo','cat','chmod','sudo','apt','brew','for','do','done'])
  let i = 0
  while (i < code.length) {
    if (code[i] === '#') { let e = code.indexOf('\n', i); if (e === -1) e = code.length; tokens.push({ type: 'comment', value: code.slice(i, e) }); i = e; continue }
    if (code[i] === '"' || code[i] === "'") { const q = code[i]; let j = i+1; while (j < code.length && code[j] !== q) { if (code[j] === '\\') j++; j++ } tokens.push({ type: 'string', value: code.slice(i, j+1) }); i = j+1; continue }
    if (code[i] === '-' && i+1 < code.length && /[a-zA-Z-]/.test(code[i+1])) { let j = i; while (j < code.length && /[\w-]/.test(code[j])) j++; tokens.push({ type: 'flag', value: code.slice(i, j) }); i = j; continue }
    if (/[a-zA-Z_]/.test(code[i])) { let j = i; while (j < code.length && /[\w.\-/]/.test(code[j])) j++; const w = code.slice(i, j); if (keywords.has(w)) tokens.push({ type: 'keyword', value: w }); else tokens.push({ type: 'text', value: w }); i = j; continue }
    if (/\d/.test(code[i])) { let j = i; while (j < code.length && /[\d.]/.test(code[j])) j++; tokens.push({ type: 'number', value: code.slice(i, j) }); i = j; continue }
    tokens.push({ type: 'text', value: code[i] }); i++
  }
  return tokens
}

function tokenizeBibtex(code) {
  const tokens = []
  let i = 0
  while (i < code.length) {
    if (code[i] === '@') { let j = i+1; while (j < code.length && /\w/.test(code[j])) j++; tokens.push({ type: 'decorator', value: code.slice(i, j) }); i = j; continue }
    if (code[i] === '{' || code[i] === '}') { tokens.push({ type: 'punctuation', value: code[i] }); i++; continue }
    if (/[a-zA-Z]/.test(code[i])) { let j = i; while (j < code.length && /[\w]/.test(code[j])) j++; const w = code.slice(i, j); let k = j; while (k < code.length && code[k] === ' ') k++; tokens.push({ type: code[k] === '=' ? 'key' : 'string', value: w }); i = j; continue }
    tokens.push({ type: 'text', value: code[i] }); i++
  }
  return tokens
}

function tokenize(code, language) {
  switch (language) {
    case 'python': case 'py': return tokenizePython(code)
    case 'bash': case 'shell': case 'sh': return tokenizeBash(code)
    case 'bibtex': return tokenizeBibtex(code)
    default: return [{ type: 'text', value: code }]
  }
}

function CodeBlock({ code, language = 'bash' }) {
  const [copied, setCopied] = useState(false)
  const { isDark } = useTheme()
  const colors = isDark ? darkColors : lightColors
  const highlighted = useMemo(() => tokenize(code, language).map((token, i) => {
    if (token.type === 'text') return <span key={i}>{token.value}</span>
    return <span key={i} style={{ color: colors[token.type] }}>{token.value}</span>
  }), [code, language, colors])
  function handleCopy() { navigator.clipboard.writeText(code); setCopied(true); setTimeout(() => setCopied(false), 2000) }
  return (
    <div className={`relative group my-3 rounded-xl overflow-hidden transition-all duration-300 ${isDark ? 'bg-[#0a0e18] border border-bb-dark-50/15 hover:border-bb-accent/20 shadow-lg shadow-black/20' : 'bg-[#f7f8fc] border border-gray-200/60 hover:border-bb-accent-dark/30 shadow-sm'}`}>
      <div className={`flex items-center justify-between px-4 py-2.5 border-b ${isDark ? 'border-bb-dark-50/10 bg-white/[0.02]' : 'border-gray-200/30 bg-gray-50/50'}`}>
        <div className="flex items-center gap-2.5">
          <div className="flex gap-1.5">
            <div className="w-2.5 h-2.5 rounded-full bg-[#ff5f57]/80" />
            <div className="w-2.5 h-2.5 rounded-full bg-[#febc2e]/80" />
            <div className="w-2.5 h-2.5 rounded-full bg-[#28c840]/80" />
          </div>
          <span className={`text-[10px] uppercase tracking-wider font-mono ${isDark ? 'text-gray-600' : 'text-gray-400'}`}>{language}</span>
        </div>
        <button onClick={handleCopy} className={`text-xs font-mono px-2.5 py-1 rounded-md transition-all duration-200 ${copied ? (isDark ? 'text-green-400 bg-green-500/10' : 'text-green-600 bg-green-500/10') : (isDark ? 'text-gray-600 hover:text-gray-300 hover:bg-white/5' : 'text-gray-400 hover:text-gray-600 hover:bg-gray-100')}`}>
          {copied ? 'Copied!' : 'Copy'}
        </button>
      </div>
      <pre className={`p-4 overflow-x-auto text-[13px] font-mono leading-relaxed ${isDark ? 'text-gray-300' : 'text-gray-700'}`}>
        <code>{highlighted}</code>
      </pre>
    </div>
  )
}

function Section({ id, icon: Icon, title, children, active }) {
  const { isDark } = useTheme()
  if (!active) return null
  const cardCls = `p-6 sm:p-8 rounded-2xl ${isDark ? 'glass-card' : 'bg-white/70 backdrop-blur-xl border border-gray-200/60 shadow-sm'}`
  return (
    <div key={id} className="animate-fade-in">
      <div className={cardCls}>
        <h2 className={`text-xl sm:text-2xl font-bold mb-4 ${isDark ? 'text-white' : 'text-gray-900'}`}>{title}</h2>
        <div className={`leading-relaxed text-sm space-y-4 ${isDark ? 'text-gray-400' : 'text-gray-600'}`}>{children}</div>
      </div>
    </div>
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
  const labels = { info: 'Note', warning: 'Warning', tip: 'Tip' }
  const labelColors = { info: isDark ? 'text-bb-accent' : 'text-bb-accent-dark', warning: 'text-yellow-500', tip: 'text-green-400' }
  return (
    <div className={`border-l-4 rounded-xl p-4 ${styles[type]}`}>
      <div className={`text-xs font-mono font-bold uppercase tracking-wider mb-1 flex items-center gap-1.5 ${labelColors[type]}`}>
        {icons[type]}
        {labels[type]}
      </div>
      <div className={`text-sm leading-relaxed ${isDark ? 'text-gray-400' : 'text-gray-600'}`}>{children}</div>
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
  { id: 'individual', icon: Cpu, label: 'Individual Judges' },
  { id: 'persona', icon: UserCheck, label: 'Persona Effect' },
  { id: 'ensemble', icon: Users, label: 'Majority Voting' },
  { id: 'debate', icon: MessageSquare, label: 'Multi-Agent Debate' },
  { id: 'open-ended', icon: BookOpenText, label: 'Open-Ended Scoring' },
  { id: 'prompts', icon: FileText, label: 'Experimental Prompts' },
  { id: 'configuration', icon: Settings, label: 'Configuration' },
  { id: 'structure', icon: Layers, label: 'Project Structure' },
  { id: 'extending', icon: GitBranch, label: 'Extending' },
  { id: 'troubleshooting', icon: Wrench, label: 'Troubleshooting' },
  { id: 'citation', icon: FileText, label: 'Citation' },
]

export default function Documentation() {
  const [activeSection, setActiveSection] = useState('overview')
  const [searchQuery, setSearchQuery] = useState('')
  const { isDark } = useTheme()

  const filteredNav = useMemo(() => {
    if (!searchQuery.trim()) return NAV_ITEMS
    const q = searchQuery.toLowerCase()
    return NAV_ITEMS.filter(i => i.label.toLowerCase().includes(q) || i.id.toLowerCase().includes(q))
  }, [searchQuery])

  return (
    <section className="pt-24 pb-20">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Header */}
        <div className="text-center mb-8">
          <span className={`inline-block px-4 py-1.5 rounded-full text-xs font-semibold tracking-wider uppercase mb-4 ${
            isDark ? 'bg-bb-accent/10 text-bb-accent border border-bb-accent/20' : 'bg-bb-accent-dark/10 text-bb-accent-dark border border-bb-accent-dark/20'
          }`}>Documentation</span>
          <h1 className={`text-3xl sm:text-4xl lg:text-5xl font-bold mb-3 ${isDark ? 'text-white' : 'text-gray-900'}`}>
            SLMJury Docs
          </h1>
          <p className={`text-lg max-w-2xl mx-auto ${isDark ? 'text-gray-400' : 'text-gray-600'}`}>
            Everything you need to evaluate with small language model judges.
          </p>
        </div>

        {/* Search */}
        <div className="max-w-md mx-auto mb-10">
          <div className={`relative rounded-xl transition-all duration-300 ${
            isDark
              ? 'bg-bb-dark-400/30 border border-bb-dark-50/20 focus-within:border-bb-accent/25'
              : 'bg-white border border-gray-200 focus-within:border-bb-accent-dark/40 shadow-sm'
          }`}>
            <Search className={`absolute left-3.5 top-1/2 -translate-y-1/2 w-4 h-4 ${isDark ? 'text-gray-500' : 'text-gray-400'}`} />
            <input
              type="text"
              placeholder="Search documentation..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className={`w-full pl-10 pr-4 py-2.5 rounded-xl text-sm bg-transparent outline-none ${
                isDark ? 'text-white placeholder:text-gray-600' : 'text-gray-900 placeholder:text-gray-400'
              }`}
            />
          </div>
        </div>

        {/* Mobile section selector */}
        <div className="lg:hidden mb-6">
          <div className="relative">
            <select
              value={activeSection}
              onChange={(e) => { setActiveSection(e.target.value); window.scrollTo({ top: 0, behavior: 'smooth' }) }}
              className={`w-full px-4 py-2.5 rounded-xl text-sm font-medium appearance-none ${
                isDark
                  ? 'bg-bb-dark-400/30 text-white border border-bb-dark-50/20'
                  : 'bg-white text-gray-900 border border-gray-200 shadow-sm'
              }`}
            >
              {NAV_ITEMS.map(item => (
                <option key={item.id} value={item.id}>{item.label}</option>
              ))}
            </select>
            <ChevronDown className={`absolute right-3 top-1/2 -translate-y-1/2 w-4 h-4 pointer-events-none ${isDark ? 'text-gray-500' : 'text-gray-400'}`} />
          </div>
        </div>

        {/* Layout: sidebar + content */}
        <div className="flex gap-8">
          {/* Sidebar */}
          <nav className={`hidden lg:block w-56 shrink-0 sticky top-24 self-start rounded-xl p-2 max-h-[calc(100vh-7rem)] overflow-y-auto ${
            isDark
              ? 'bg-bb-dark-400/20 border border-bb-dark-50/10'
              : 'bg-white/60 border border-gray-200/30 shadow-sm'
          }`}>
            <div className="space-y-0.5">
              {filteredNav.map((item) => (
                <button
                  key={item.id}
                  onClick={() => { setActiveSection(item.id); window.scrollTo({ top: 0, behavior: 'smooth' }) }}
                  className={`w-full text-left px-3 py-2.5 rounded-lg text-sm transition-all duration-200 flex items-center gap-2.5 ${
                    activeSection === item.id
                      ? isDark
                        ? 'bg-bb-accent/10 text-bb-accent font-semibold'
                        : 'bg-bb-accent-dark/10 text-bb-accent-dark font-semibold'
                      : isDark
                        ? 'text-gray-500 hover:text-gray-300 hover:bg-bb-dark-300/30'
                        : 'text-gray-500 hover:text-gray-700 hover:bg-gray-50'
                  }`}
                >
                  <item.icon className="w-4 h-4 shrink-0" />
                  {item.label}
                </button>
              ))}
            </div>
          </nav>

          {/* Content */}
          <div className="flex-1 min-w-0">

          {/* ── Overview ── */}
          <Section active={activeSection === 'overview'} id="overview" icon={BookOpen} title="Overview">
            <p>
              <strong className={isDark ? 'text-gray-200' : 'text-gray-800'}>SLMJury</strong> is a comprehensive framework to evaluate whether Small Language Models can serve as accurate, cost-effective judges across both <strong className={isDark ? 'text-bb-accent' : 'text-bb-accent-dark'}>closed-ended</strong> (accuracy-based) and <strong className={isDark ? 'text-bb-accent' : 'text-bb-accent-dark'}>open-ended</strong> (correlation-based) evaluation paradigms. The project explores <strong className={isDark ? 'text-bb-accent' : 'text-bb-accent-dark'}>six evaluation modes</strong>: individual judging, persona-based evaluation, majority-vote ensembles, multi-agent debate, human agreement scoring (SummEval), and LLM agreement scoring (MT-Bench).
            </p>
            <div className="grid grid-cols-1 md:grid-cols-4 gap-3 mt-4">
              <div className="glass-card p-4">
                <div className={`text-sm font-semibold mb-1 ${isDark ? 'text-gray-200' : 'text-gray-800'}`}>16 Judges</div>
                <div className="text-xs text-gray-500">0.6B to 14B parameters from Qwen, Microsoft, and Meta model families.</div>
              </div>
              <div className="glass-card p-4">
                <div className={`text-sm font-semibold mb-1 ${isDark ? 'text-gray-200' : 'text-gray-800'}`}>10 Benchmarks</div>
                <div className="text-xs text-gray-500">8 closed-ended (accuracy) + 2 open-ended (SummEval, MT-Bench) datasets.</div>
              </div>
              <div className="glass-card p-4">
                <div className={`text-sm font-semibold mb-1 ${isDark ? 'text-gray-200' : 'text-gray-800'}`}>4,100+ Experiments</div>
                <div className="text-xs text-gray-500">Judges × students × token settings × personas × ensembles × oracle/student combos.</div>
              </div>
              <div className="glass-card p-4">
                <div className={`text-sm font-semibold mb-1 ${isDark ? 'text-gray-200' : 'text-gray-800'}`}>6 Eval Modes</div>
                <div className="text-xs text-gray-500">Individual · Persona · Ensemble · Debate · Human Agreement · LLM Agreement.</div>
              </div>
            </div>
          </Section>

          <Section active={activeSection === 'installation'} id="installation" icon={Package} title="Installation">
            <SubSection title="From PyPI">
              <CodeBlock code={`pip install slmjury

# Optional extras
pip install slmjury[vllm]       # GPU inference with vLLM
pip install slmjury[together]   # Together API for oracle scoring
pip install slmjury[full]       # Everything`} />
            </SubSection>

            <SubSection title="From Source (Development)">
              <CodeBlock code={`git clone https://github.com/anishh15/SLMJury.git
cd SLMJury
pip install -e .`} />
            </SubSection>

            <SubSection title="Requirements">
              <ul className="list-disc list-inside text-gray-500 space-y-1 text-xs">
                <li>Python 3.10+</li>
                <li>PyTorch 2.0+ with CUDA support</li>
                <li>vLLM (for running model inference)</li>
                <li>Transformers &amp; Tokenizers</li>
                <li>NVIDIA GPU with sufficient VRAM (see models.yaml for memory settings)</li>
              </ul>
            </SubSection>

            <SubSection title="Verify Installation">
              <CodeBlock code={`# Verify all imports work
python -c "from slmjury import load_models_config; print('OK')"

# Check available judge models
python -c "from slmjury.configs import load_models_config; print(list(load_models_config()['judge_models'].keys()))"`} />
            </SubSection>
          </Section>

          {/* ── Quick Start ── */}
          <Section active={activeSection === 'quickstart'} id="quickstart" icon={Zap} title="Quick Start">
            <SubSection title="1. Generate Student Solutions">
              <CodeBlock code={`python scripts/run_student.py \\
  --model qwen3-4b \\
  --datasets gsm8k gsm_plus math arc_easy arc_challenge \\
  --output-dir results/student_solutions`} />
            </SubSection>

            <SubSection title="2. Run Judge Evaluations">
              <CodeBlock code={`python scripts/run_judge.py \\
  --judge qwen3-4b \\
  --max-tokens 10 8192 \\
  --solutions-dir results/student_solutions \\
  --output-dir results/judgements`} />
            </SubSection>

            <SubSection title="3. Compute Accuracy Summaries">
              <CodeBlock code={`python scripts/run_evaluation.py \\
  --judgements-dir results/judgements \\
  --output-dir results/evaluations`} />
            </SubSection>

            <SubSection title="Python API">
              <CodeBlock language="python" code={`from slmjury.core.judge import JudgeModel
from slmjury.core.evaluator import JudgeEvaluator

# Initialize a judge model
judge = JudgeModel("qwen3-4b", output_dir="results/judgements")

# Evaluate student solutions
results = judge.evaluate_batch(student_results, max_tokens=10)
judge.save_results(results, "llama3.1-8b", "gsm8k", max_tokens=10)

# Compute metrics
evaluator = JudgeEvaluator("qwen3-4b", "llama3.1-8b", "gsm8k", 10, results)
summary = evaluator.evaluate()
print(f"Accuracy: {evaluator.accuracy:.2%}, IFR: {evaluator.ifr:.2%}")`} />
            </SubSection>
          </Section>

          {/* ── CLI Reference ── */}
          <Section active={activeSection === 'cli'} id="cli" icon={Terminal} title="CLI Reference">
            <SubSection title="run_student.py">
              <p>Run student model inference across specified datasets.</p>
              <div className="glass-card p-4 font-mono text-xs space-y-2 text-gray-400">
                <div><span className={isDark ? 'text-bb-accent' : 'text-bb-accent-dark'}>--model</span> TEXT &nbsp;&nbsp; Student model key from models.yaml (required)</div>
                <div><span className={isDark ? 'text-bb-accent' : 'text-bb-accent-dark'}>--datasets</span> TEXT... &nbsp;&nbsp; Datasets to solve (default: all 8 closed-ended)</div>
                <div><span className={isDark ? 'text-bb-accent' : 'text-bb-accent-dark'}>--num-samples</span> INT &nbsp;&nbsp; Limit number of problems per dataset</div>
                <div><span className={isDark ? 'text-bb-accent' : 'text-bb-accent-dark'}>--output-dir</span> PATH &nbsp;&nbsp; Output directory (default: results/student_solutions)</div>
              </div>
            </SubSection>

            <SubSection title="run_judge.py">
              <p>Run judge model evaluation on student solutions.</p>
              <div className="glass-card p-4 font-mono text-xs space-y-2 text-gray-400">
                <div><span className={isDark ? 'text-bb-accent' : 'text-bb-accent-dark'}>--judge</span> TEXT &nbsp;&nbsp; Judge model key from models.yaml (required)</div>
                <div><span className={isDark ? 'text-bb-accent' : 'text-bb-accent-dark'}>--max-tokens</span> INT... &nbsp;&nbsp; Token settings to evaluate (default: 10 8192)</div>
                <div><span className={isDark ? 'text-bb-accent' : 'text-bb-accent-dark'}>--solutions-dir</span> PATH &nbsp;&nbsp; Directory containing student solutions (default: results/student_solutions)</div>
                <div><span className={isDark ? 'text-bb-accent' : 'text-bb-accent-dark'}>--output-dir</span> PATH &nbsp;&nbsp; Output directory (default: results/judgements)</div>
              </div>
              <Callout type="tip">
                Models marked as <code className={`font-mono ${isDark ? 'text-bb-accent' : 'text-bb-accent-dark'}`}>always_thinks: true</code> in models.yaml automatically skip <code className={`font-mono ${isDark ? 'text-bb-accent' : 'text-bb-accent-dark'}`}>max_tokens=10</code> since they cannot suppress chain-of-thought.
              </Callout>
            </SubSection>

            <SubSection title="run_evaluation.py">
              <p>Evaluate judge accuracy and generate summary reports.</p>
              <div className="glass-card p-4 font-mono text-xs space-y-2 text-gray-400">
                <div><span className={isDark ? 'text-bb-accent' : 'text-bb-accent-dark'}>--judgements-dir</span> PATH &nbsp;&nbsp; Directory containing judgement files (default: results/judgements)</div>
                <div><span className={isDark ? 'text-bb-accent' : 'text-bb-accent-dark'}>--output-dir</span> PATH &nbsp;&nbsp; Output directory (default: results/evaluations)</div>
              </div>
            </SubSection>

            <SubSection title="run_scoring_judge.py">
              <p>Run open-ended scoring evaluation (SummEval or MT-Bench) for a single judge model.</p>
              <div className="glass-card p-4 font-mono text-xs space-y-2 text-gray-400">
                <div><span className={isDark ? 'text-bb-accent' : 'text-bb-accent-dark'}>--judge</span> TEXT &nbsp;&nbsp; Judge model key from models.yaml (required)</div>
                <div><span className={isDark ? 'text-bb-accent' : 'text-bb-accent-dark'}>--dataset</span> TEXT &nbsp;&nbsp; Dataset to evaluate: <code>summeval</code> or <code>mtbench</code> (required)</div>
                <div><span className={isDark ? 'text-bb-accent' : 'text-bb-accent-dark'}>--max-tokens</span> INT &nbsp;&nbsp; Token budget for scoring (default: 8192)</div>
                <div><span className={isDark ? 'text-bb-accent' : 'text-bb-accent-dark'}>--output-dir</span> PATH &nbsp;&nbsp; Scoring results directory (default: results/scoring)</div>
                <div><span className={isDark ? 'text-bb-accent' : 'text-bb-accent-dark'}>--eval-dir</span> PATH &nbsp;&nbsp; Evaluation metrics directory (default: results/scoring_evaluations)</div>
                <div><span className={isDark ? 'text-bb-accent' : 'text-bb-accent-dark'}>--oracle-scores</span> PATH &nbsp;&nbsp; Path to oracle scores for MT-Bench (file or directory)</div>
                <div><span className={isDark ? 'text-bb-accent' : 'text-bb-accent-dark'}>--force</span> &nbsp;&nbsp; Re-score even if result files already exist</div>
              </div>
              <Callout type="tip">
                For MT-Bench, the model loads once and scores all oracle files automatically. Oracle files are auto-discovered from <code className={`font-mono ${isDark ? 'text-bb-accent' : 'text-bb-accent-dark'}`}>--oracle-scores</code> directory.
              </Callout>
            </SubSection>
          </Section>

          {/* ── Python API ── */}
          <Section active={activeSection === 'python-api'} id="python-api" icon={Code} title="Python API">
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
judge.save_results(results, "llama3.1-8b", "gsm8k", max_tokens=10)

# Cleanup GPU memory
judge.cleanup()`} />
                </div>

                <div className="glass-card p-4">
                  <div className={`text-sm font-semibold font-mono mb-2 ${isDark ? 'text-bb-accent' : 'text-bb-accent-dark'}`}>StudentSolver</div>
                  <p className="text-xs text-gray-500 mb-2">Runs student model inference to generate solutions for benchmark problems.</p>
                  <CodeBlock language="python" code={`from slmjury.core.solver import StudentSolver
from slmjury.data import load_dataset

solver = StudentSolver("llama3.1-8b", output_dir="results/student_solutions")
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
    judge_model="qwen3-4b",
    student_model="llama3.1-8b",
    dataset="gsm8k",
    max_tokens=10,
    judgements=results  # list of judgement dicts
)
summary = evaluator.evaluate()
print(f"Accuracy: {evaluator.accuracy:.2%}")
print(f"Instruction Following Rate: {evaluator.ifr:.2%}")`} />
                </div>
              </div>
            </SubSection>

            <SubSection title="Advanced Strategies">
              <CodeBlock language="python" code={`# Persona Effect
from slmjury.strategies.persona import run_persona_evaluation, get_personas
personas = get_personas()  # dict of 6 persona prompts
run_persona_evaluation(
    judge_model_key="qwen3-4b",
    student_results=student_results,
    student_model="qwen2.5-32b",
    dataset="gsm8k",
    max_tokens=10,
    persona_name="strict",
    persona_prompt=personas["strict"],
)

# Majority Voting Ensemble
from slmjury.strategies.ensemble import majority_vote, generate_all_ensembles
verdict = majority_vote(["Correct", "Incorrect", "Correct"])  # → "Correct"
generate_all_ensembles(judgements_dir="results/judgements",
                       output_dir="results/majority_voting")

# Multi-Agent Debate
from slmjury.strategies.debate import run_debate
run_debate(combo_models=["qwen3-4b", "phi4mi-3.8b", "qwen2.5-3b"],
           combo_temps=[0.0, 0.0, 0.0],
           student_results=student_results, dataset_name="gsm8k")`} />
            </SubSection>
          </Section>

          {/* ── Datasets ── */}
          <Section active={activeSection === 'datasets'} id="datasets" icon={Database} title="Supported Datasets">
            <SubSection title="Closed-Ended (Accuracy-Based)">
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
                    <tr className={`border-b ${isDark ? 'border-bb-dark-50/10' : 'border-gray-100'}`}><td className="py-1.5">GSM8K</td><td className="py-1.5">Grade School Math</td><td className="py-1.5">1,319</td><td className="py-1.5 font-mono">gsm8k</td></tr>
                    <tr className={`border-b ${isDark ? 'border-bb-dark-50/10' : 'border-gray-100'}`}><td className="py-1.5">GSM-Plus</td><td className="py-1.5">Extended Math</td><td className="py-1.5">10,552</td><td className="py-1.5 font-mono">gsm_plus</td></tr>
                    <tr className={`border-b ${isDark ? 'border-bb-dark-50/10' : 'border-gray-100'}`}><td className="py-1.5">MATH</td><td className="py-1.5">Competition Math</td><td className="py-1.5">5,000</td><td className="py-1.5 font-mono">math</td></tr>
                    <tr className={`border-b ${isDark ? 'border-bb-dark-50/10' : 'border-gray-100'}`}><td className="py-1.5">ARC-Easy</td><td className="py-1.5">Science (Easy)</td><td className="py-1.5">2,376</td><td className="py-1.5 font-mono">arc_easy</td></tr>
                    <tr className={`border-b ${isDark ? 'border-bb-dark-50/10' : 'border-gray-100'}`}><td className="py-1.5">ARC-Challenge</td><td className="py-1.5">Science (Hard)</td><td className="py-1.5">1,172</td><td className="py-1.5 font-mono">arc_challenge</td></tr>
                    <tr className={`border-b ${isDark ? 'border-bb-dark-50/10' : 'border-gray-100'}`}><td className="py-1.5">HellaSwag</td><td className="py-1.5">Commonsense NLI</td><td className="py-1.5">10,042</td><td className="py-1.5 font-mono">hellaswag</td></tr>
                    <tr className={`border-b ${isDark ? 'border-bb-dark-50/10' : 'border-gray-100'}`}><td className="py-1.5">WinoGrande</td><td className="py-1.5">Coreference</td><td className="py-1.5">1,267</td><td className="py-1.5 font-mono">winogrande</td></tr>
                    <tr><td className="py-1.5">TruthfulQA</td><td className="py-1.5">Factuality</td><td className="py-1.5">684</td><td className="py-1.5 font-mono">truthfulqa</td></tr>
                  </tbody>
                </table>
              </div>
            </SubSection>

            <SubSection title="Open-Ended (Correlation-Based)">
              <div className="glass-card p-4 text-xs">
                <table className="w-full">
                  <thead>
                    <tr className={`border-b ${isDark ? 'border-bb-dark-50/20' : 'border-gray-200'}`}>
                      <th className={`text-left py-2 ${isDark ? 'text-gray-500' : 'text-gray-400'}`}>Dataset</th>
                      <th className={`text-left py-2 ${isDark ? 'text-gray-500' : 'text-gray-400'}`}>Task</th>
                      <th className={`text-left py-2 ${isDark ? 'text-gray-500' : 'text-gray-400'}`}>Items</th>
                      <th className={`text-left py-2 ${isDark ? 'text-gray-500' : 'text-gray-400'}`}>Ground Truth</th>
                      <th className={`text-left py-2 ${isDark ? 'text-gray-500' : 'text-gray-400'}`}>Key</th>
                    </tr>
                  </thead>
                  <tbody className={isDark ? 'text-gray-400' : 'text-gray-600'}>
                    <tr className={`border-b ${isDark ? 'border-bb-dark-50/10' : 'border-gray-100'}`}><td className="py-1.5">SummEval</td><td className="py-1.5">Summary Quality (4 dims)</td><td className="py-1.5">1,600</td><td className="py-1.5">Human expert annotations</td><td className="py-1.5 font-mono">summeval</td></tr>
                    <tr><td className="py-1.5">MT-Bench</td><td className="py-1.5">Multi-turn QA (1–5 scale)</td><td className="py-1.5">80</td><td className="py-1.5">LLM oracle scores</td><td className="py-1.5 font-mono">mtbench</td></tr>
                  </tbody>
                </table>
              </div>
              <Callout type="info">
                Open-ended datasets use <strong>correlation metrics</strong> (Pearson, Spearman, Cohen's κ) instead of accuracy. SummEval measures agreement with human experts; MT-Bench measures agreement with large LLM oracles (GPT-OSS-120B, Qwen3.5-397B).
              </Callout>
            </SubSection>
          </Section>

          {/* ── Individual Judge Evaluation ── */}
          <Section active={activeSection === 'individual'} id="individual" icon={Cpu} title="Individual Judge Evaluation">
            <p>Each judge model independently evaluates student solutions under two <strong className={isDark ? 'text-gray-200' : 'text-gray-800'}>token budget settings</strong>:</p>
            <ul className="list-disc list-inside text-gray-500 space-y-1 text-xs mt-2">
              <li><strong>Concise (max_tokens=10)</strong> — forces a one-word verdict: "Correct" or "Incorrect"</li>
              <li><strong>Reasoned (max_tokens=8192)</strong> — allows chain-of-thought reasoning before a \boxed verdict</li>
            </ul>

            <SubSection title="Running Judge Evaluation">
              <CodeBlock code={`# Evaluate a single judge across all students & datasets
python scripts/run_judge.py --judge qwen3-4b --max-tokens 10 8192

# Evaluate with only the quick verdict mode
python scripts/run_judge.py --judge phi4mi-3.8b --max-tokens 10`} />
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
          <Section active={activeSection === 'persona'} id="persona" icon={UserCheck} title="Persona Effect">
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
          <Section active={activeSection === 'ensemble'} id="ensemble" icon={Users} title="Majority Voting Ensemble">
            <p>
              <strong className={isDark ? 'text-gray-200' : 'text-gray-800'}>C(5,3) = 10 combinations</strong> of the top 5 individual judges. Each problem is evaluated by 3 judge models and the majority verdict wins. This demonstrates that ensembles of small judges can exceed any individual judge's accuracy.
            </p>
            <SubSection title="Running Majority Voting">
              <CodeBlock language="python" code={`from slmjury.strategies.ensemble import majority_vote, generate_all_ensembles

# Generate all C(5,3)=10 ensemble combinations from pre-computed judgements
generate_all_ensembles(
    judgements_dir="results/judgements",
    output_dir="results/majority_voting"
)

# Or apply majority vote to individual verdicts
verdict = majority_vote(["Correct", "Incorrect", "Correct"])  # → "Correct"`} />
            </SubSection>
          </Section>

          {/* ── Multi-Agent Debate ── */}
          <Section active={activeSection === 'debate'} id="debate" icon={MessageSquare} title="Multi-Agent Debate">
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

          {/* ── Open-Ended Scoring ── */}
          <Section active={activeSection === 'open-ended'} id="open-ended" icon={BookOpenText} title="Open-Ended Scoring">
            <p>
              Beyond closed-ended accuracy, SLMJury also evaluates judges on <strong className={isDark ? 'text-gray-200' : 'text-gray-800'}>open-ended tasks</strong> where there is no single correct answer. Instead of accuracy, we measure <strong className={isDark ? 'text-bb-accent' : 'text-bb-accent-dark'}>correlation</strong> between SLM judge scores and ground truth scores (human or oracle LLM).
            </p>

            <SubSection title="SummEval: Human Agreement">
              <p>Each SLM judge scores 1,600 news summaries on <strong className={isDark ? 'text-gray-200' : 'text-gray-800'}>4 dimensions</strong> (1–5 scale): Coherence, Consistency, Fluency, and Relevance. These scores are compared against human expert annotations using correlation metrics.</p>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mt-3">
                {['Coherence', 'Consistency', 'Fluency', 'Relevance'].map(dim => (
                  <div key={dim} className="glass-card p-3">
                    <div className={`text-sm font-semibold mb-1 ${isDark ? 'text-gray-200' : 'text-gray-800'}`}>{dim}</div>
                    <div className="text-xs text-gray-500">{dim === 'Coherence' ? 'Well-organized and easy to follow' : dim === 'Consistency' ? 'Accurate to source without contradictions' : dim === 'Fluency' ? 'Grammatically correct and well-written' : 'Captures important information'}</div>
                  </div>
                ))}
              </div>
            </SubSection>

            <SubSection title="MT-Bench: LLM Agreement">
              <p>Each SLM judge scores 80 multi-turn conversations on a <strong className={isDark ? 'text-gray-200' : 'text-gray-800'}>1–5 holistic scale</strong>. Scores are compared against oracle LLM scores across 2 oracles × 2 students = <strong className={isDark ? 'text-bb-accent' : 'text-bb-accent-dark'}>4 oracle/student combinations</strong>.</p>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-3 mt-3">
                <div className="glass-card p-4">
                  <div className={`text-sm font-semibold mb-1 ${isDark ? 'text-gray-200' : 'text-gray-800'}`}>Oracle Models</div>
                  <div className="text-xs text-gray-500 font-mono">GPT-OSS-120B · Qwen3.5-397B</div>
                </div>
                <div className="glass-card p-4">
                  <div className={`text-sm font-semibold mb-1 ${isDark ? 'text-gray-200' : 'text-gray-800'}`}>Student Models</div>
                  <div className="text-xs text-gray-500 font-mono">Llama-3.1-8B · Qwen2.5-32B</div>
                </div>
              </div>
              <Callout type="info">
                MT-Bench questions span 8 categories: Coding, Extraction, Humanities, Math, Reasoning, Roleplay, STEM, and Writing. Per-category correlations are computed for fine-grained analysis.
              </Callout>
            </SubSection>

            <SubSection title="Running Open-Ended Scoring">
              <CodeBlock code={`# Score SummEval with a single judge
python scripts/run_scoring_judge.py \\
  --judge qwen3-4b --dataset summeval \\
  --tensor-parallel-size 1 --gpu-memory-utilization 0.95

# Score MT-Bench with a single judge (all oracle files auto-discovered)
python scripts/run_scoring_judge.py \\
  --judge qwen3-4b --dataset mtbench \\
  --oracle-scores results/mtbench_oracle/ \\
  --tensor-parallel-size 1

# Run all judges sequentially
for JUDGE in qwen3-4b phi4-14b qwen3-14b; do
  python scripts/run_scoring_judge.py --judge $JUDGE --dataset summeval --force
  python scripts/run_scoring_judge.py --judge $JUDGE --dataset mtbench --force
done`} />
            </SubSection>

            <SubSection title="Correlation Metrics">
              <div className="glass-card p-4 text-xs space-y-2">
                <div><span className={`font-mono font-semibold ${isDark ? 'text-bb-accent' : 'text-bb-accent-dark'}`}>Pearson r</span> — Linear correlation between judge and ground truth scores</div>
                <div><span className={`font-mono font-semibold ${isDark ? 'text-bb-accent' : 'text-bb-accent-dark'}`}>Spearman ρ</span> — Rank correlation (robust to non-linear relationships)</div>
                <div><span className={`font-mono font-semibold ${isDark ? 'text-bb-accent' : 'text-bb-accent-dark'}`}>Cohen's κ</span> — Binary agreement (scores ≥ 4 = GOOD, &lt; 4 = BAD)</div>
                <div><span className={`font-mono font-semibold ${isDark ? 'text-bb-accent' : 'text-bb-accent-dark'}`}>Accuracy</span> — Binary classification accuracy after thresholding at 4</div>
                <div><span className={`font-mono font-semibold ${isDark ? 'text-bb-accent' : 'text-bb-accent-dark'}`}>MSE</span> — Mean squared error (absolute score calibration)</div>
              </div>
            </SubSection>

            <SubSection title="Python API">
              <CodeBlock language="python" code={`from slmjury.core.scoring_judge import ScoringJudge
from slmjury.core.scoring_evaluator import evaluate_summeval, evaluate_mtbench
from slmjury.data import load_dataset

# Initialize scoring judge
judge = ScoringJudge("qwen3-4b", output_dir="results/scoring")

# Score SummEval (4-dimension scoring)
summeval_data = load_dataset("summeval")
results = judge.score_summeval(summeval_data, max_tokens=8192)
judge.save_results(results, "summeval")

# Score MT-Bench (holistic 1-5 scoring)
mtbench_data = load_dataset("mtbench")
results = judge.score_mtbench(mtbench_data, max_tokens=8192)
judge.save_results(results, "mtbench", suffix="gpt-oss-120b_llama3.1-8b")

# Evaluate correlations
eval_result = evaluate_summeval(results)
print(f"Overall Spearman: {eval_result['overall']['spearman']:.4f}")

judge.cleanup()`} />
            </SubSection>

            <SubSection title="Result Structure">
              <CodeBlock language="text" code={`results/
├── scoring/                          # Raw judge scores
│   └── {judge}/
│       ├── summeval.json             # 1600 scored summaries
│       ├── mtbench_gpt-oss-120b_llama3.1-8b.json    # 80 scored conversations
│       ├── mtbench_gpt-oss-120b_qwen2.5-32b.json
│       ├── mtbench_qwen3.5-397b_llama3.1-8b.json
│       └── mtbench_qwen3.5-397b_qwen2.5-32b.json
└── scoring_evaluations/              # Correlation metrics
    └── {judge}/
        ├── summeval_eval.json        # Per-dimension + overall correlations
        ├── mtbench_gpt-oss-120b_llama3.1-8b_eval.json
        ├── mtbench_gpt-oss-120b_qwen2.5-32b_eval.json
        ├── mtbench_qwen3.5-397b_llama3.1-8b_eval.json
        └── mtbench_qwen3.5-397b_qwen2.5-32b_eval.json`} />
            </SubSection>
          </Section>

          {/* ── Experimental Prompts ── */}
          <Section active={activeSection === 'prompts'} id="prompts" icon={FileText} title="Experimental Prompts">
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

            <SubSection title="SummEval Scoring Prompt">
              <CodeBlock language="text" code={`Evaluate the following summary of a news article on four dimensions.
Rate EACH dimension from 1 (very poor) to 5 (excellent).

Source Article:
{source_text}

Summary to Evaluate:
{summary}

Dimensions:
- Coherence: Is the summary well-organized and easy to follow?
- Consistency: Does the summary accurately reflect the source article without contradictions?
- Fluency: Is the summary grammatically correct and well-written?
- Relevance: Does the summary capture the important information from the source?

Provide your ratings as: \\\\boxed{coherence=X, consistency=X, fluency=X, relevance=X}`} />
            </SubSection>

            <SubSection title="MT-Bench Scoring Prompt">
              <CodeBlock language="text" code={`Rate the overall quality of the assistant's responses in the following 2-turn conversation on a scale of 1-5.
1 = Very poor, 2 = Poor, 3 = Average, 4 = Good, 5 = Excellent.

[Turn 1 Question]:
{turn1_question}

[Turn 1 Response]:
{turn1_response}

[Turn 2 Question]:
{turn2_question}

[Turn 2 Response]:
{turn2_response}

Consider helpfulness, accuracy, depth, creativity, and how well the assistant maintains context across both turns.
Provide your rating as: \\\\boxed{SCORE}`} />
              <Callout type="info">
                The MT-Bench scoring prompt is shared identically between <code className={`font-mono ${isDark ? 'text-bb-accent' : 'text-bb-accent-dark'}`}>ScoringJudge</code> and <code className={`font-mono ${isDark ? 'text-bb-accent' : 'text-bb-accent-dark'}`}>run_mtbench_oracle.py</code> to ensure fair SLM-LLM correlation measurement.
              </Callout>
            </SubSection>
          </Section>

          {/* ── Configuration ── */}
          <Section active={activeSection === 'configuration'} id="configuration" icon={Settings} title="Configuration">
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
          <Section active={activeSection === 'structure'} id="structure" icon={Layers} title="Project Structure">
            <CodeBlock language="text" code={`SLMJury/
├── slmjury/                  # Core framework package
│   ├── __init__.py           # Public API exports
│   ├── configs/              # Model + prompt configuration
│   │   ├── __init__.py       # load_models_config()
│   │   └── models.yaml       # Judge & student model definitions
│   ├── core/                 # Core evaluation logic
│   │   ├── evaluator.py      # JudgeEvaluator + accuracy metrics
│   │   ├── judge.py          # JudgeModel (vLLM-based inference)
│   │   ├── solver.py         # StudentSolver (student inference)
│   │   ├── scoring_judge.py  # ScoringJudge (open-ended scoring)
│   │   └── scoring_evaluator.py  # Correlation-based eval metrics
│   ├── data/                 # Dataset loaders
│   │   └── __init__.py       # load_dataset() for all benchmarks
│   ├── parsers/              # Output parsing
│   │   ├── answer.py         # Student answer parsing
│   │   ├── judgement.py      # Judge verdict parsing
│   │   └── score.py          # Score extraction (1-5, multi-dim)
│   └── strategies/           # Advanced strategies
│       ├── debate.py         # Multi-Agent Debate (MAD)
│       ├── ensemble.py       # Majority Voting
│       └── persona.py        # Persona Effect evaluation
├── scripts/                  # CLI entry-points
│   ├── run_student.py        # Generate student solutions
│   ├── run_judge.py          # Run judge evaluations
│   ├── run_evaluation.py     # Compute accuracy summaries
│   └── run_scoring_judge.py  # Open-ended scoring evaluation
├── results/                  # Experiment outputs
│   ├── student_solutions/    # Student model outputs
│   ├── judgements/           # Judge verdict files
│   ├── evaluations/          # Accuracy summaries
│   ├── scoring/              # Open-ended judge scores
│   └── scoring_evaluations/  # Open-ended correlation metrics
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
          <Section active={activeSection === 'extending'} id="extending" icon={GitBranch} title="Extending SLMJury">
            <SubSection title="Adding a New Judge Model">
              <ol className="list-decimal list-inside text-gray-500 space-y-2 text-xs">
                <li>Add the model entry to <code className={`font-mono ${isDark ? 'text-bb-accent' : 'text-bb-accent-dark'}`}>slmjury/configs/models.yaml</code> under <code className="font-mono">judge_models</code>.</li>
                <li>If the model has special tokenizer requirements, update <code className={`font-mono ${isDark ? 'text-bb-accent' : 'text-bb-accent-dark'}`}>slmjury/core/judge.py</code> (chat template handling).</li>
                <li>Run <code className={`font-mono ${isDark ? 'text-bb-accent' : 'text-bb-accent-dark'}`}>python scripts/run_judge.py --judge your-model-key</code> to evaluate.</li>
              </ol>
            </SubSection>

            <SubSection title="Adding a New Dataset">
              <ol className="list-decimal list-inside text-gray-500 space-y-2 text-xs">
                <li>Add a loader function in <code className={`font-mono ${isDark ? 'text-bb-accent' : 'text-bb-accent-dark'}`}>slmjury/data/__init__.py</code>.</li>
                <li>Add an answer parser in <code className={`font-mono ${isDark ? 'text-bb-accent' : 'text-bb-accent-dark'}`}>slmjury/parsers/answer.py</code>.</li>
                <li>Register the dataset key in the loader's dispatch map.</li>
                <li>Run <code className={`font-mono ${isDark ? 'text-bb-accent' : 'text-bb-accent-dark'}`}>python scripts/run_student.py --model your-model --datasets your-dataset</code>.</li>
              </ol>
            </SubSection>

            <SubSection title="Adding a New Persona">
              <p>Add your persona system prompt to the <code className={`font-mono ${isDark ? 'text-bb-accent' : 'text-bb-accent-dark'}`}>PERSONAS</code> dictionary in <code className="font-mono">slmjury/strategies/persona.py</code>. The persona evaluation pipeline will automatically pick it up.</p>
            </SubSection>
          </Section>

          {/* ── Troubleshooting ── */}
          <Section active={activeSection === 'troubleshooting'} id="troubleshooting" icon={Wrench} title="Troubleshooting">
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
          <Section active={activeSection === 'citation'} id="citation" icon={BarChart3} title="Citation">
            <CodeBlock language="bibtex" code={`@article{laddha2025slmjury,
  title   = {SLMJury: Can Small Language Models Judge as Well as Large Language Models?},
  author  = {Anish Laddha and Nitesh Pradhan and Gaurav Srivastava},
  year    = {2025},
  note    = {Manuscript in preparation}
}`} />
          </Section>
          </div>
        </div>
      </div>
    </section>
  )
}
