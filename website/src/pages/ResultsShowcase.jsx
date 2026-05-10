import { useState, useMemo } from 'react'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, Cell,
  RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis,
  ScatterChart, Scatter, ZAxis,
} from 'recharts'
import { BarChart3, Target, Zap, TrendingUp, ChevronDown, X, Plus, Layers, Scale } from 'lucide-react'
import { modelData, majorityVotingData, personaData, madData, summevalData, mtbenchData } from '../data/modelData'
import { useTheme } from '../context/ThemeContext'

const MODEL_COLORS = ['#00AAFF', '#6366F1', '#a855f7', '#06b6d4', '#f59e0b', '#ef4444']

function shortName(name) {
  const s = name.replace(/^[^/]+\//, '').replace(/-(?:Instruct|instruct|chat|Chat)$/, '')
  return s.length > 18 ? s.slice(0, 16) + '…' : s
}

function parseParams(p) {
  if (!p) return 0
  const n = parseFloat(p)
  return isNaN(n) ? 0 : n
}

function ChartTooltip({ active, payload, label }) {
  const { isDark } = useTheme()
  if (!active || !payload?.length) return null
  return (
    <div className={`px-3 py-2 rounded-lg border text-xs shadow-xl ${
      isDark ? 'bg-bb-dark-300/95 border-bb-dark-50/30 text-white' : 'bg-white/95 border-bb-light-300 text-gray-900'
    }`}>
      {label && <p className="font-semibold mb-1 truncate max-w-[220px]">{label}</p>}
      {payload.map((p, i) => (
        <p key={i} style={{ color: p.fill || p.color || p.stroke }}>
          {p.name}: <span className="font-mono font-bold">{typeof p.value === 'number' ? p.value.toFixed(1) + '%' : p.value}</span>
        </p>
      ))}
    </div>
  )
}

function ScatterTip({ active, payload }) {
  const { isDark } = useTheme()
  if (!active || !payload?.length) return null
  const d = payload[0]?.payload
  if (!d) return null
  return (
    <div className={`px-3 py-2 rounded-lg border text-xs shadow-xl max-w-[220px] ${
      isDark ? 'bg-bb-dark-300/95 border-bb-dark-50/30 text-white' : 'bg-white/95 border-bb-light-300 text-gray-900'
    }`}>
      <p className="font-semibold truncate">{d.shortModel || d.label}</p>
      <p className={isDark ? 'text-gray-400' : 'text-gray-500'}>Accuracy: <span className="font-mono font-bold">{d.accuracy?.toFixed(1)}%</span></p>
      <p className={isDark ? 'text-gray-400' : 'text-gray-500'}>Params: <span className="font-mono font-bold">{d.params}</span></p>
    </div>
  )
}

function SectionHeader({ icon: Icon, title, subtitle, isDark }) {
  return (
    <div className="flex items-start gap-4 mb-6">
      <div className={`p-3 rounded-xl flex-shrink-0 ${isDark ? 'bg-bb-accent/10' : 'bg-bb-accent-dark/10'}`}>
        <Icon className={`w-6 h-6 ${isDark ? 'text-bb-accent' : 'text-bb-accent-dark'}`} />
      </div>
      <div>
        <h2 className={`text-xl font-bold ${isDark ? 'text-white' : 'text-gray-900'}`}>{title}</h2>
        <p className={`text-sm mt-0.5 ${isDark ? 'text-gray-400' : 'text-gray-500'}`}>{subtitle}</p>
      </div>
    </div>
  )
}

function Card({ children, isDark, className = '' }) {
  return (
    <div className={`rounded-2xl border p-6 ${
      isDark ? 'bg-bb-dark-300/60 backdrop-blur-xl border-bb-dark-50/30' : 'bg-white/70 backdrop-blur-xl border-bb-light-300/60 shadow-sm'
    } ${className}`}>
      {children}
    </div>
  )
}

const DATASETS = [
  { key: 'gsm8k_acc', label: 'GSM8K' },
  { key: 'gsm_plus_acc', label: 'GSM+' },
  { key: 'math_acc', label: 'MATH' },
  { key: 'arc_easy_acc', label: 'ARC-E' },
  { key: 'arc_challenge_acc', label: 'ARC-C' },
  { key: 'hellaswag_acc', label: 'Hella' },
  { key: 'winogrande_acc', label: 'Wino' },
  { key: 'truthfulqa_acc', label: 'TQA' },
]

export default function ResultsShowcase() {
  const { isDark } = useTheme()
  const axisStyle = { fill: isDark ? '#9ca3af' : '#6b7280', fontSize: 10 }
  const gridColor = isDark ? 'rgba(255,255,255,0.07)' : 'rgba(0,0,0,0.08)'

  // ── Chart 1: Grouped bar — t=10 vs t=8192 ──
  const pairedData = useMemo(() => {
    const map = {}
    modelData.forEach(m => {
      const base = m.shortName.replace(/-t\d+$/, '')
      if (!map[base]) map[base] = { name: base }
      if (m.tokens === 10) map[base].concise = m.accuracy
      else if (m.tokens === 8192) map[base].reasoned = m.accuracy
    })
    return Object.values(map)
      .filter(d => d.concise != null || d.reasoned != null)
      .sort((a, b) => Math.max(b.concise || 0, b.reasoned || 0) - Math.max(a.concise || 0, a.reasoned || 0))
      .slice(0, 16)
  }, [])

  // ── Chart 2: Radar — Compare Judges ──
  const [selectedModels, setSelectedModels] = useState(() =>
    [...modelData].sort((a, b) => b.accuracy - a.accuracy).slice(0, 3).map(m => m.shortName)
  )
  const [selectorOpen, setSelectorOpen] = useState(false)
  const [searchQuery, setSearchQuery] = useState('')

  const filteredOptions = useMemo(() => {
    const q = searchQuery.toLowerCase()
    return modelData.filter(m => m.shortName.toLowerCase().includes(q) || m.model.toLowerCase().includes(q)).slice(0, 20)
  }, [searchQuery])

  const addModel = (key) => {
    if (selectedModels.length >= 5 || selectedModels.includes(key)) return
    setSelectedModels(prev => [...prev, key])
    setSelectorOpen(false)
    setSearchQuery('')
  }
  const removeModel = (key) => setSelectedModels(prev => prev.filter(m => m !== key))

  const radarData = useMemo(() => {
    return DATASETS.map(({ key, label }) => {
      const row = { metric: label }
      selectedModels.forEach(sn => {
        const m = modelData.find(d => d.shortName === sn)
        if (m) row[sn] = +(m[key] || 0).toFixed(1)
      })
      return row
    })
  }, [selectedModels])

  // ── Chart 3: Model Size vs Accuracy ──
  const scatterData = useMemo(() => {
    return modelData.map(m => ({
      ...m,
      paramNum: parseParams(m.params),
      label: `${m.shortName} (t=${m.tokens})`,
    }))
  }, [])

  // ── Chart 4: Strategy Comparison (Max per strategy) ──
  const strategyData = useMemo(() => {
    const maxInd = Math.max(...modelData.map(m => m.accuracy))
    const maxPersona = Math.max(...personaData.map(m => m.accuracy))
    const maxMV = Math.max(...majorityVotingData.map(m => m.accuracy))
    const maxMAD = Math.max(...madData.map(m => m.accuracy))
    return [
      { name: 'Individual', accuracy: maxInd },
      { name: 'Persona', accuracy: maxPersona },
      { name: 'Majority Voting', accuracy: maxMV },
      { name: 'MAD', accuracy: maxMAD },
    ]
  }, [])

  const strategyBest = useMemo(() => {
    const bestInd = [...modelData].sort((a, b) => b.accuracy - a.accuracy)[0]
    const bestPe = [...personaData].sort((a, b) => b.accuracy - a.accuracy)[0]
    const bestMV = [...majorityVotingData].sort((a, b) => b.accuracy - a.accuracy)[0]
    const bestMAD = [...madData].sort((a, b) => b.accuracy - a.accuracy)[0]
    return {
      'Individual': bestInd?.shortName || bestInd?.model,
      'Persona': bestPe?.shortName || bestPe?.model,
      'Majority Voting': bestMV?.ensemble_name?.split('+').map(s => s.replace(/-t\d+$/, '')).join(' + '),
      'MAD': bestMAD?.ensemble_name?.split('+').map(s => s.replace(/-t[\d.]+$/, '')).join(' + '),
    }
  }, [])

  // ── Chart 5: Open-Ended Correlation ──
  const openEndedData = useMemo(() => {
    const map = {}
    summevalData.forEach(m => {
      map[m.shortName] = { name: m.shortName, summeval: m.spearman }
    })
    mtbenchData.forEach(m => {
      if (map[m.shortName]) map[m.shortName].mtbench = m.spearman
      else map[m.shortName] = { name: m.shortName, mtbench: m.spearman }
    })
    return Object.values(map)
      .filter(d => d.summeval != null && d.mtbench != null)
      .sort((a, b) => ((b.summeval || 0) + (b.mtbench || 0)) - ((a.summeval || 0) + (a.mtbench || 0)))
  }, [])

  // ── Summary Stats ──
  const stats = useMemo(() => ({
    models: modelData.length,
    best: Math.max(...modelData.map(m => m.accuracy)).toFixed(1),
    avg: (modelData.reduce((s, m) => s + m.accuracy, 0) / modelData.length).toFixed(1),
    datasets: 10,
  }), [])

  function StrategyTooltip({ active, payload, label }) {
    if (!active || !payload?.length) return null
    return (
      <div className={`px-3 py-2 rounded-lg border text-xs shadow-xl ${
        isDark ? 'bg-bb-dark-300/95 border-bb-dark-50/30 text-white' : 'bg-white/95 border-bb-light-300 text-gray-900'
      }`}>
        <p className="font-semibold mb-1">{label}</p>
        <p style={{ color: payload[0]?.fill }}>Max Accuracy: <span className="font-mono font-bold">{payload[0]?.value?.toFixed(2)}%</span></p>
        {strategyBest[label] && <p className={isDark ? 'text-gray-400' : 'text-gray-500'}>Best: {strategyBest[label]}</p>}
      </div>
    )
  }

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
      {/* Page header */}
      <div className="mb-10">
        <div className="flex items-center gap-3 mb-3">
          <div className={`p-2 rounded-xl ${isDark ? 'bg-bb-accent/10' : 'bg-bb-accent-dark/10'}`}>
            <BarChart3 className={`w-7 h-7 ${isDark ? 'text-bb-accent' : 'text-bb-accent-dark'}`} />
          </div>
          <h1 className={`text-3xl font-bold ${isDark ? 'text-white' : 'text-gray-900'}`}>
            Results Showcase
          </h1>
        </div>
        <p className={`text-base max-w-2xl ${isDark ? 'text-gray-400' : 'text-gray-600'}`}>
          Interactive visualizations comparing 16 judge models across accuracy, model size, evaluation strategies, and open-ended correlation metrics.
        </p>
      </div>

      {/* ── Chart 1: Top Judges Grouped Bar ── */}
      <Card isDark={isDark} className="mb-8">
        <SectionHeader
          icon={TrendingUp}
          title="Concise vs. Reasoned Evaluation"
          subtitle="Judge accuracy at max_tokens=10 (concise) vs. max_tokens=8192 (reasoned)"
          isDark={isDark}
        />
        <div className="h-80">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={pairedData} margin={{ top: 5, right: 20, left: 0, bottom: 80 }}>
              <CartesianGrid strokeDasharray="3 3" stroke={gridColor} />
              <XAxis dataKey="name" tick={axisStyle} angle={-45} textAnchor="end" interval={0} height={90} />
              <YAxis tick={axisStyle} domain={[40, 100]} tickFormatter={v => `${v}%`} width={45} />
              <Tooltip content={<ChartTooltip />} />
              <Legend wrapperStyle={{ fontSize: 12, color: isDark ? '#9ca3af' : '#6b7280', paddingTop: 8 }} />
              <Bar dataKey="concise" name="Concise (t=10)" fill="#00AAFF" radius={[3, 3, 0, 0]} />
              <Bar dataKey="reasoned" name="Reasoned (t=8192)" fill="#6366F1" radius={[3, 3, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </Card>

      {/* ── Chart 2: Radar — Compare Judges ── */}
      <Card isDark={isDark} className="mb-8">
        <SectionHeader
          icon={Target}
          title="Compare Judges Across Datasets"
          subtitle="Select up to 5 judges to compare their per-dataset accuracy profiles"
          isDark={isDark}
        />
        {/* Selected model chips */}
        <div className="flex flex-wrap gap-2 mb-4">
          {selectedModels.map((sn, idx) => (
            <span key={sn} className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-medium border"
              style={{ borderColor: MODEL_COLORS[idx % MODEL_COLORS.length] + '60', backgroundColor: MODEL_COLORS[idx % MODEL_COLORS.length] + '15', color: MODEL_COLORS[idx % MODEL_COLORS.length] }}>
              <span className="max-w-[160px] truncate">{sn}</span>
              <button onClick={() => removeModel(sn)} className="hover:opacity-70 transition-opacity"><X className="w-3 h-3" /></button>
            </span>
          ))}
          {selectedModels.length < 5 && (
            <div className="relative">
              <button onClick={() => setSelectorOpen(o => !o)}
                className={`inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-medium border transition-all ${
                  isDark ? 'border-bb-dark-50/40 text-gray-400 hover:border-bb-accent/40 hover:text-bb-accent' : 'border-bb-light-300 text-gray-500 hover:border-bb-accent-dark/40 hover:text-bb-accent-dark'
                }`}>
                <Plus className="w-3 h-3" /> Add model <ChevronDown className={`w-3 h-3 transition-transform ${selectorOpen ? 'rotate-180' : ''}`} />
              </button>
              {selectorOpen && (
                <div className={`absolute top-full left-0 mt-1 z-30 w-72 rounded-xl border shadow-2xl overflow-hidden ${
                  isDark ? 'bg-bb-dark-300 border-bb-dark-50/30' : 'bg-white border-bb-light-300'
                }`}>
                  <div className={`p-2 border-b ${isDark ? 'border-bb-dark-50/20' : 'border-bb-light-300/50'}`}>
                    <input autoFocus type="text" placeholder="Search models..." value={searchQuery} onChange={e => setSearchQuery(e.target.value)}
                      className={`w-full px-3 py-1.5 rounded-lg text-xs outline-none ${
                        isDark ? 'bg-bb-dark-500/50 text-white placeholder-gray-600 border border-bb-dark-50/20' : 'bg-bb-light-100 text-gray-900 placeholder-gray-400 border border-bb-light-300'
                      }`} />
                  </div>
                  <div className="max-h-48 overflow-y-auto">
                    {filteredOptions.map(m => (
                      <button key={m.shortName} onClick={() => addModel(m.shortName)} disabled={selectedModels.includes(m.shortName)}
                        className={`w-full text-left px-3 py-2 text-xs transition-colors ${
                          selectedModels.includes(m.shortName)
                            ? isDark ? 'text-gray-600 cursor-not-allowed' : 'text-gray-300 cursor-not-allowed'
                            : isDark ? 'text-gray-300 hover:bg-bb-dark-500/50 hover:text-white' : 'text-gray-700 hover:bg-bb-light-100 hover:text-gray-900'
                        }`}>
                        <span className="block truncate">{m.shortName} <span className="opacity-60">({m.params}, t={m.tokens})</span></span>
                        <span className={`text-[10px] ${isDark ? 'text-gray-500' : 'text-gray-400'}`}>Acc: {m.accuracy.toFixed(1)}%</span>
                      </button>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
        {selectedModels.length > 0 ? (
          <div className="h-80">
            <ResponsiveContainer width="100%" height="100%">
              <RadarChart data={radarData} margin={{ top: 10, right: 40, bottom: 10, left: 40 }}>
                <PolarGrid stroke={gridColor} />
                <PolarAngleAxis dataKey="metric" tick={{ fill: isDark ? '#d1d5db' : '#374151', fontSize: 12, fontWeight: 500 }} />
                <PolarRadiusAxis angle={30} domain={[0, 100]} tick={{ fill: isDark ? '#6b7280' : '#9ca3af', fontSize: 10 }} tickFormatter={v => `${v}%`} />
                {selectedModels.map((sn, idx) => (
                  <Radar key={sn} name={sn} dataKey={sn} stroke={MODEL_COLORS[idx % MODEL_COLORS.length]} fill={MODEL_COLORS[idx % MODEL_COLORS.length]} fillOpacity={0.08} strokeWidth={2} />
                ))}
                <Legend wrapperStyle={{ fontSize: 11, color: isDark ? '#9ca3af' : '#6b7280' }} />
                <Tooltip content={<ChartTooltip />} />
              </RadarChart>
            </ResponsiveContainer>
          </div>
        ) : (
          <div className={`h-80 flex items-center justify-center rounded-xl border-2 border-dashed ${isDark ? 'border-bb-dark-50/20 text-gray-600' : 'border-bb-light-300 text-gray-400'}`}>
            <p className="text-sm">Add at least one model to see the radar chart</p>
          </div>
        )}
      </Card>

      {/* ── Chart 3: Model Size vs Accuracy ── */}
      <Card isDark={isDark} className="mb-8">
        <SectionHeader
          icon={Scale}
          title="Model Size vs. Judging Accuracy"
          subtitle="Can smaller models judge as well as larger ones? Each point is a judge configuration."
          isDark={isDark}
        />
        <div className="h-96">
          <ResponsiveContainer width="100%" height="100%">
            <ScatterChart margin={{ top: 10, right: 20, left: 0, bottom: 30 }}>
              <CartesianGrid strokeDasharray="3 3" stroke={gridColor} />
              <XAxis dataKey="paramNum" name="Params (B)" type="number" tick={axisStyle}
                label={{ value: 'Model Size (B params)', position: 'insideBottom', offset: -15, fill: isDark ? '#9ca3af' : '#6b7280', fontSize: 11 }} />
              <YAxis dataKey="accuracy" name="Accuracy" type="number" domain={[40, 100]} tick={axisStyle} tickFormatter={v => `${v}%`}
                label={{ value: 'Overall Accuracy (%)', angle: -90, position: 'insideLeft', offset: 15, fill: isDark ? '#9ca3af' : '#6b7280', fontSize: 11 }} />
              <ZAxis range={[40, 150]} />
              <Tooltip content={<ScatterTip />} />
              <Scatter data={scatterData} fill="#00AAFF" fillOpacity={0.7} stroke={isDark ? '#00AAFF80' : '#006EB880'} strokeWidth={1} />
            </ScatterChart>
          </ResponsiveContainer>
        </div>
        <p className={`text-xs mt-2 ${isDark ? 'text-gray-500' : 'text-gray-400'}`}>
          * Models under 4B parameters can achieve 85%+ accuracy, demonstrating that small judges are viable.
        </p>
      </Card>

      {/* ── Chart 4: Strategy Comparison ── */}
      <Card isDark={isDark} className="mb-8">
        <SectionHeader
          icon={Layers}
          title="Best Accuracy by Evaluation Strategy"
          subtitle="Maximum accuracy achieved by each evaluation paradigm — which strategy produces the best judges?"
          isDark={isDark}
        />
        <div className="h-80">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={strategyData} margin={{ top: 5, right: 20, left: 0, bottom: 20 }}>
              <CartesianGrid strokeDasharray="3 3" stroke={gridColor} />
              <XAxis dataKey="name" tick={axisStyle} />
              <YAxis tick={axisStyle} domain={[80, 100]} tickFormatter={v => `${v}%`} width={45} />
              <Tooltip content={<StrategyTooltip />} />
              <Bar dataKey="accuracy" name="Max Accuracy" radius={[6, 6, 0, 0]}>
                {strategyData.map((entry, idx) => {
                  const colors = ['#00AAFF', '#a855f7', '#06b6d4', '#f59e0b']
                  return <Cell key={idx} fill={colors[idx]} />
                })}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </Card>

      {/* ── Chart 5: Open-Ended Correlation ── */}
      <Card isDark={isDark} className="mb-8">
        <SectionHeader
          icon={Zap}
          title="Open-Ended Correlation: SummEval vs. MT-Bench"
          subtitle="Spearman ρ for each judge on human agreement (SummEval) vs. LLM agreement (MT-Bench)"
          isDark={isDark}
        />
        <div className="h-80">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={openEndedData} margin={{ top: 5, right: 20, left: 0, bottom: 80 }}>
              <CartesianGrid strokeDasharray="3 3" stroke={gridColor} />
              <XAxis dataKey="name" tick={axisStyle} angle={-45} textAnchor="end" interval={0} height={90} />
              <YAxis tick={axisStyle} domain={[-0.1, 0.7]} tickFormatter={v => v.toFixed(1)} width={40} />
              <Tooltip content={<ChartTooltip />} />
              <Legend wrapperStyle={{ fontSize: 11, color: isDark ? '#9ca3af' : '#6b7280', paddingTop: 8 }} />
              <Bar dataKey="summeval" name="SummEval (ρ)" fill="#00AAFF" radius={[3, 3, 0, 0]} />
              <Bar dataKey="mtbench" name="MT-Bench (ρ)" fill="#6366F1" radius={[3, 3, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
        <p className={`text-xs mt-2 ${isDark ? 'text-gray-500' : 'text-gray-400'}`}>
          * SummEval measures human agreement on summarization quality. MT-Bench measures LLM oracle agreement on multi-turn dialogue.
        </p>
      </Card>

      {/* Summary stats row */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
        {[
          { label: 'Judge Configurations', value: stats.models + '' },
          { label: 'Best Accuracy', value: stats.best + '%' },
          { label: 'Avg Accuracy', value: stats.avg + '%' },
          { label: 'Benchmarks', value: stats.datasets + '' },
        ].map(({ label, value }) => (
          <div key={label} className={`rounded-xl p-4 text-center border ${
            isDark ? 'bg-bb-dark-300/40 border-bb-dark-50/20' : 'bg-white/50 border-bb-light-300/50 shadow-sm'
          }`}>
            <div className={`text-2xl font-bold font-mono ${isDark ? 'text-bb-accent' : 'text-bb-accent-dark'}`}>{value}</div>
            <div className={`text-xs mt-1 ${isDark ? 'text-gray-500' : 'text-gray-400'}`}>{label}</div>
          </div>
        ))}
      </div>
    </div>
  )
}
