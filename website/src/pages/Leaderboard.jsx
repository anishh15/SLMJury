import { useState, useMemo, useEffect } from 'react'
import { Trophy, Search, ChevronDown, ChevronUp, Filter, Brain, Target, Crown, Medal, Star, ArrowUpRight, Users, Database, Layers, UserCheck, Scale, MessageSquare, FlaskConical } from 'lucide-react'
import { modelData, majorityVotingData, personaData, madData, PERSONAS, projectInfo } from '../data/modelData'
import { useTheme } from '../context/ThemeContext'

/* ─── Animations ────────────────────────────────────────────────── */

function AnimateIn({ children, delay = 0, className = '', direction = 'up' }) {
  const [show, setShow] = useState(false)
  useEffect(() => { const t = setTimeout(() => setShow(true), delay); return () => clearTimeout(t) }, [delay])
  const transforms = { up: 'translate-y-8', scale: 'scale-95' }
  return (
    <div className={`transition-all duration-700 ease-out ${show ? 'opacity-100 translate-y-0 scale-100' : `opacity-0 ${transforms[direction] || transforms.up}`} ${className}`}>
      {children}
    </div>
  )
}

/* ─── Constants ─────────────────────────────────────────────────── */

const FAMILY_COLORS = { 'Qwen': '#7c3aed', 'Microsoft': '#f59e0b', 'Meta': '#22c55e' }

const PERSONA_LABELS = {
  helpfulness: 'Helpful', industry: 'Industry', lenient: 'Lenient',
  logic: 'Logic', safety: 'Safety', strict: 'Strict',
}

const TABS = [
  { id: 'phase1', label: 'Individual Judges', icon: Brain },
  { id: 'majority', label: 'Majority Voting', icon: Users },
  { id: 'persona', label: 'Persona Effect', icon: UserCheck },
  { id: 'mad', label: 'Multi-Agent Debate', icon: MessageSquare },
]

/* ─── Helpers ───────────────────────────────────────────────────── */

/** Format a number to exactly 2 decimal places */
const fmt = (v) => v != null ? v.toFixed(2) : '—'

function getUniqueKey(m) { return `${m.shortName}_t${m.tokens}` }

/** Build Phase 1 accuracy lookup: "shortName_tTokens" → accuracy */
const phase1AccMap = (() => {
  const map = {}
  modelData.forEach(m => { map[`${m.shortName}_t${m.tokens}`] = m.accuracy })
  return map
})()

/**
 * For an ensemble entry, compute the best individual Phase 1 accuracy
 * among the constituent judges — used as the comparison baseline.
 * MV judges have `tokens` → exact key_tTokens lookup.
 * MAD judges have `temperature` instead → all MAD runs are Reasoned
 * (max_tokens = 8192) so we compare against t=8192 Phase 1 accuracy.
 */
function bestComboBase(m) {
  if (!m.judges || m.judges.length === 0) return null
  let best = -Infinity
  const seen = new Set()
  for (const j of m.judges) {
    if (j.tokens != null) {
      // MV: exact token-setting lookup
      const acc = phase1AccMap[`${j.key}_t${j.tokens}`]
      if (acc != null && acc > best) best = acc
    } else {
      // MAD: all agents are Reasoned (t=8192), compare with t=8192 only
      if (!seen.has(j.key)) {
        seen.add(j.key)
        const acc = phase1AccMap[`${j.key}_t8192`]
        if (acc != null && acc > best) best = acc
      }
    }
  }
  return best === -Infinity ? null : best
}

/* ─── Delta Component (with numeric value) ──────────────────────── */

function Delta({ value, base }) {
  if (base == null || value == null) return null
  const diff = value - base
  if (Math.abs(diff) < 0.005) return null
  const isUp = diff > 0
  return (
    <span className={`inline-flex items-center ml-1 text-[9px] font-semibold whitespace-nowrap ${isUp ? 'text-emerald-400' : 'text-red-400'}`}>
      ({isUp ? '+' : ''}{diff.toFixed(2)})
    </span>
  )
}

/* ─── Shared Components ─────────────────────────────────────────── */

function StatCard({ icon: Icon, label, value, sub }) {
  const { isDark } = useTheme()
  return (
    <div className={`group relative overflow-hidden rounded-xl p-5 flex flex-col items-center text-center transition-all duration-300 hover:scale-[1.03] hover:-translate-y-1 ${isDark ? 'bg-bb-dark-300/60 backdrop-blur-xl border border-bb-dark-50/30 hover:border-bb-accent/30'
      : 'bg-white/70 backdrop-blur-xl border border-bb-light-300/60 hover:border-bb-accent-dark/40 shadow-sm hover:shadow-lg'
      }`}>
      <div className={`absolute inset-0 opacity-0 group-hover:opacity-100 transition-opacity duration-500 ${isDark ? 'bg-gradient-to-br from-bb-accent/5 via-transparent to-bb-teal/5' : 'bg-gradient-to-br from-bb-accent-dark/5 via-transparent to-bb-teal/5'
        }`} />
      <div className={`relative w-11 h-11 rounded-xl flex items-center justify-center mb-3 transition-all duration-300 group-hover:scale-110 ${isDark ? 'bg-bb-accent/10 group-hover:bg-bb-accent/20' : 'bg-bb-accent-dark/10 group-hover:bg-bb-accent-dark/15'}`}>
        <Icon className={`w-5 h-5 ${isDark ? 'text-bb-accent' : 'text-bb-accent-dark'}`} />
      </div>
      <div className={`text-2xl font-bold font-mono relative ${isDark ? 'text-white' : 'text-gray-900'}`}>{value}</div>
      <div className={`text-xs mt-1.5 font-medium ${isDark ? 'text-gray-500' : 'text-gray-500'}`}>{label}</div>
      {sub && <div className={`text-[10px] mt-0.5 ${isDark ? 'text-gray-600' : 'text-gray-400'}`}>{sub}</div>}
    </div>
  )
}

/* ─── Model Name Cell (shared styling across all tabs) ──────────── */

function ModelNameCell({ name, family, params, isDark }) {
  const familyColor = FAMILY_COLORS[family] || '#6b7280'
  return (
    <div className="flex items-center gap-2 whitespace-nowrap">
      <div className="w-2 h-2 rounded-full shrink-0 ring-2 ring-offset-1" style={{ backgroundColor: familyColor, ringColor: familyColor + '40', ringOffsetColor: isDark ? '#0d131b' : '#ffffff' }} />
      <div className="min-w-0">
        <div className={`text-[13px] font-medium truncate ${isDark ? 'text-gray-200' : 'text-gray-800'}`}>{name}</div>
        <div className={`text-[10px] ${isDark ? 'text-gray-600' : 'text-gray-400'}`}>{family} · {params}</div>
      </div>
    </div>
  )
}

function PodiumCard({ model, index }) {
  const { isDark } = useTheme()
  const configs = [
    { medal: <Crown className="w-6 h-6" />, color: 'text-yellow-400', bg: isDark ? 'from-yellow-400/10 via-yellow-400/5 to-transparent' : 'from-yellow-400/15 via-yellow-400/5 to-transparent', border: isDark ? 'border-yellow-400/30 hover:border-yellow-400/50' : 'border-yellow-400/40 hover:border-yellow-400/60', glow: 'shadow-[0_0_30px_rgba(250,204,21,0.1)]' },
    { medal: <Medal className="w-5 h-5" />, color: 'text-gray-300', bg: isDark ? 'from-gray-300/10 via-gray-300/5 to-transparent' : 'from-gray-400/10 via-gray-400/5 to-transparent', border: isDark ? 'border-gray-400/20 hover:border-gray-400/40' : 'border-gray-400/30 hover:border-gray-400/50', glow: '' },
    { medal: <Star className="w-5 h-5" />, color: 'text-amber-600', bg: isDark ? 'from-amber-600/10 via-amber-600/5 to-transparent' : 'from-amber-600/10 via-amber-600/5 to-transparent', border: isDark ? 'border-amber-600/20 hover:border-amber-600/40' : 'border-amber-600/30 hover:border-amber-600/50', glow: '' },
  ]
  const c = configs[index]
  const m = model
  const displayName = m.model.split('/')[1] || m.model
  const subtitle = `${m.family} · ${m.params} · t=${m.tokens}`

  return (
    <div className={`group relative overflow-hidden rounded-xl border p-5 transition-all duration-300 hover:scale-[1.02] hover:-translate-y-1 ${c.border} ${c.glow} ${isDark ? 'bg-bb-dark-300/60 backdrop-blur-xl' : 'bg-white/80 backdrop-blur-xl shadow-sm hover:shadow-lg'}`}>
      <div className={`absolute inset-0 bg-gradient-to-br ${c.bg} opacity-60`} />
      <div className={`absolute top-0 right-0 w-24 h-24 bg-gradient-to-bl ${c.bg} rounded-bl-full opacity-40`} />
      <div className="relative">
        <div className="flex items-start gap-3">
          <div className={`flex items-center justify-center w-10 h-10 rounded-lg ${isDark ? 'bg-bb-dark-400/60' : 'bg-white/60'}`}>
            <span className={c.color}>{c.medal}</span>
          </div>
          <div className="flex-1 min-w-0">
            <div className={`text-sm font-semibold truncate ${isDark ? 'text-white' : 'text-gray-900'}`}>{displayName}</div>
            <div className={`text-xs truncate ${isDark ? 'text-gray-500' : 'text-gray-400'}`}>{subtitle}</div>
          </div>
        </div>
        <div className="mt-4 grid grid-cols-3 gap-3">
          {[
            { val: m.accuracy, label: 'Overall', accent: true },
            { val: m.gsm8k_acc, label: 'GSM8K' },
            { val: m.math_acc, label: 'MATH' },
          ].map(({ val, label, accent }) => (
            <div key={label} className={`text-center p-2 rounded-lg ${isDark ? 'bg-bb-dark-400/40' : 'bg-bb-light-200/60'}`}>
              <div className={`text-lg font-bold font-mono ${accent ? (isDark ? 'text-bb-accent' : 'text-bb-accent-dark') : (isDark ? 'text-gray-300' : 'text-gray-700')}`}>{fmt(val)}%</div>
              <div className={`text-[10px] font-medium ${isDark ? 'text-gray-600' : 'text-gray-400'}`}>{label}</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}

/* ─── Sort Header ───────────────────────────────────────────────── */

function SortHeaderCell({ k, sortKey, sortDir, toggleSort, isDark, children, className = '' }) {
  const active = sortKey === k
  return (
    <th className={`px-2 py-3 text-[11px] font-semibold uppercase tracking-wider cursor-pointer select-none transition-colors whitespace-nowrap ${active ? (isDark ? 'text-bb-accent' : 'text-bb-accent-dark') : (isDark ? 'text-gray-500 hover:text-gray-300' : 'text-gray-400 hover:text-gray-600')
      } ${className}`} onClick={() => toggleSort(k)}>
      <div className="flex items-center gap-1">
        {children}
        {active && (sortDir === 'desc' ? <ChevronDown className="w-3 h-3" /> : <ChevronUp className="w-3 h-3" />)}
      </div>
    </th>
  )
}

/* ─── Detail Row (shared across all tabs) ───────────────────────── */

function DetailRow({ m, colSpan, isDark }) {
  return (
    <tr className={isDark ? 'bg-bb-dark-400/30' : 'bg-bb-light-200/40'}>
      <td colSpan={colSpan} className="px-6 py-5">
        <div className="grid grid-cols-2 md:grid-cols-5 gap-4 text-sm animate-fade-in">
          {[{ label: 'GSM8K', val: m.gsm8k_acc, color: isDark ? 'text-bb-accent' : 'text-bb-accent-dark' }, { label: 'GSM-Plus', val: m.gsm_plus_acc, color: 'text-bb-teal' }, { label: 'MATH', val: m.math_acc, color: 'text-bb-mint' }, { label: 'ARC-Easy', val: m.arc_easy_acc, color: 'text-bb-cyan' }, { label: 'ARC-Challenge', val: m.arc_challenge_acc, color: 'text-bb-purple' }].map(({ label, val, color }) => (
            <div key={label} className={`p-3 rounded-lg ${isDark ? 'bg-bb-dark-300/40' : 'bg-white/60'}`}>
              <div className={`text-[10px] uppercase tracking-wider mb-2 font-semibold ${isDark ? 'text-gray-600' : 'text-gray-400'}`}>{label}</div>
              <div className={`${color} font-mono font-semibold`}>{fmt(val)}%</div>
            </div>
          ))}
        </div>
      </td>
    </tr>
  )
}

/* ─── Rank Cell ─────────────────────────────────────────────────── */

function RankCell({ rank, isDark }) {
  const color = rank === 1 ? 'text-yellow-400' : rank === 2 ? 'text-gray-300' : rank === 3 ? 'text-amber-600' : isDark ? 'text-gray-600' : 'text-gray-400'
  return <td className="px-3 py-3 text-center"><span className={`font-mono text-sm font-bold ${color}`}>{rank}</span></td>
}

/* ─── Token Setting label ───────────────────────────────────────── */

function TokenCell({ tokens, isDark }) {
  return (
    <td className={`px-2 py-3 text-center font-mono text-[13px] whitespace-nowrap ${isDark ? 'text-gray-400' : 'text-gray-500'}`}>
      {tokens}
    </td>
  )
}

/* ─── Ensemble Table (shared between MV and MAD) ────────────────── */

function EnsembleTable({ data, description }) {
  const { isDark } = useTheme()
  const [sortKey, setSortKey] = useState('accuracy')
  const [sortDir, setSortDir] = useState('desc')
  const [expandedRow, setExpandedRow] = useState(null)

  const sortedData = useMemo(() => {
    const d = [...data]
    d.sort((a, b) => (sortDir === 'desc' ? -1 : 1) * (a[sortKey] - b[sortKey]))
    return d
  }, [data, sortKey, sortDir])

  function toggleSort(key) { if (sortKey === key) setSortDir(d => d === 'desc' ? 'asc' : 'desc'); else { setSortKey(key); setSortDir('desc') } }
  const SortHeader = ({ k, children, className }) => <SortHeaderCell k={k} sortKey={sortKey} sortDir={sortDir} toggleSort={toggleSort} isDark={isDark} className={className}>{children}</SortHeaderCell>

  if (!data || data.length === 0) {
    return (
      <div className={`rounded-xl overflow-hidden border p-12 text-center ${isDark ? 'bg-bb-dark-300/40 border-bb-dark-50/30' : 'bg-white/80 border-bb-light-300/60 shadow-lg'}`}>
        <FlaskConical className={`w-12 h-12 mx-auto mb-4 ${isDark ? 'text-gray-600' : 'text-gray-300'}`} />
        <h3 className={`text-lg font-semibold mb-2 ${isDark ? 'text-gray-400' : 'text-gray-600'}`}>Experiments in Progress</h3>
        <p className={`text-sm ${isDark ? 'text-gray-600' : 'text-gray-400'}`}>
          Results will appear here once generated. Run <code className="px-1.5 py-0.5 rounded bg-bb-dark-400/50 text-xs font-mono">python temp/generate_model_data.py</code> to refresh.
        </p>
      </div>
    )
  }

  return (
    <div className={`rounded-xl overflow-hidden border ${isDark ? 'bg-bb-dark-300/40 border-bb-dark-50/30 neon-border' : 'bg-white/80 border-bb-light-300/60 shadow-lg'}`}>
      <div className="overflow-x-auto">
        <table className="w-full">
          <thead>
            <tr className={`border-b ${isDark ? 'border-bb-dark-50/20 bg-bb-dark-400/40' : 'border-bb-light-300/50 bg-bb-light-100/80'}`}>
              <th className={`px-2 py-3 text-[11px] font-semibold uppercase tracking-wider text-center w-10 ${isDark ? 'text-gray-500' : 'text-gray-400'}`}>#</th>
              <th className={`px-2 py-3 text-[11px] font-semibold uppercase tracking-wider text-left ${isDark ? 'text-gray-500' : 'text-gray-400'}`}>Ensemble</th>
              <SortHeader k="accuracy">Overall</SortHeader>
              <SortHeader k="gsm8k_acc">GSM8K</SortHeader>
              <SortHeader k="gsm_plus_acc">GSM+</SortHeader>
              <SortHeader k="math_acc">MATH</SortHeader>
              <SortHeader k="arc_easy_acc">ARC-E</SortHeader>
              <SortHeader k="arc_challenge_acc">ARC-C</SortHeader>
              <SortHeader k="ifr">IFR</SortHeader>
            </tr>
          </thead>
          <tbody>
            {sortedData.map((m, i) => {
              const isExpanded = expandedRow === m.ensemble_name
              const base = bestComboBase(m)
              return (
                <>
                  <tr key={m.ensemble_name} className={`border-b cursor-pointer transition-all duration-150 ${isDark ? `border-bb-dark-50/10 ${isExpanded ? 'bg-bb-dark-300/40' : i < 3 ? 'bg-bb-accent/[0.02] hover:bg-bb-dark-300/30' : 'hover:bg-bb-dark-300/20'}` : `border-bb-light-300/30 ${isExpanded ? 'bg-bb-light-200/60' : i < 3 ? 'bg-bb-accent-dark/[0.03] hover:bg-bb-light-200/40' : 'hover:bg-bb-light-200/40'}`}`} onClick={() => setExpandedRow(isExpanded ? null : m.ensemble_name)}>
                    <RankCell rank={i + 1} isDark={isDark} />
                    <td className="px-2 py-3">
                      <div className="flex flex-wrap gap-1">
                        {m.judges?.map((j, idx) => {
                          const familyColor = FAMILY_COLORS[j.family] || '#6b7280'
                          return (
                            <span key={idx} className={`inline-flex items-center gap-1 px-1.5 py-0.5 rounded text-[13px] font-medium ${isDark ? 'bg-bb-dark-400/60 text-gray-300' : 'bg-bb-light-200/80 text-gray-600'}`}>
                              <span className="w-1.5 h-1.5 rounded-full shrink-0" style={{ backgroundColor: familyColor }} />
                              {j.name}
                              <span className={`text-[9px] font-mono ${isDark ? 'text-gray-600' : 'text-gray-400'}`}>{j.tokens != null ? `t=${j.tokens}` : (j.temperature != null ? `T=${j.temperature}` : '')}</span>
                            </span>
                          )
                        })}
                      </div>
                    </td>
                    <td className="px-2 py-3 text-center whitespace-nowrap">
                      <span className={`font-mono text-[13px] font-semibold ${isDark ? 'text-bb-accent' : 'text-bb-accent-dark'}`}>{fmt(m.accuracy)}%</span>
                      <Delta value={m.accuracy} base={base} />
                    </td>
                    <td className={`px-2 py-3 text-center font-mono text-[13px] whitespace-nowrap ${isDark ? 'text-gray-300' : 'text-gray-600'}`}>{fmt(m.gsm8k_acc)}%</td>
                    <td className={`px-2 py-3 text-center font-mono text-[13px] whitespace-nowrap ${isDark ? 'text-gray-300' : 'text-gray-600'}`}>{fmt(m.gsm_plus_acc)}%</td>
                    <td className={`px-2 py-3 text-center font-mono text-[13px] whitespace-nowrap ${isDark ? 'text-gray-300' : 'text-gray-600'}`}>{fmt(m.math_acc)}%</td>
                    <td className={`px-2 py-3 text-center font-mono text-[13px] whitespace-nowrap ${isDark ? 'text-gray-300' : 'text-gray-600'}`}>{fmt(m.arc_easy_acc)}%</td>
                    <td className={`px-2 py-3 text-center font-mono text-[13px] whitespace-nowrap ${isDark ? 'text-gray-300' : 'text-gray-600'}`}>{fmt(m.arc_challenge_acc)}%</td>
                    <td className={`px-2 py-3 text-center font-mono text-[13px] whitespace-nowrap ${isDark ? 'text-gray-500' : 'text-gray-400'}`}>{fmt(m.ifr)}%</td>
                  </tr>
                  {isExpanded && <DetailRow key={`${m.ensemble_name}-d`} m={m} colSpan={9} isDark={isDark} />}
                </>
              )
            })}
          </tbody>
        </table>
      </div>
      <div className={`px-4 py-3 border-t text-xs text-center ${isDark ? 'border-bb-dark-50/10 text-gray-600' : 'border-bb-light-300/30 text-gray-400'}`}>
        {sortedData.length} ensembles · Δ compared to best individual judge in each combo · Click a row for details
      </div>
    </div>
  )
}

/* ─── Phase 1 Table ─────────────────────────────────────────────── */

function Phase1Table() {
  const { isDark } = useTheme()
  const [search, setSearch] = useState('')
  const [sortKey, setSortKey] = useState('accuracy')
  const [sortDir, setSortDir] = useState('desc')
  const [familyFilter, setFamilyFilter] = useState('all')
  const [tokenFilter, setTokenFilter] = useState('all')
  const [showFilters, setShowFilters] = useState(false)
  const [expandedRow, setExpandedRow] = useState(null)

  const families = useMemo(() => ['all', ...new Set(modelData.map(m => m.family))].sort(), [])

  const sortedData = useMemo(() => {
    let data = [...modelData]
    if (search) { const q = search.toLowerCase(); data = data.filter(m => m.model.toLowerCase().includes(q) || m.shortName.toLowerCase().includes(q)) }
    if (familyFilter !== 'all') data = data.filter(m => m.family === familyFilter)
    if (tokenFilter !== 'all') data = data.filter(m => m.tokens === Number(tokenFilter))
    data.sort((a, b) => { const mult = sortDir === 'desc' ? -1 : 1; return typeof a[sortKey] === 'string' ? a[sortKey].localeCompare(b[sortKey]) * mult : (a[sortKey] - b[sortKey]) * mult })
    return data
  }, [search, sortKey, sortDir, familyFilter, tokenFilter])

  const globalRanked = useMemo(() => {
    const ranked = [...modelData].sort((a, b) => b.accuracy - a.accuracy)
    const map = {}; ranked.forEach((m, i) => { map[getUniqueKey(m)] = i + 1 }); return map
  }, [])

  function toggleSort(key) { if (sortKey === key) setSortDir(d => d === 'desc' ? 'asc' : 'desc'); else { setSortKey(key); setSortDir('desc') } }
  const SortHeader = ({ k, children, className }) => <SortHeaderCell k={k} sortKey={sortKey} sortDir={sortDir} toggleSort={toggleSort} isDark={isDark} className={className}>{children}</SortHeaderCell>
  const activeFilterCount = (familyFilter !== 'all' ? 1 : 0) + (tokenFilter !== 'all' ? 1 : 0)

  return (
    <>
      {/* Filters */}
      <div className={`rounded-xl p-4 mb-6 backdrop-blur-xl border transition-all ${isDark ? 'bg-bb-dark-300/60 border-bb-dark-50/30' : 'bg-white/70 border-bb-light-300/60 shadow-sm'}`}>
        <div className="flex flex-col sm:flex-row gap-3">
          <div className="relative flex-1">
            <Search className={`absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 ${isDark ? 'text-gray-500' : 'text-gray-400'}`} />
            <input type="text" placeholder="Search models..." value={search} onChange={e => setSearch(e.target.value)}
              className={`w-full pl-10 pr-4 py-2.5 rounded-lg text-sm transition-all duration-200 focus:outline-none focus:ring-2 ${isDark ? 'bg-bb-dark-400/60 border border-bb-dark-50/20 text-gray-200 placeholder-gray-600 focus:border-bb-accent/40 focus:ring-bb-accent/10' : 'bg-bb-light-100 border border-bb-light-300 text-gray-900 placeholder-gray-400 focus:border-bb-accent-dark/40 focus:ring-bb-accent-dark/10'}`} />
          </div>
          <button onClick={() => setShowFilters(!showFilters)} className={`flex items-center gap-2 px-4 py-2.5 rounded-lg text-sm font-medium transition-all ${showFilters ? (isDark ? 'bg-bb-accent/10 text-bb-accent border border-bb-accent/30' : 'bg-bb-accent-dark/10 text-bb-accent-dark border border-bb-accent-dark/30') : (isDark ? 'bg-bb-dark-400/60 border border-bb-dark-50/20 text-gray-400 hover:text-white' : 'bg-bb-light-100 border border-bb-light-300 text-gray-500 hover:text-gray-700')}`}>
            <Filter className="w-4 h-4" /> Filters
            {activeFilterCount > 0 && <span className={`ml-1 w-5 h-5 flex items-center justify-center rounded-full text-[10px] font-bold ${isDark ? 'bg-bb-accent/20 text-bb-accent' : 'bg-bb-accent-dark/20 text-bb-accent-dark'}`}>{activeFilterCount}</span>}
          </button>
        </div>
        {showFilters && (
          <div className={`mt-3 pt-3 border-t space-y-4 ${isDark ? 'border-bb-dark-50/10' : 'border-bb-light-300/50'}`}>
            <div>
              <div className={`text-xs mb-2 font-medium ${isDark ? 'text-gray-500' : 'text-gray-400'}`}>Model Family</div>
              <div className="flex flex-wrap gap-2">
                {families.map(f => <button key={f} onClick={() => setFamilyFilter(f)} className={`px-3 py-1.5 rounded-full text-xs font-medium transition-all ${familyFilter === f ? (isDark ? 'bg-bb-accent/20 text-bb-accent border border-bb-accent/30' : 'bg-bb-accent-dark/15 text-bb-accent-dark border border-bb-accent-dark/30') : (isDark ? 'bg-bb-dark-400/40 text-gray-500 border border-bb-dark-50/10 hover:text-gray-300' : 'bg-bb-light-200 text-gray-500 border border-bb-light-300 hover:text-gray-700')}`}>{f === 'all' ? 'All' : f} {f !== 'all' && <span className="ml-1 opacity-60">({modelData.filter(m => m.family === f).length})</span>}</button>)}
              </div>
            </div>
            <div>
              <div className={`text-xs mb-2 font-medium ${isDark ? 'text-gray-500' : 'text-gray-400'}`}>Token Setting</div>
              <div className="flex flex-wrap gap-2">
                {['all', '10', '8192'].map(t => <button key={t} onClick={() => setTokenFilter(t)} className={`px-3 py-1.5 rounded-full text-xs font-medium transition-all ${tokenFilter === t ? (isDark ? 'bg-bb-accent/20 text-bb-accent border border-bb-accent/30' : 'bg-bb-accent-dark/15 text-bb-accent-dark border border-bb-accent-dark/30') : (isDark ? 'bg-bb-dark-400/40 text-gray-500 border border-bb-dark-50/10 hover:text-gray-300' : 'bg-bb-light-200 text-gray-500 border border-bb-light-300 hover:text-gray-700')}`}>{t === 'all' ? 'All' : t === '10' ? 'Concise (10)' : 'Reasoned (8192)'}</button>)}
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Table */}
      <div className={`rounded-xl overflow-hidden border ${isDark ? 'bg-bb-dark-300/40 border-bb-dark-50/30 neon-border' : 'bg-white/80 border-bb-light-300/60 shadow-lg'}`}>
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className={`border-b ${isDark ? 'border-bb-dark-50/20 bg-bb-dark-400/40' : 'border-bb-light-300/50 bg-bb-light-100/80'}`}>
                <th className={`px-2 py-3 text-[11px] font-semibold uppercase tracking-wider text-center w-10 ${isDark ? 'text-gray-500' : 'text-gray-400'}`}>#</th>
                <SortHeader k="model" className="text-left">Model</SortHeader>
                <th className={`px-2 py-3 text-[11px] font-semibold uppercase tracking-wider ${isDark ? 'text-gray-500' : 'text-gray-400'}`}>Max Tokens</th>
                <SortHeader k="accuracy">Overall</SortHeader>
                <SortHeader k="gsm8k_acc">GSM8K</SortHeader>
                <SortHeader k="gsm_plus_acc">GSM+</SortHeader>
                <SortHeader k="math_acc">MATH</SortHeader>
                <SortHeader k="arc_easy_acc">ARC-E</SortHeader>
                <SortHeader k="arc_challenge_acc">ARC-C</SortHeader>
                <SortHeader k="ifr">IFR</SortHeader>
              </tr>
            </thead>
            <tbody>
              {sortedData.map((m) => {
                const key = getUniqueKey(m)
                const rank = globalRanked[key]
                const isExpanded = expandedRow === key
                const isTop3 = rank <= 3
                return (
                  <>
                    <tr key={key} className={`border-b cursor-pointer transition-all duration-150 ${isDark ? `border-bb-dark-50/10 ${isExpanded ? 'bg-bb-dark-300/40' : isTop3 ? 'bg-bb-accent/[0.02] hover:bg-bb-dark-300/30' : 'hover:bg-bb-dark-300/20'}` : `border-bb-light-300/30 ${isExpanded ? 'bg-bb-light-200/60' : isTop3 ? 'bg-bb-accent-dark/[0.03] hover:bg-bb-light-200/40' : 'hover:bg-bb-light-200/40'}`}`} onClick={() => setExpandedRow(isExpanded ? null : key)}>
                      <RankCell rank={rank} isDark={isDark} />
                      <td className="px-2 py-3">
                        <ModelNameCell name={m.model.split('/')[1] || m.model} family={m.family} params={m.params} isDark={isDark} />
                      </td>
                      <TokenCell tokens={m.tokens} isDark={isDark} />
                      <td className="px-2 py-3 text-center whitespace-nowrap"><span className={`font-mono text-[13px] font-semibold ${isDark ? 'text-bb-accent' : 'text-bb-accent-dark'}`}>{fmt(m.accuracy)}%</span></td>
                      <td className={`px-2 py-3 text-center font-mono text-[13px] whitespace-nowrap ${isDark ? 'text-gray-300' : 'text-gray-600'}`}>{fmt(m.gsm8k_acc)}%</td>
                      <td className={`px-2 py-3 text-center font-mono text-[13px] whitespace-nowrap ${isDark ? 'text-gray-300' : 'text-gray-600'}`}>{fmt(m.gsm_plus_acc)}%</td>
                      <td className={`px-2 py-3 text-center font-mono text-[13px] whitespace-nowrap ${isDark ? 'text-gray-300' : 'text-gray-600'}`}>{fmt(m.math_acc)}%</td>
                      <td className={`px-2 py-3 text-center font-mono text-[13px] whitespace-nowrap ${isDark ? 'text-gray-300' : 'text-gray-600'}`}>{fmt(m.arc_easy_acc)}%</td>
                      <td className={`px-2 py-3 text-center font-mono text-[13px] whitespace-nowrap ${isDark ? 'text-gray-300' : 'text-gray-600'}`}>{fmt(m.arc_challenge_acc)}%</td>
                      <td className={`px-2 py-3 text-center font-mono text-[13px] whitespace-nowrap ${isDark ? 'text-gray-500' : 'text-gray-400'}`}>{fmt(m.ifr)}%</td>
                    </tr>
                    {isExpanded && <DetailRow key={`${key}-d`} m={m} colSpan={10} isDark={isDark} />}
                  </>
                )
              })}
            </tbody>
          </table>
        </div>
        <div className={`px-4 py-3 border-t text-xs text-center ${isDark ? 'border-bb-dark-50/10 text-gray-600' : 'border-bb-light-300/30 text-gray-400'}`}>
          Showing {sortedData.length} of {modelData.length} configurations · Click a row for dataset breakdown
        </div>
      </div>
    </>
  )
}

/* ─── Persona Effect Table ──────────────────────────────────────── */

function PersonaTable() {
  const { isDark } = useTheme()
  const [sortKey, setSortKey] = useState('baseAcc')
  const [sortDir, setSortDir] = useState('desc')
  const [expandedRow, setExpandedRow] = useState(null)

  const sortedData = useMemo(() => {
    const data = personaData.map(m => ({
      ...m,
      baseAcc: phase1AccMap[getUniqueKey(m)] || 0,
    }))
    data.sort((a, b) => {
      const mult = sortDir === 'desc' ? -1 : 1
      if (sortKey === 'baseAcc') return (a.baseAcc - b.baseAcc) * mult
      return typeof a[sortKey] === 'string' ? a[sortKey].localeCompare(b[sortKey]) * mult : (a[sortKey] - b[sortKey]) * mult
    })
    return data
  }, [sortKey, sortDir])

  function toggleSort(key) { if (sortKey === key) setSortDir(d => d === 'desc' ? 'asc' : 'desc'); else { setSortKey(key); setSortDir('desc') } }
  const SortHeader = ({ k, children, className }) => <SortHeaderCell k={k} sortKey={sortKey} sortDir={sortDir} toggleSort={toggleSort} isDark={isDark} className={className}>{children}</SortHeaderCell>

  return (
    <div className={`rounded-xl overflow-hidden border ${isDark ? 'bg-bb-dark-300/40 border-bb-dark-50/30 neon-border' : 'bg-white/80 border-bb-light-300/60 shadow-lg'}`}>
      <div className="overflow-x-auto">
        <table className="w-full">
          <thead>
            <tr className={`border-b ${isDark ? 'border-bb-dark-50/20 bg-bb-dark-400/40' : 'border-bb-light-300/50 bg-bb-light-100/80'}`}>
              <th className={`px-2 py-3 text-[10px] font-semibold uppercase tracking-wider text-center w-8 ${isDark ? 'text-gray-500' : 'text-gray-400'}`}>#</th>
              <th className={`px-2 py-3 text-[10px] font-semibold uppercase tracking-wider text-left ${isDark ? 'text-gray-500' : 'text-gray-400'}`}>Model</th>
              <th className={`px-2 py-3 text-[10px] font-semibold uppercase tracking-wider whitespace-nowrap ${isDark ? 'text-gray-500' : 'text-gray-400'}`}>Max Tokens</th>
              <SortHeader k="baseAcc">Base Acc</SortHeader>
              {PERSONAS.map(p => (
                <th key={p} className={`px-2 py-3 text-[10px] font-semibold uppercase tracking-wider ${isDark ? 'text-gray-500' : 'text-gray-400'}`}>
                  {PERSONA_LABELS[p]}
                </th>
              ))}
              <SortHeader k="ifr">IFR</SortHeader>
            </tr>
          </thead>
          <tbody>
            {sortedData.map((m, i) => {
              const key = getUniqueKey(m)
              const isExpanded = expandedRow === key
              return (
                <>
                  <tr key={key} className={`border-b cursor-pointer transition-all duration-150 ${isDark ? `border-bb-dark-50/10 ${isExpanded ? 'bg-bb-dark-300/40' : i < 3 ? 'bg-bb-accent/[0.02] hover:bg-bb-dark-300/30' : 'hover:bg-bb-dark-300/20'}` : `border-bb-light-300/30 ${isExpanded ? 'bg-bb-light-200/60' : i < 3 ? 'bg-bb-accent-dark/[0.03] hover:bg-bb-light-200/40' : 'hover:bg-bb-light-200/40'}`}`} onClick={() => setExpandedRow(isExpanded ? null : key)}>
                    <RankCell rank={i + 1} isDark={isDark} />
                    <td className="px-2 py-3">
                      <ModelNameCell name={m.model.split('/')[1] || m.model} family={m.family} params={m.params} isDark={isDark} />
                    </td>
                    <TokenCell tokens={m.tokens} isDark={isDark} />
                    <td className="px-2 py-3 text-center whitespace-nowrap">
                      <span className={`font-mono text-[13px] font-semibold ${isDark ? 'text-gray-400' : 'text-gray-500'}`}>{fmt(m.baseAcc)}%</span>
                    </td>
                    {PERSONAS.map(p => {
                      const pAcc = m.persona_acc?.[p]
                      return (
                        <td key={p} className={`px-2 py-3 text-center font-mono text-[13px] whitespace-nowrap ${isDark ? 'text-gray-300' : 'text-gray-600'}`}>
                          {pAcc != null ? `${fmt(pAcc)}%` : '—'}
                          {pAcc != null && <Delta value={pAcc} base={m.baseAcc} />}
                        </td>
                      )
                    })}
                    <td className={`px-2 py-3 text-center font-mono text-[13px] whitespace-nowrap ${isDark ? 'text-gray-500' : 'text-gray-400'}`}>{fmt(m.ifr)}%</td>
                  </tr>
                  {isExpanded && <DetailRow key={`${key}-d`} m={m} colSpan={PERSONAS.length + 5} isDark={isDark} />}
                </>
              )
            })}
          </tbody>
        </table>
      </div>
      <div className={`px-4 py-3 border-t text-xs text-center ${isDark ? 'border-bb-dark-50/10 text-gray-600' : 'border-bb-light-300/30 text-gray-400'}`}>
        {sortedData.length} configs · Base Acc = Phase 1 individual accuracy · Δ = persona effect vs base · Click a row for dataset breakdown
      </div>
    </div>
  )
}

/* ─── Main Leaderboard ──────────────────────────────────────────── */

export default function Leaderboard() {
  const { isDark } = useTheme()
  const [activeTab, setActiveTab] = useState('phase1')

  const top3 = useMemo(() => [...modelData].sort((a, b) => b.accuracy - a.accuracy).slice(0, 3), [])

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      {/* Hero Section */}
      <div className="text-center mb-14 relative">
        <div className={`absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[600px] h-[400px] rounded-full blur-3xl pointer-events-none ${isDark ? 'bg-bb-accent/[0.04]' : 'bg-bb-accent-dark/[0.03]'}`} />
        <div className="relative pt-10 pb-6">
          <AnimateIn delay={0} direction="scale">
            <div className="flex items-center justify-center gap-3 mb-5">
              <div className="relative">
                <Scale className={`w-12 h-12 animate-float ${isDark ? 'text-bb-accent' : 'text-bb-accent-dark'}`} />
                <div className={`absolute inset-0 w-12 h-12 rounded-full blur-2xl animate-glow-pulse ${isDark ? 'bg-bb-accent/30' : 'bg-bb-accent-dark/20'}`} />
              </div>
            </div>
          </AnimateIn>
          <AnimateIn delay={150} direction="up">
            <h1 className="text-5xl sm:text-6xl font-extrabold mb-4 tracking-tight">
              <span className={isDark ? 'text-white' : 'text-gray-900'}>SLM</span>
              <span className={`${isDark ? 'text-bb-accent neon-text' : 'text-bb-accent-dark'}`}>Jury</span>
            </h1>
          </AnimateIn>
          <AnimateIn delay={300} direction="up">
            <p className={`text-lg sm:text-xl max-w-2xl mx-auto mb-5 font-light ${isDark ? 'text-gray-400' : 'text-gray-600'}`}>
              Can Small Language Models Judge as Well as Large Language Models?
            </p>
          </AnimateIn>
          <AnimateIn delay={450} direction="up">
            <div className="flex flex-wrap items-center justify-center gap-2 text-xs">
              <a href="#" className={`inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full font-mono transition-all hover:scale-105 ${isDark ? 'bg-bb-dark-300/60 text-gray-400 border border-bb-dark-50/20 hover:text-gray-300' : 'bg-bb-light-200 text-gray-500 border border-bb-light-300 hover:text-gray-700'}`}>
                arXiv (Coming Soon) <ArrowUpRight className="w-3 h-3" />
              </a>
              <a href={projectInfo.github} target="_blank" rel="noopener noreferrer" className={`inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full font-mono transition-all hover:scale-105 ${isDark ? 'bg-bb-dark-300/60 text-gray-400 border border-bb-dark-50/20 hover:text-gray-300' : 'bg-bb-light-200 text-gray-500 border border-bb-light-300 hover:text-gray-700'}`}>
                GitHub <ArrowUpRight className="w-3 h-3" />
              </a>
            </div>
          </AnimateIn>
          <AnimateIn delay={600} direction="up">
            <div className={`mt-10 rounded-2xl p-6 sm:p-8 backdrop-blur-xl border ${isDark ? 'bg-bb-dark-300/40 border-bb-dark-50/20' : 'bg-white/50 border-bb-light-300/40'}`}>
              <div className="flex flex-wrap items-center justify-center gap-x-3 gap-y-2">
                {projectInfo.authors.map((a, i) => (
                  <span key={a.name} className="inline-flex items-center">
                    <span className={`text-base sm:text-lg font-semibold tracking-tight ${isDark ? 'text-white' : 'text-gray-800'}`}>{a.name}</span>
                    {i < projectInfo.authors.length - 1 && <span className={`ml-3 ${isDark ? 'text-gray-600' : 'text-gray-300'}`}>&bull;</span>}
                  </span>
                ))}
              </div>
              <div className="mt-4 flex flex-wrap items-center justify-center">
                <span className={`inline-flex items-center gap-2 px-4 py-1.5 rounded-full text-sm font-medium ${isDark ? 'bg-gradient-to-r from-bb-accent/10 to-bb-teal/10 text-gray-200 border border-bb-accent/20' : 'bg-gradient-to-r from-bb-accent-dark/8 to-bb-teal/8 text-gray-700 border border-bb-accent-dark/20'}`}>
                  <span className={`w-2 h-2 rounded-full ${isDark ? 'bg-bb-accent' : 'bg-bb-accent-dark'}`} />
                  LNM Institute of Information Technology, India
                </span>
              </div>
            </div>
          </AnimateIn>
        </div>
      </div>

      {/* Stats */}
      <AnimateIn delay={700} direction="up">
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-12">
          <StatCard icon={Brain} label="Judge Models" value="16" sub="0.6B to 14B parameters" />
          <StatCard icon={Database} label="Problems Evaluated" value="20K+" sub="Across 5 benchmark datasets" />
          <StatCard icon={Layers} label="Total Experiments" value="2,400+" sub="Judges × Students × Settings" />
          <StatCard icon={FlaskConical} label="Evaluation Techniques" value="4" sub="Token Budget · Persona · Ensemble · Debate" />
        </div>
      </AnimateIn>

      {/* Top 3 Podium */}
      <AnimateIn delay={900} direction="up">
        <div className="mb-12">
          <div className="flex items-center gap-3 mb-5">
            <div className={`flex items-center justify-center w-8 h-8 rounded-lg ${isDark ? 'bg-bb-accent/10' : 'bg-bb-accent-dark/10'}`}>
              <Trophy className={`w-4 h-4 ${isDark ? 'text-bb-accent' : 'text-bb-accent-dark'}`} />
            </div>
            <h2 className={`text-lg font-semibold ${isDark ? 'text-gray-200' : 'text-gray-800'}`}>Top Performing Judges</h2>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {top3.map((m, i) => <PodiumCard key={getUniqueKey(m)} model={m} index={i} />)}
          </div>
        </div>
      </AnimateIn>

      {/* Tab Bar */}
      <AnimateIn delay={1100} direction="up">
        <div className={`flex flex-wrap gap-2 mb-6 p-1.5 rounded-xl ${isDark ? 'bg-bb-dark-300/60 border border-bb-dark-50/30' : 'bg-white/70 border border-bb-light-300/60 shadow-sm'}`}>
          {TABS.map(tab => {
            const Icon = tab.icon
            const isActive = activeTab === tab.id
            return (
              <button key={tab.id} onClick={() => setActiveTab(tab.id)}
                className={`flex items-center gap-2 px-4 py-2.5 rounded-lg text-sm font-medium transition-all duration-200 ${isActive
                  ? isDark ? 'bg-bb-accent/15 text-bb-accent shadow-[0_0_15px_rgba(0,170,255,0.1)] border border-bb-accent/20' : 'bg-bb-accent-dark/10 text-bb-accent-dark border border-bb-accent-dark/20'
                  : isDark ? 'text-gray-500 hover:text-gray-300 hover:bg-bb-dark-400/40 border border-transparent' : 'text-gray-500 hover:text-gray-700 hover:bg-bb-light-200/60 border border-transparent'
                  }`}>
                <Icon className="w-4 h-4" />
                {tab.label}
              </button>
            )
          })}
        </div>

        {/* Tab Content */}
        {activeTab === 'phase1' && <Phase1Table />}

        {activeTab === 'majority' && (
          <>
            <div className={`rounded-xl p-4 mb-6 ${isDark ? 'bg-bb-dark-300/40 border border-bb-dark-50/20' : 'bg-white/60 border border-bb-light-300/40 shadow-sm'}`}>
              <p className={`text-sm ${isDark ? 'text-gray-400' : 'text-gray-600'}`}>
                <span className="font-semibold">Majority Voting Ensembles:</span> C(5,3) = 10 combinations of the top 5 small judges. Each problem is judged by 3 models and the majority verdict wins.
                <span className={`ml-2 text-xs ${isDark ? 'text-gray-600' : 'text-gray-400'}`}>t=10 (Concise) · t=8192 (Reasoned)</span>
              </p>
            </div>
            <EnsembleTable data={majorityVotingData} description="majority voting" />
          </>
        )}

        {activeTab === 'persona' && (
          <>
            <div className={`rounded-xl p-4 mb-6 ${isDark ? 'bg-bb-dark-300/40 border border-bb-dark-50/20' : 'bg-white/60 border border-bb-light-300/40 shadow-sm'}`}>
              <p className={`text-sm ${isDark ? 'text-gray-400' : 'text-gray-600'}`}>
                <span className="font-semibold">Persona Effect:</span> Each judge adopts 6 distinct evaluation personas to assess how role-prompting impacts judging accuracy.
                <span className={`ml-2 text-xs ${isDark ? 'text-gray-600' : 'text-gray-400'}`}>Base Acc = Phase 1 accuracy · Δ = persona effect vs base</span>
              </p>
            </div>
            <PersonaTable />
          </>
        )}

        {activeTab === 'mad' && (
          <>
            <div className={`rounded-xl p-4 mb-6 ${isDark ? 'bg-bb-dark-300/40 border border-bb-dark-50/20' : 'bg-white/60 border border-bb-light-300/40 shadow-sm'}`}>
              <p className={`text-sm ${isDark ? 'text-gray-400' : 'text-gray-600'}`}>
                <span className="font-semibold">Multi-Agent Debate:</span> Multiple judge agents engage in structured debate rounds to reach consensus. Agents share reasoning and refine their verdicts across multiple rounds before a final majority vote.
                <span className={`ml-2 text-xs ${isDark ? 'text-gray-600' : 'text-gray-400'}`}>All agents run with max_tokens = 8192 (Reasoned) · T = temperature</span>
              </p>
            </div>
            <EnsembleTable data={madData} description="multi-agent debate" />
          </>
        )}
      </AnimateIn>
    </div>
  )
}
