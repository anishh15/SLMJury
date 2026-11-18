import { useEffect, useRef } from 'react'
import { useTabIndicator } from '../lib/motion'
import { Icon, MarkIcon, Segmented } from '../components/ui'
import { byKey, claims, facts, judges, name, QUICK, REASONED } from '../lib/data'
import { download, toCSV } from '../lib/export'
import { count, fmt, signed } from '../lib/format'

// The leaderboard's other tabs: majority voting, adversarial personas, multi-agent
// debate and open-ended scoring. Each is one sortable table with CSV, JSON and a link.

const R = count(REASONED)
export const TABS = [
  { key: 'judges', label: 'Judges' },
  { key: 'voting', label: 'Majority voting' },
  { key: 'personas', label: 'Adversarial personas' },
  { key: 'debate', label: 'Multi-agent debate' },
  { key: 'open-ended', label: 'Open-ended' },
]
export const TAB_KEYS = TABS.map((t) => t.key)
const SE_DIMS = ['coherence', 'consistency', 'fluency', 'relevance']
const cap = (s) => s.charAt(0).toUpperCase() + s.slice(1)

export function TabBar({ tab, onChange }) {
  const ref = useRef(null)
  const bar = useTabIndicator(ref, tab, { inset: 10 })
  // keep the active tab in view when the row scrolls (phones), without scrolling the page
  useEffect(() => {
    const box = ref.current
    const el = box?.querySelector('[aria-selected="true"]')
    if (el && box.scrollWidth > box.clientWidth) box.scrollLeft = el.offsetLeft - (box.clientWidth - el.offsetWidth) / 2
  }, [tab])
  const onKey = (e) => {
    const i = TAB_KEYS.indexOf(tab)
    const n = e.key === 'ArrowRight' ? i + 1 : e.key === 'ArrowLeft' ? i - 1 : e.key === 'Home' ? 0 : e.key === 'End' ? TAB_KEYS.length - 1 : null
    if (n == null) return
    e.preventDefault()
    const k = TAB_KEYS[(n + TAB_KEYS.length) % TAB_KEYS.length]
    onChange(k)
    requestAnimationFrame(() => ref.current?.querySelector(`[data-tab="${k}"]`)?.focus())
  }
  return (
    <div className="lb-tabs" role="tablist" aria-label="Leaderboard tables" ref={ref} onKeyDown={onKey}>
      {TABS.map((t) => (
        <button key={t.key} type="button" role="tab" data-tab={t.key} id={`tab-${t.key}`} aria-selected={tab === t.key}
          aria-controls="lb-panel" tabIndex={tab === t.key ? 0 : -1} onClick={() => onChange(t.key)}>
          {t.label}
        </button>
      ))}
      <span className="tab-bar" ref={bar} aria-hidden="true" />
    </div>
  )
}

function Who({ k, sub }) {
  const j = byKey[k]
  return (
    <span className="tt-who">
      <MarkIcon fam={j.family} size={11} ring={j.kind === 'ladder'} />
      <a href={`#/judge/${k}`}>{j.name}</a>
      {sub && <span className="faint">{sub}</span>}
    </span>
  )
}
const budgetSub = (b) => `B = ${count(b)}`
const tempList = (ms) => ms.map((m) => m.t).join(', ')

// ── Column sets ─────────────────────────────────────────────────────────────
function votingTable(res) {
  const rows = res.strategyTables.juries.map((x, i) => ({ ...x, n: i + 1 }))
  const columns = [
    { key: 'n', label: '#', value: (x) => x.n, cls: 'c-rank', cell: (x) => x.n },
    { key: 'jury', label: 'Jury of three', value: (x) => x.members.map((m) => name(m.key)).join(' + '), left: true, sort: false,
      cell: (x) => <span className="tt-stack">{x.members.map((m) => <Who key={m.key} k={m.key} sub={budgetSub(m.budget)} />)}</span> },
    { key: 'acc', label: 'Accuracy (%)', value: (x) => x.acc, cell: (x) => fmt(x.acc), best: 'max' },
    { key: 'gain', label: 'Gain over best member (points)', value: (x) => x.gain, cell: (x) => signed(x.gain), best: 'max' },
    { key: 'best', label: 'Best member', value: (x) => x.bestAcc, left: true,
      cell: (x) => <span className="tt-stack"><Who k={x.best} sub={budgetSub(x.bestBudget)} /><span className="faint tnum">{fmt(x.bestAcc)}%</span></span> },
  ]
  const exportRow = (x) => ({
    rank: x.n,
    ...Object.fromEntries(x.members.flatMap((m, i) => [[`judge_${i + 1}`, name(m.key)], [`judge_${i + 1}_budget`, m.budget]])),
    accuracy: x.acc, ifr: x.ifr, best_member: name(x.best), best_member_budget: x.bestBudget,
    best_member_accuracy: x.bestAcc, gain_points: x.gain,
  })
  const note = <>Each jury takes the majority of its three judges’ verdicts. The best member is the jury’s most accurate judge
    at the budget it votes with, and the gain is the jury’s accuracy minus that judge’s, in points. The best jury gains
    +{claims.juryGain} points.</>
  return { rows, columns, exportRow, note, defaultSort: 'acc' }
}

function personasTable(res, pb) {
  const ks = Object.keys(res.personas).sort((a, b) => (byKey[a].rank ?? 99) - (byKey[b].rank ?? 99))
  const rows = ks.flatMap((k) => [QUICK, REASONED].filter((b) => pb === 'both' || pb === b)
    .filter((b) => res.personas[k][String(b)]).map((b) => ({ key: k, budget: b, ...res.personas[k][String(b)] })))
  const cols = facts.personas.map((p) => ({ key: p, label: p, value: (x) => x[p], cell: (x) => fmt(x[p]), title: `Accuracy with the ${p} persona (%)` }))
  const columns = [
    { key: 'judge', label: 'Judge', value: (x) => name(x.key), left: true, cls: 'c-name sticky-col', cell: (x) => <Who k={x.key} /> },
    { key: 'budget', label: 'Budget', value: (x) => x.budget, cell: (x) => budgetSub(x.budget) },
    { key: 'base', label: 'No persona', value: (x) => x.base, cell: (x) => fmt(x.base), title: 'Accuracy without a persona (%)' },
    ...cols,
    { key: 'range', label: 'Range (points)', value: (x) => x.range, cell: (x) => fmt(x.range), title: 'Highest minus lowest accuracy over the six personas, in points', best: 'min' },
  ]
  const exportRow = (x) => ({ judge: name(x.key), key: x.key, budget: x.budget, no_persona: x.base,
    ...Object.fromEntries(facts.personas.map((p) => [p.toLowerCase(), x[p]])), range_points: x.range })
  const note = <>Accuracy (%) of each judge under each of the {facts.nPersonas} adversarial personas, with no persona for
    reference. Range is the highest minus the lowest persona accuracy, in points. Qwen3-14B and Phi-4 stay within a point
    at B = {QUICK}.</>
  return { rows, columns, exportRow, note, defaultSort: 'base' }
}

function debateTable(res) {
  const rows = res.strategyTables.debates.map((x, i) => ({ ...x, n: i + 1 }))
  const agents = (x) => (x.variant === 'same'
    ? <span className="tt-stack"><Who k={x.members[0].key} sub={`× ${x.members.length}`} /><span className="faint">temperatures {tempList(x.members)}</span></span>
    : <span className="tt-stack">{x.members.map((m) => <Who key={m.key} k={m.key} />)}</span>)
  const columns = [
    { key: 'n', label: '#', value: (x) => x.n, cls: 'c-rank', cell: (x) => x.n },
    { key: 'agents', label: 'Agents', value: (x) => x.members.map((m) => name(m.key)).join(' + '), left: true, sort: false, cell: agents },
    { key: 'acc', label: 'Accuracy (%)', value: (x) => x.acc, cell: (x) => fmt(x.acc), best: 'max' },
    { key: 'change', label: 'Change (points)', value: (x) => x.change, cell: (x) => signed(x.change), best: 'max' },
    { key: 'strongest', label: `Strongest member, B = ${R}`, value: (x) => x.strongestAcc, left: true,
      cell: (x) => <span className="tt-stack"><Who k={x.strongest} /><span className="faint tnum">{fmt(x.strongestAcc)}%</span></span> },
    { key: 'variant', label: 'Variant', value: (x) => x.variant, left: true, cell: (x) => (x.variant === 'same' ? 'Same model' : 'Cross-architecture') },
    { key: 'budget', label: 'Budget', value: (x) => x.budget, cell: (x) => <>{budgetSub(x.budget)}{x.budget !== REASONED && <sup>*</sup>}</> },
  ]
  const exportRow = (x) => ({
    rank: x.n, variant: x.variant === 'same' ? 'same model' : 'cross-architecture',
    ...Object.fromEntries(x.members.flatMap((m, i) => [[`agent_${i + 1}`, name(m.key)], [`agent_${i + 1}_temperature`, m.t ?? '']])),
    budget: x.budget, accuracy: x.acc, ifr: x.ifr, strongest_member: name(x.strongest),
    strongest_member_accuracy_b8192: x.strongestAcc, change_points: x.change,
  })
  const note = <>Three agents give verdicts, then critique and refine them for up to five rounds. Change is the debate’s
    accuracy minus its strongest member judging alone at B = {R}, in points; every debate is below it.
    <sup>*</sup> These configurations ran at B = 4,096 for most files; their Qwen2.5-32B GSM8K files, and in one
    configuration its GSM-Plus file, ran at B = {R}.</>
  return { rows, columns, exportRow, note, defaultSort: 'acc' }
}

function openTable() {
  const rows = judges.filter((j) => j.se != null)
  const columns = [
    { key: 'judge', label: 'Judge', value: (j) => j.name, left: true, cls: 'c-name sticky-col', cell: (j) => <Who k={j.key} sub={j.kind === 'ladder' ? 'ladder' : null} /> },
    { key: 'se', label: 'SummEval ρ', value: (j) => j.se, cell: (j) => fmt(j.se, 3), best: 'max', title: 'Spearman’s ρ with the human scores, over all four dimensions' },
    ...SE_DIMS.map((d) => ({ key: d, label: cap(d), value: (j) => j.seDims[d], cell: (j) => fmt(j.seDims[d], 3), best: 'max', title: `Spearman’s ρ with the human ${d} scores` })),
    { key: 'mt', label: 'MT-Bench ρ', value: (j) => j.mt, cell: (j) => fmt(j.mt, 3), best: 'max', title: 'Spearman’s ρ with the reference scorer' },
    { key: 'size', label: 'Size', value: (j) => j.params, cell: (j) => j.size },
  ]
  const exportRow = (j) => ({ judge: j.name, key: j.key, kind: j.kind, size_b: j.params, summeval_rho: j.se,
    ...Object.fromEntries(SE_DIMS.map((d) => [`summeval_${d}_rho`, j.seDims[d]])), mtbench_rho: j.mt })
  const note = <>Spearman’s ρ between each judge’s 1–{facts.scoreScale} scores and the reference scores. SummEval:
    {' '}{count(facts.nSummaries)} human-scored summaries of {facts.nArticles} articles, overall and for each of the four
    dimensions. MT-Bench: {facts.nConversations} questions answered by both students, against one reference scorer
    (Qwen3.5-397B-A17B). Bold is the best in a column.</>
  return { rows, columns, exportRow, note, defaultSort: 'se' }
}

export const TAB_SORTS = {
  voting: ['n', 'acc', 'best', 'gain'],
  personas: ['judge', 'budget', 'base', ...facts.personas, 'range'],
  debate: ['n', 'variant', 'budget', 'acc', 'strongest', 'change'],
  'open-ended': ['judge', 'size', 'se', ...SE_DIMS, 'mt'],
}
export const TAB_DEFAULT_SORT = { voting: 'acc', personas: 'base', debate: 'acc', 'open-ended': 'se' }

export function buildTab(tab, res, pb) {
  if (tab === 'voting') return votingTable(res)
  if (tab === 'personas') return personasTable(res, pb)
  if (tab === 'debate') return debateTable(res)
  return openTable()
}

export function TabPanel({ tab, res, sort, dir, pb, onSort, onBudget, copied, onCopy }) {
  const t = buildTab(tab, res, pb)
  const col = t.columns.find((c) => c.key === sort && c.sort !== false) || t.columns.find((c) => c.key === t.defaultSort)
  const sign = dir === 'asc' ? 1 : -1
  const rows = [...t.rows].sort((a, b) => {
    const va = col.value(a)
    const vb = col.value(b)
    if (va == null) return 1
    if (vb == null) return -1
    return (va < vb ? -1 : va > vb ? 1 : 0) * sign
  })
  const best = Object.fromEntries(t.columns.filter((c) => c.best).map((c) => {
    const vs = t.rows.map(c.value).filter((v) => v != null)
    return [c.key, c.best === 'max' ? Math.max(...vs) : Math.min(...vs)]
  }))
  const fileBase = `slmjury-${tab}${tab === 'personas' && pb !== 'both' ? `-b${pb}` : ''}`
  const data = () => rows.map(t.exportRow)
  const csv = () => {
    const d = data()
    const keys = Object.keys(d[0] || {})
    return toCSV(keys.map((k) => ({ key: k, label: k })), d)
  }
  return (
    <div id="lb-panel" role="tabpanel" aria-labelledby={`tab-${tab}`}>
      <div className="tt-bar">
        {tab === 'personas' && (
          <Segmented label="Budget" value={pb} onChange={onBudget}
            options={[{ value: 'both', label: 'Both budgets' }, { value: QUICK, label: `B = ${QUICK}` }, { value: REASONED, label: `B = ${R}` }]} />
        )}
        <span className="spacer" />
        <div className="lb-actions">
          <button type="button" className="btn btn-sm" onClick={() => download(`${fileBase}.csv`, csv(), 'text/csv')}><Icon.download /> CSV</button>
          <button type="button" className="btn btn-sm" onClick={() => download(`${fileBase}.json`, JSON.stringify(data(), null, 2) + '\n', 'application/json')}><Icon.download /> JSON</button>
          <button type="button" className="btn btn-sm" onClick={onCopy} aria-live="polite">{copied ? <Icon.check /> : <Icon.link />} {copied ? 'Copied' : 'Copy link'}</button>
        </div>
      </div>
      <div className="tt-scroll" tabIndex={0} role="region" aria-label={`${TABS.find((x) => x.key === tab).label} table`}>
        <table className="lb tt">
          <caption className="sr-only">{TABS.find((x) => x.key === tab).label}</caption>
          <thead>
            <tr>
              {t.columns.map((c) => {
                const active = col.key === c.key
                return (
                  <th key={c.key} className={`${c.left ? '' : 'r '}${c.cls || ''}`} aria-sort={active ? (dir === 'asc' ? 'ascending' : 'descending') : undefined}>
                    {c.sort === false ? <span className="th-plain">{c.label}</span> : (
                      <button type="button" onClick={() => onSort(c.key)} title={c.title}>
                        {c.label}
                        <span className={`sort-ic${active ? ' on' : ''}`}>{active ? (dir === 'asc' ? <Icon.sortUp /> : <Icon.sortDown />) : <Icon.sort />}</span>
                      </button>
                    )}
                  </th>
                )
              })}
            </tr>
          </thead>
          <tbody>
            {rows.map((x, i) => (
              <tr key={i}>
                {t.columns.map((c) => (
                  <td key={c.key} className={`${c.left ? '' : 'r tnum '}${c.cls || ''}${c.best && c.value(x) === best[c.key] ? ' best' : ''}`}>{c.cell(x)}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="lb-note">{t.note}</p>
    </div>
  )
}
