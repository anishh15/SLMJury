import { useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react'
import { copyText, Icon, MarkIcon, Segmented } from '../components/ui'
import JudgePicker from '../components/JudgePicker'
import { byKey, claims, core, facts, FAMILIES, famColor, judges, QUICK, REASONED, useResults } from '../lib/data'
import { count, fmt, signed } from '../lib/format'
import { replaceHash } from '../lib/router'
import { TabBar, TabPanel, TAB_DEFAULT_SORT, TAB_KEYS, TAB_SORTS } from './StrategyTabs'
import { useFlip } from '../lib/flip'
import { clearTags, lastJudge, setLastJudge, tagJudge } from '../lib/vt'
import '../styles/content.css'
import '../styles/leaderboard.css'

const R = count(REASONED)
const DS = core.datasets.closed
// One row model: the budget a judge is shown at (null when it has no verdict there), and every
// value, rank and export at that budget.
const budgetOf = (j, b) => (b === 'best' ? j.bestBudget : b === QUICK ? (j.acc10 == null ? null : QUICK) : REASONED)
const atBudget = (j, b, quick, reasoned) => { const e = budgetOf(j, b); return e == null ? null : e === QUICK ? j[quick] : j[reasoned] }
const accAt = (j, b) => atBudget(j, b, 'acc10', 'acc8192')
const ifrAt = (j, b) => atBudget(j, b, 'ifr10', 'ifr8192')
const costAt = (j, b) => atBudget(j, b, 'cost10', 'cost8192')
const rankAt = (j, b) => (b === 'best' ? j.rank : b === QUICK ? j.rank10 : j.rank8192)
const MAX_COMPARE = 4
const ASC_KEYS = ['n', 'judge', 'variant', 'budget', 'size', 'range']
const LEADS = {
  judges: <>All {facts.nSmall + facts.nLarge} judges and the Qwen3-32B ladder judge are scored on the same {count(facts.nItems)} judgments. Tick up to four to compare them.</>,
  voting: <>{facts.nJuries} juries of three, each drawn from the top {facts.nJuryPool} small judges and decided by majority vote, on the same {count(facts.nItems)} judgments.</>,
  personas: <>{facts.nPersonas} adversarial personas, each a system prompt that pushes the judge, at both budgets.</>,
  debate: <>{facts.nDebates} debates of {facts.nAgents} small agents each, on the same {count(facts.nItems)} judgments.</>,
  'open-ended': <>Agreement with human scores on SummEval and with a reference scorer on MT-Bench, for every judge.</>,
}
const BUDGET_NAME = { best: 'better budget', [QUICK]: `B = ${QUICK}`, [REASONED]: `B = ${R}` }

const COLS = [
  { key: 'rank', label: '#', get: (j) => j.rank ?? 99, asc: true, cls: 'c-rank' },
  { key: 'name', label: 'Judge', get: (j) => j.name.toLowerCase(), asc: true, cls: 'c-name' },
  { key: 'acc', label: 'Accuracy', r: true, cls: 'c-acc' },
  { key: 'params', label: 'Size', get: (j) => j.params, r: true },
  { key: 'acc10', label: `B = ${QUICK}`, get: (j) => j.acc10, r: true, title: `Accuracy at the quick verdict (B = ${QUICK})` },
  { key: 'acc8192', label: `B = ${R}`, get: (j) => j.acc8192, r: true, title: `Accuracy at the reasoned verdict (B = ${R})` },
  { key: 'delta', label: 'Δ', get: (j) => j.delta, r: true, title: `Accuracy at B = ${QUICK} minus B = ${R}, in points` },
  { key: 'ifr', label: 'IFR', r: true, title: 'Instruction-following rate: the share of parseable verdicts (%)' },
  { key: 'se', label: 'SummEval ρ', get: (j) => j.se, r: true, title: 'Spearman correlation with human scores on SummEval' },
  { key: 'mt', label: 'MT-Bench ρ', get: (j) => j.mt, r: true, title: 'Spearman correlation with the reference scorer on MT-Bench' },
  { key: 'cost', label: '$ / 1k', r: true, title: 'Estimated serving cost per 1,000 verdicts, at the same budget as the accuracy column' },
]

function readQuery(q) {
  const b = q.get('budget')
  const compare = (q.get('compare') || '').split(',').filter((k) => judges.some((j) => j.key === k)).slice(0, MAX_COMPARE)
  const tab = TAB_KEYS.includes(q.get('tab')) ? q.get('tab') : 'judges'
  const sorts = tab === 'judges' ? COLS.map((c) => c.key) : TAB_SORTS[tab]
  const pb = q.get('pbudget')
  return {
    tab,
    pb: pb === String(QUICK) ? QUICK : pb === String(REASONED) ? REASONED : 'both',
    budget: b === String(QUICK) ? QUICK : b === String(REASONED) ? REASONED : 'best',
    sort: sorts.includes(q.get('sort')) ? q.get('sort') : tab === 'judges' ? 'acc' : TAB_DEFAULT_SORT[tab],
    dir: q.get('dir') === 'asc' ? 'asc' : 'desc',
    fams: q.get('family') ? q.get('family').split(',') : [],
    kind: ['small', 'large'].includes(q.get('type')) ? q.get('type') : 'all',
    text: q.get('q') || '',
    view: q.get('view') === 'datasets' ? 'datasets' : 'overview',
    ref: q.get('ref') === 'median' || byKey[q.get('ref')] ? q.get('ref') : 'phi4-14b',
    compare,
  }
}
function toQuery(st) {
  const q = {}
  if (st.tab !== 'judges') {
    q.tab = st.tab
    if (st.sort !== TAB_DEFAULT_SORT[st.tab]) q.sort = st.sort
    if (st.dir !== 'desc') q.dir = st.dir
    if (st.tab === 'personas' && st.pb !== 'both') q.pbudget = st.pb
    return q
  }
  if (st.view !== 'overview') q.view = st.view
  if (st.budget !== 'best') q.budget = st.budget
  if (st.sort !== 'acc') q.sort = st.sort
  if (st.dir !== 'desc') q.dir = st.dir
  if (st.fams.length) q.family = st.fams.join(',')
  if (st.kind !== 'all') q.type = st.kind
  if (st.text) q.q = st.text
  if (st.view === 'datasets' && st.ref !== 'phi4-14b') q.ref = st.ref
  if (st.compare.length) q.compare = st.compare.join(',')
  return q
}

function download(name, text, type) {
  const url = URL.createObjectURL(new Blob([text], { type }))
  const a = document.createElement('a')
  a.href = url
  a.download = name
  document.body.appendChild(a)
  a.click()
  a.remove()
  setTimeout(() => URL.revokeObjectURL(url), 1000)
}
const csvCell = (v) => (v == null ? '' : /[",\n]/.test(String(v)) ? `"${String(v).replace(/"/g, '""')}"` : String(v))

// ── Compare drawer ──────────────────────────────────────────────────────────
function Dumbbells({ keys, res }) {
  const vals = keys.flatMap((k) => ['10', '8192'].flatMap((b) => res.perDataset[k]?.[b] || [])).filter((v) => v != null)
  const lo = Math.floor(Math.min(...vals) / 5) * 5
  const x = (v) => `${((v - lo) / (100 - lo)) * 100}%`
  const ticks = []
  for (let t = lo; t <= 100; t += lo < 60 ? 10 : 5) ticks.push(t)
  return (
    <div className="dumbbells">
      <div className="db-axis" aria-hidden="true">
        <span className="db-label" />
        <span className="db-track">{ticks.map((t) => <i key={t} style={{ left: x(t) }}>{t}</i>)}</span>
      </div>
      {DS.map((d, di) => (
        <div key={d.key} className="db-group">
          <div className="db-ds">{d.name}<span>{d.domain}</span></div>
          {keys.map((k) => {
            const j = judges.find((x) => x.key === k)
            const a = res.perDataset[k]?.['10']?.[di]
            const b = res.perDataset[k]?.['8192']?.[di]
            const both = a != null && b != null
            return (
              <div key={k} className="db-row" title={`${j.name}: ${a != null ? `${fmt(a)}% at B = ${QUICK}` : ''}${both ? ', ' : ''}${b != null ? `${fmt(b)}% at B = ${R}` : ''}`}>
                <span className="db-label"><MarkIcon fam={j.family} size={11} /> {j.name}</span>
                <span className="db-track">
                  {ticks.map((t) => <i key={t} className="grid" style={{ left: x(t) }} />)}
                  {both && <span className="db-line" style={{ left: x(Math.min(a, b)), width: `calc(${x(Math.max(a, b))} - ${x(Math.min(a, b))})`, background: famColor(j.family) }} />}
                  {a != null && <span className="db-dot" style={{ left: x(a), background: famColor(j.family) }} />}
                  {b != null && <span className="db-dot open" style={{ left: x(b), borderColor: famColor(j.family) }} />}
                </span>
                <span className="db-vals tnum">{a != null ? fmt(a) : '—'} · {b != null ? fmt(b) : '—'}</span>
              </div>
            )
          })}
        </div>
      ))}
    </div>
  )
}

function CompareDrawer({ keys, onClose, onRemove, res }) {
  const ref = useRef(null)
  useEffect(() => {
    const opener = document.activeElement
    const onKey = (e) => e.key === 'Escape' && onClose()
    window.addEventListener('keydown', onKey)
    document.documentElement.classList.add('lock')
    ref.current?.focus()
    return () => {
      window.removeEventListener('keydown', onKey)
      document.documentElement.classList.remove('lock')
      // focus goes back to what opened the drawer, or to the tray's Compare button
      requestAnimationFrame(() => {
        const back = opener && opener !== document.body && opener.isConnected ? opener : document.querySelector('.compare-tray .btn-primary')
        back?.focus()
      })
    }
  }, [])
  const js = keys.map((k) => judges.find((j) => j.key === k))
  const costOf = (k, b) => res?.cost.find((c) => c.key === k && c.budget === b)?.cost
  const rows = [
    ['Accuracy, B = ' + QUICK, (j) => (j.acc10 == null ? null : `${fmt(j.acc10)}%`), (j) => j.acc10],
    ['Accuracy, B = ' + R, (j) => `${fmt(j.acc8192)}%`, (j) => j.acc8192],
    ['IFR at its better budget', (j) => `${fmt(budgetOf(j, 'best') === QUICK ? j.ifr10 : j.ifr8192)}%`, (j) => (budgetOf(j, 'best') === QUICK ? j.ifr10 : j.ifr8192)],
    ['SummEval ρ', (j) => fmt(j.se, 3), (j) => j.se],
    ['MT-Bench ρ', (j) => fmt(j.mt, 3), (j) => j.mt],
    [`$ per 1k verdicts, B = ${QUICK}`, (j) => (costOf(j.key, QUICK) == null ? null : `$${fmt(costOf(j.key, QUICK))}`), (j) => -(costOf(j.key, QUICK) ?? Infinity)],
    [`$ per 1k verdicts, B = ${R}`, (j) => (costOf(j.key, REASONED) == null ? null : `$${fmt(costOf(j.key, REASONED))}`), (j) => -(costOf(j.key, REASONED) ?? Infinity)],
  ]
  return (
    <>
      <div className="drawer-scrim" onClick={onClose} />
      <section className="drawer" role="dialog" aria-modal="true" aria-label="Compare judges" tabIndex={-1} ref={ref}>
        <header className="drawer-head">
          <div>
            <div className="label">Compare</div>
            <h2 className="h3">{js.map((j) => j.name).join(' · ')}</h2>
          </div>
          <div className="row">
            <button type="button" className="btn btn-sm" onClick={() => copyText(window.location.href)}><Icon.link /> Copy link</button>
            <button type="button" className="icon-btn" onClick={onClose} aria-label="Close the comparison"><Icon.close /></button>
          </div>
        </header>
        <div className="drawer-body">
          <div className="cmp-table-wrap">
            <table className="cmp-table">
              <thead>
                <tr><th />{js.map((j) => (
                  <th key={j.key}>
                    <span className="cmp-judge"><MarkIcon fam={j.family} size={13} /> <a href={`#/judge/${j.key}`}>{j.name}</a></span>
                    <span className="cmp-sub">{j.size} · {j.kind}{j.thinks === 'always' ? ' · always reasons' : ''}</span>
                    {keys.length > 2 && <button type="button" className="cmp-x" onClick={() => onRemove(j.key)} aria-label={`Remove ${j.name}`}><Icon.close /></button>}
                  </th>
                ))}</tr>
              </thead>
              <tbody>
                {rows.map(([label, show, val]) => {
                  const vs = js.map(val).filter((v) => v != null)
                  const top = vs.length > 1 ? Math.max(...vs) : null
                  return (
                    <tr key={label}><th>{label}</th>{js.map((j) => {
                      const v = show(j)
                      return <td key={j.key} className={`tnum${val(j) === top ? ' best' : ''}`}>{v ?? <span className="faint">—</span>}</td>
                    })}</tr>
                  )
                })}
              </tbody>
            </table>
          </div>
          <div className="cmp-section">
            <div className="fig-title">Accuracy on each dataset</div>
            <div className="fig-sub">Filled: B = {QUICK}. Open: B = {R}. The line joins one judge’s two budgets.</div>
            {res ? <Dumbbells keys={keys} res={res} /> : <div className="skeleton" style={{ height: 300 }} />}
          </div>
        </div>
      </section>
    </>
  )
}

// ── Phone bottom sheet ──────────────────────────────────────────────────────
function Sheet({ title, onClose, children }) {
  const ref = useRef(null)
  useEffect(() => {
    const opener = document.activeElement
    const onKey = (e) => e.key === 'Escape' && onClose()
    window.addEventListener('keydown', onKey)
    document.documentElement.classList.add('lock')
    ref.current?.focus()
    return () => {
      window.removeEventListener('keydown', onKey)
      document.documentElement.classList.remove('lock')
      requestAnimationFrame(() => opener?.isConnected && opener.focus())
    }
  }, [])
  return (
    <>
      <div className="drawer-scrim" onClick={onClose} />
      <section className="sheet" role="dialog" aria-modal="true" aria-label={title} tabIndex={-1} ref={ref}>
        <header className="sheet-head">
          <h2 className="h3">{title}</h2>
          <button type="button" className="icon-btn" onClick={onClose} aria-label="Close"><Icon.close /></button>
        </header>
        {children}
      </section>
    </>
  )
}

// ── Difference matrix ──────────────────────────────────────────────────────
// Each cell is a judge's accuracy minus the reference's on that dataset, in
// points. Small differences stay near the surface colour; only real gaps show.
const SHADE_FROM = 0.3
const SHADE_FULL = 5
const shade = (d) => Math.min(1, Math.max(0, Math.abs(d) - SHADE_FROM) / (SHADE_FULL - SHADE_FROM)) ** 0.75
const median = (vs) => {
  const a = [...vs].sort((x, y) => x - y)
  const m = a.length >> 1
  return a.length % 2 ? a[m] : (a[m - 1] + a[m]) / 2
}

function DiffTable({ rows, budget, res, refKey, onSort, sortKey }) {
  const cols = [...DS.map((d, i) => ({
    key: d.key, label: d.name, group: d.domain,
    get: (k, b) => res.perDataset[k]?.[b]?.[i], exact: (k, b) => res.perDatasetExact[k]?.[b]?.[i],
  })),
  ...['math', 'science', 'general'].map((g) => ({
    key: `domain-${g}`, label: g[0].toUpperCase() + g.slice(1), group: 'mean',
    get: (k, b) => res.domainMeans[k]?.[b]?.[g], exact: (k, b) => res.domainMeans[k]?.[b]?.exact[g],
  })),
  ]
  const bud = (j) => (budgetOf(j, budget) == null ? null : String(budgetOf(j, budget)))
  // the reference: one judge at this budget, or each column's median over the ranked judges shown at it
  const asked = refKey === 'median' ? null : byKey[refKey]
  const refJ = asked && bud(asked) ? asked : null
  const refB = refJ ? bud(refJ) : null
  const pool = judges.filter((j) => j.rank && bud(j))
  const refExact = Object.fromEntries(cols.map((c) => [c.key, refJ ? c.exact(refJ.key, refB) : median(pool.map((j) => c.exact(j.key, bud(j))).filter((v) => v != null))]))
  const refName = refJ ? refJ.name : 'the median'
  return (
    <>
      {asked && !refJ && <p className="diff-warn">{asked.name} has no verdicts at B = {QUICK}, because it always reasons, so the reference here is the median.</p>}
      <div className="lb-scroll heat-scroll" tabIndex={0} role="region" aria-label="Accuracy on each dataset">
        <table className="lb heat">
          <caption className="sr-only">Accuracy on each dataset as points above or below {refName}</caption>
          <thead>
            <tr>
              <th className="c-rank">#</th>
              <th className="c-name sticky-col">Judge</th>
              {cols.map((c) => <th key={c.key} className={`r h-${c.group}${sortKey === c.key ? ' sorted' : ''}`}><button type="button" onClick={() => onSort(c)}>{c.label}</button></th>)}
              <th className="r h-gap" title="Math minus general accuracy, in points">Gap</th>
            </tr>
          </thead>
          <tbody>
            {!refJ && (
              <tr className="ref-row">
                <td className="c-rank" />
                <td className="c-name sticky-col"><span className="jcell"><span className="ref-tag">Median</span><span className="faint tiny">of {pool.length} judges</span></span></td>
                {cols.map((c) => <td key={c.key} className="r dcell ref tnum">{fmt(refExact[c.key], 2)}</td>)}
                <td className="r gap-cell" />
              </tr>
            )}
            {rows.map((j) => {
              const isRef = refJ && j.key === refJ.key
              const b = bud(j)
              return (
                <tr key={j.key} data-flip={j.key} className={isRef ? 'ref-row' : undefined} onClick={() => { window.location.hash = `#/judge/${j.key}` }}>
                  <td className="c-rank tnum">{rankAt(j, budget) ?? '·'}</td>
                  <td className="c-name sticky-col">
                    <span className="jcell">
                      <MarkIcon fam={j.family} size={12} open={b !== String(QUICK)} ring={j.kind === 'ladder'} />
                      <a href={`#/judge/${j.key}`} onClick={(e) => e.stopPropagation()}>{j.name}</a>
                      {isRef && <span className="ref-tag">Reference</span>}
                    </span>
                  </td>
                  {cols.map((c) => {
                    const v = b ? c.get(j.key, b) : null
                    if (v == null) return <td key={c.key} className="r dcell none">—</td>
                    if (isRef) return <td key={c.key} className="r dcell ref tnum" title={`${j.name}: ${fmt(v)}%`}>{fmt(v)}</td>
                    const d = c.exact(j.key, b) - refExact[c.key]
                    const s = shade(d)
                    return (
                      <td key={c.key} className={`r dcell tnum ${d > 0 ? 'pos' : 'neg'}${s > 0.5 ? ' strong' : ''}`} style={{ '--t': s.toFixed(3) }}
                        title={`${j.name}: ${fmt(v)}% · ${refName}: ${fmt(refExact[c.key])}%`}>
                        {Math.abs(d) < 0.005 ? '0.00' : signed(d, 2)}
                      </td>
                    )
                  })}
                  <td className="r tnum gap-cell">{b ? fmt(res.domainMeans[j.key]?.[b]?.gap, 1) : '—'}</td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
    </>
  )
}

// The four leading judges, marked by one bracket when they sit together.
function useBracket(box, keys, deps) {
  const [b, setB] = useState(null)
  useLayoutEffect(() => {
    const host = box.current
    if (!host) return setB(null)
    const trs = [...host.querySelectorAll('tbody tr[data-flip]')]
    const idx = trs.map((tr, i) => (keys.includes(tr.dataset.flip) ? i : -1)).filter((i) => i >= 0)
    if (idx.length !== keys.length || idx[idx.length - 1] - idx[0] !== idx.length - 1) return setB(null)
    const first = trs[idx[0]]
    const last = trs[idx[idx.length - 1]]
    setB({ top: first.offsetTop + 6, height: last.offsetTop + last.offsetHeight - first.offsetTop - 12 })
  }, deps)
  return b
}

export default function Leaderboard({ route }) {
  const res = useResults()
  const [st, setSt] = useState(() => readQuery(route.query))
  const [drawer, setDrawer] = useState(() => readQuery(route.query).compare.length >= 2)
  const [openCard, setOpenCard] = useState(null)
  const [heatSort, setHeatSort] = useState(null)
  const [copied, setCopied] = useState(false)
  const [sheet, setSheet] = useState(null)
  const tbody = useRef(null)
  const tableBox = useRef(null)
  const first = useRef(true)
  useEffect(() => {
    if (first.current) { first.current = false; return }
    const next = readQuery(route.query)
    setSt(next)
    if (next.compare.length >= 2) setDrawer(true)
  }, [route.raw])
  const update = (patch) => {
    const next = { ...st, ...patch }
    setSt(next)
    replaceHash('/leaderboard', toQuery(next))
  }

  const getters = { acc: (j) => accAt(j, st.budget), ifr: (j) => ifrAt(j, st.budget), cost: (j) => costAt(j, st.budget), rank: (j) => rankAt(j, st.budget) }
  const getter = (key) => getters[key] || COLS.find((c) => c.key === key).get
  const rows = useMemo(() => {
    const t = st.text.trim().toLowerCase()
    const list = judges.filter((j) =>
      (!st.fams.length || st.fams.includes(j.family)) &&
      (st.kind === 'all' || j.kind === st.kind) &&
      (!t || j.name.toLowerCase().includes(t) || j.key.includes(t) || j.weights.toLowerCase().includes(t)))
    const g = heatSort && st.view === 'datasets' ? heatSort.get : getter(st.tab === 'judges' ? st.sort : 'acc')
    const sign = st.dir === 'asc' ? 1 : -1
    const sorted = [...list].sort((a, b) => {
      const va = g(a)
      const vb = g(b)
      if (va == null && vb == null) return (rankAt(a, st.budget) ?? a.rank ?? 99) - (rankAt(b, st.budget) ?? b.rank ?? 99)
      if (va == null) return 1
      if (vb == null) return -1
      if (va < vb) return -sign
      if (va > vb) return sign
      return (rankAt(a, st.budget) ?? 99) - (rankAt(b, st.budget) ?? 99)
    })
    return [...sorted.filter((j) => j.kind !== 'ladder'), ...sorted.filter((j) => j.kind === 'ladder')]
  }, [st, heatSort])
  useFlip(tbody, [rows.map((j) => j.key).join(), st.view])

  // coming back from a judge page: that judge's row takes the shared names
  useLayoutEffect(() => {
    const k = lastJudge()
    if (!k) return
    setLastJudge(null)
    const el = [...document.querySelectorAll(`table.lb [data-flip="${k}"], .lb-cards [data-key="${k}"]`)].find((x) => x.offsetParent !== null)
    tagJudge(el)
    const t = setTimeout(clearTags, 700)
    return () => clearTimeout(t)
  }, [])
  const openJudge = (e, key) => {
    tagJudge(e.currentTarget.closest('[data-flip], [data-key]'))
    window.location.hash = `#/judge/${key}`
  }

  const best = useMemo(() => {
    const out = {}
    for (const k of ['acc', 'acc10', 'acc8192', 'se', 'mt', 'ifr']) out[k] = Math.max(...judges.map(getter(k)).filter((v) => v != null))
    out.cost = Math.min(...judges.map((j) => costAt(j, st.budget)).filter((v) => v != null))
    return out
  }, [st.budget])
  // the accuracy column's dot strip: one labelled axis, no bars
  const accVals = judges.map((j) => accAt(j, st.budget)).filter((v) => v != null)
  const lo = Math.floor(Math.min(...accVals) / 10) * 10
  const hi = Math.ceil(Math.max(...accVals) / 5) * 5
  const axTicks = []
  for (let t = lo; t <= hi; t += 10) axTicks.push(t)
  const pos = (v) => `${((v - lo) / (hi - lo)) * 100}%`
  const bracket = useBracket(tableBox, claims.tostGroup, [rows.map((j) => j.key).join(), st.view, st.budget])
  const showBracket = bracket && st.view === 'overview' && st.budget === 'best'

  const onSort = (key) => {
    setHeatSort(null)
    if (st.sort === key) update({ dir: st.dir === 'desc' ? 'asc' : 'desc' })
    else update({ sort: key, dir: COLS.find((c) => c.key === key)?.asc ? 'asc' : 'desc' })
  }
  const toggleFam = (f) => update({ fams: st.fams.includes(f) ? st.fams.filter((x) => x !== f) : [...st.fams, f] })
  const toggleCompare = (k) => {
    const has = st.compare.includes(k)
    if (!has && st.compare.length >= MAX_COMPARE) return
    update({ compare: has ? st.compare.filter((x) => x !== k) : [...st.compare, k] })
  }
  const filtered = st.fams.length > 0 || st.text || st.kind !== 'all'

  const exportRows = () => rows.map((j) => {
    const b = budgetOf(j, st.budget)
    return {
      rank: rankAt(j, st.budget) ?? '', judge: j.name, key: j.key, kind: j.kind, family: j.family, size_b: j.params,
      budget_used: b ?? '', [`accuracy_${st.budget === 'best' ? 'better_budget' : `b${st.budget}`}`]: accAt(j, st.budget),
      ifr: ifrAt(j, st.budget), est_cost_per_1k_usd: costAt(j, st.budget),
      acc_b10: j.acc10, acc_b8192: j.acc8192, delta_points: j.delta, summeval_rho: j.se, mtbench_rho: j.mt,
      ...(st.view === 'datasets' && res ? Object.fromEntries(DS.map((d, i) => [d.key, b == null ? null : res.perDataset[j.key]?.[String(b)]?.[i] ?? null])) : {}),
    }
  })
  const fileBase = `slmjury-leaderboard-${st.view}-${st.budget === 'best' ? 'better-budget' : `b${st.budget}`}`
  const toCsv = () => {
    const data = exportRows()
    const head = Object.keys(data[0] || {})
    return [head.join(','), ...data.map((r) => head.map((h) => csvCell(r[h])).join(','))].join('\n') + '\n'
  }
  const copyLink = async () => { await copyText(window.location.href); setCopied(true); setTimeout(() => setCopied(false), 1500) }

  return (
    <>
      <section className="page-head lb-head">
        <div className="wrap-wide lb-head-row">
          <div>
            <h1 className="h-display">Leaderboard</h1>
            <p className="lead">{LEADS[st.tab]}</p>
          </div>
          <dl className="lb-facts">
            <div><dt>Leading group</dt><dd>{claims.tostGroup.length} judges, pairwise equivalent within ±{claims.margin} points</dd></div>
          </dl>
        </div>
      </section>

      <section className="wrap-wide lb-wrap">
        <TabBar tab={st.tab} onChange={(tab) => tab !== st.tab && update({ tab, sort: tab === 'judges' ? 'acc' : TAB_DEFAULT_SORT[tab], dir: 'desc' })} />
        {st.tab !== 'judges' ? (res ? (
          <TabPanel tab={st.tab} res={res} sort={st.sort} dir={st.dir} pb={st.pb} copied={copied} onCopy={copyLink}
            onBudget={(pb) => update({ pb })}
            onSort={(key) => (st.sort === key ? update({ dir: st.dir === 'desc' ? 'asc' : 'desc' }) : update({ sort: key, dir: ASC_KEYS.includes(key) ? 'asc' : 'desc' }))} />
        ) : <div className="skeleton" style={{ height: 480 }} />) : (<>
        <div className="lb-mbar">
          <label className="lb-select">
            <span className="sr-only">View</span>
            <select value={st.view} onChange={(e) => update({ view: e.target.value })}>
              <option value="overview">Overview</option>
              <option value="datasets">Per dataset</option>
            </select>
            <Icon.chevronDown />
          </label>
          <label className="lb-select">
            <span className="sr-only">Budget</span>
            <select value={String(st.budget)} onChange={(e) => update({ budget: e.target.value === 'best' ? 'best' : Number(e.target.value) })}>
              <option value="best">Better budget</option>
              <option value={String(QUICK)}>B = {QUICK}</option>
              <option value={String(REASONED)}>B = {R}</option>
            </select>
            <Icon.chevronDown />
          </label>
          <button type="button" className="btn btn-sm" aria-haspopup="dialog" onClick={() => setSheet('filter')}>
            <Icon.search /> Filter{filtered ? ` (${rows.length})` : ''}
          </button>
          <span className="spacer" />
          <button type="button" className="icon-btn" aria-haspopup="dialog" aria-label="Download or share" onClick={() => setSheet('more')}><Icon.download /></button>
        </div>
        {sheet && (
          <Sheet title={sheet === 'filter' ? 'Filter judges' : 'Download or share'} onClose={() => setSheet(null)}>
            {sheet === 'filter' ? (
              <div className="sheet-stack">
                <Segmented label="Judge size" value={st.kind} onChange={(kind) => update({ kind })}
                  options={[{ value: 'all', label: 'All' }, { value: 'small', label: 'Small' }, { value: 'large', label: 'Large' }]} />
                <label className="search">
                  <span className="sr-only">Search judges</span>
                  <Icon.search />
                  <input type="search" placeholder="Search judges or weights" value={st.text} onChange={(e) => update({ text: e.target.value })} />
                </label>
                <div className="chips" role="group" aria-label="Filter by family">
                  {FAMILIES.map((f) => (
                    <button key={f.key} type="button" className="chip" aria-pressed={st.fams.includes(f.key)} onClick={() => toggleFam(f.key)}>
                      <MarkIcon fam={f.key} size={12} /> {f.label}
                    </button>
                  ))}
                </div>
                <div className="row">
                  <button type="button" className="btn btn-primary" onClick={() => setSheet(null)}>Show {rows.length} judges</button>
                  {filtered && <button type="button" className="btn btn-ghost" onClick={() => update({ fams: [], text: '', kind: 'all' })}>Clear filters</button>}
                </div>
              </div>
            ) : (
              <div className="sheet-stack">
                <button type="button" className="btn" onClick={() => download(`${fileBase}.csv`, toCsv(), 'text/csv')}><Icon.download /> Download CSV</button>
                <button type="button" className="btn" onClick={() => download(`${fileBase}.json`, JSON.stringify(exportRows(), null, 2) + '\n', 'application/json')}><Icon.download /> Download JSON</button>
                <button type="button" className="btn" onClick={copyLink}>{copied ? <Icon.check /> : <Icon.link />} {copied ? 'Copied' : 'Copy link to this view'}</button>
              </div>
            )}
          </Sheet>
        )}
        <div className="lb-toolbar">
          <Segmented label="View" value={st.view} onChange={(view) => update({ view })}
            options={[{ value: 'overview', label: <><Icon.table /> Overview</> }, { value: 'datasets', label: <><Icon.grid /> Per dataset</> }]} />
          <Segmented label="Budget" value={st.budget} onChange={(budget) => update({ budget })}
            options={[{ value: 'best', label: 'Better budget' }, { value: QUICK, label: `B = ${QUICK}` }, { value: REASONED, label: `B = ${R}` }]} />
          <Segmented label="Judge size" value={st.kind} onChange={(kind) => update({ kind })}
            options={[{ value: 'all', label: 'All' }, { value: 'small', label: 'Small' }, { value: 'large', label: 'Large' }]} />
          <label className="search">
            <span className="sr-only">Search judges</span>
            <Icon.search />
            <input type="search" placeholder="Search judges or weights" value={st.text} onChange={(e) => update({ text: e.target.value })} />
          </label>
          <span className="spacer" />
          <div className="lb-actions">
            <button type="button" className="btn btn-sm" onClick={() => download(`${fileBase}.csv`, toCsv(), 'text/csv')}><Icon.download /> CSV</button>
            <button type="button" className="btn btn-sm" onClick={() => download(`${fileBase}.json`, JSON.stringify(exportRows(), null, 2) + '\n', 'application/json')}><Icon.download /> JSON</button>
            <button type="button" className="btn btn-sm" onClick={copyLink} aria-live="polite">{copied ? <Icon.check /> : <Icon.link />} {copied ? 'Copied' : 'Copy link'}</button>
          </div>
        </div>
        <div className="chips lb-chips" role="group" aria-label="Filter by family">
          {FAMILIES.map((f) => (
            <button key={f.key} type="button" className="chip" aria-pressed={st.fams.includes(f.key)} onClick={() => toggleFam(f.key)}>
              <MarkIcon fam={f.key} size={12} /> {f.label}
            </button>
          ))}
          {filtered && <button type="button" className="chip ghost" onClick={() => update({ fams: [], text: '', kind: 'all' })}><Icon.close /> Clear filters</button>}
          <span className="lb-count faint">{rows.length} of {judges.length} judges</span>
        </div>

        {st.view === 'overview' ? (
          <div className="lb-scroll" ref={tableBox}>
            {showBracket && (
              <div className="lb-bracket" style={{ top: bracket.top, height: bracket.height }} aria-hidden="true" />
            )}
            <table className="lb">
              <caption className="sr-only">Judge leaderboard. Accuracy in percent; SummEval and MT-Bench columns are Spearman correlations.</caption>
              <thead>
                <tr>
                  <th className="c-check"><span className="sr-only">Compare</span></th>
                  {COLS.map((c) => {
                    const active = st.sort === c.key
                    const label = c.key === 'acc' ? `Accuracy, ${BUDGET_NAME[st.budget]}` : c.key === 'cost' ? `$ / 1k, ${BUDGET_NAME[st.budget]}` : c.label
                    return (
                      <th key={c.key} className={`${c.r ? 'r ' : ''}${c.cls || ''}${c.key === 'name' ? ' sticky-col' : ''}`} aria-sort={active ? (st.dir === 'asc' ? 'ascending' : 'descending') : 'none'}>
                        <button type="button" onClick={() => onSort(c.key)} title={c.title}>
                          {label}
                          <span className={`sort-ic${active ? ' on' : ''}`}>{active ? (st.dir === 'asc' ? <Icon.sortUp /> : <Icon.sortDown />) : <Icon.sort />}</span>
                        </button>
                        {c.key === 'acc' && <span className="acc-axis" aria-hidden="true">{axTicks.map((t) => <i key={t} style={{ left: pos(t) }}>{t}</i>)}</span>}
                      </th>
                    )
                  })}
                </tr>
              </thead>
              <tbody ref={tbody}>
                {rows.map((j) => {
                  const a = accAt(j, st.budget)
                  const ifr = ifrAt(j, st.budget)
                  const bb = budgetOf(j, st.budget)
                  const cell = (v, nd, key) => (v == null ? <span className="none">—</span> : <span className={v === best[key] ? 'best' : ''}>{fmt(v, nd)}</span>)
                  const checked = st.compare.includes(j.key)
                  return (
                    <tr key={j.key} data-flip={j.key} className={`${j.kind === 'ladder' ? 'ladder' : ''}${checked ? ' picked' : ''}`}
                      onClick={(e) => openJudge(e, j.key)}>
                      <td className="c-check" onClick={(e) => e.stopPropagation()}>
                        <input type="checkbox" checked={checked} onChange={() => toggleCompare(j.key)} disabled={!checked && st.compare.length >= MAX_COMPARE}
                          aria-label={`Compare ${j.name}`} />
                      </td>
                      <td className="c-rank tnum">{rankAt(j, st.budget) ?? '·'}</td>
                      <td className="c-name sticky-col">
                        <span className="jcell">
                          <span data-vt="mark" className="vt-mark"><MarkIcon fam={j.family} size={14} open={bb !== QUICK} ring={j.kind === 'ladder'} /></span>
                          <a href={`#/judge/${j.key}`} data-vt="name" onClick={(e) => { e.stopPropagation(); e.preventDefault(); openJudge(e, j.key) }}>{j.name}</a>
                          {j.thinks === 'always' && <span className="tag" title="Always reasons, so it runs at the reasoned budget only">†</span>}
                          {j.kind === 'ladder' && <span className="tag">ladder</span>}
                        </span>
                      </td>
                      <td className="r c-acc">
                        {a == null ? <span className="none">—</span> : (
                          <span className="acc-cell">
                            <span className="acc-strip" aria-hidden="true">
                              {axTicks.map((t) => <i key={t} style={{ left: pos(t) }} />)}
                              <b style={{ left: pos(a) }}><MarkIcon fam={j.family} size={11} open={bb !== QUICK} /></b>
                            </span>
                            <span className={`tnum${a === best.acc ? ' best' : ''}`}>{fmt(a)}</span>
                          </span>
                        )}
                      </td>
                      <td className="r tnum">{j.size}</td>
                      <td className="r tnum">{cell(j.acc10, 2, 'acc10')}</td>
                      <td className="r tnum">{cell(j.acc8192, 2, 'acc8192')}</td>
                      <td className="r tnum">{j.delta == null ? <span className="none">—</span> : signed(j.delta)}</td>
                      <td className="r tnum">{cell(ifr, 2, 'ifr')}</td>
                      <td className="r tnum">{cell(j.se, 3, 'se')}</td>
                      <td className="r tnum">{cell(j.mt, 3, 'mt')}</td>
                      <td className="r tnum">{costAt(j, st.budget) != null ? <span className={costAt(j, st.budget) === best.cost ? 'best' : ''}>${fmt(costAt(j, st.budget))}</span> : <span className="none">—</span>}</td>
                    </tr>
                  )
                })}
                {!rows.length && <tr><td colSpan={COLS.length + 1} className="empty">No judge matches these filters. <button type="button" className="link-btn" onClick={() => update({ fams: [], text: '', kind: 'all' })}>Clear them</button></td></tr>}
              </tbody>
            </table>
          </div>
        ) : res ? (
          <>
            <div className="diff-bar">
              <span className="diff-l">Points above or below</span>
              <Segmented label="Reference" value={st.ref === 'median' ? 'median' : 'judge'} size="sm"
                onChange={(m) => update({ ref: m === 'median' ? 'median' : 'phi4-14b' })}
                options={[{ value: 'judge', label: 'A judge' }, { value: 'median', label: 'The column median' }]} />
              {st.ref !== 'median' && (
                <JudgePicker value={st.ref} onChange={(k) => update({ ref: k })} label="Reference judge" className="diff-pick" />
              )}
              <span className="diff-key" aria-hidden="true">
                <span><i className="sw neg" />below</span><span><i className="sw pos" />above</span>
              </span>
            </div>
            <div ref={tbody}><DiffTable rows={rows} budget={st.budget} res={res} refKey={st.ref} sortKey={heatSort?.key}
              onSort={(c) => setHeatSort((h) => (h?.key === c.key ? null : { key: c.key, get: (j) => c.get(j.key, String(budgetOf(j, st.budget))) }))} /></div>
          </>
        ) : <div className="skeleton" style={{ height: 600 }} />}

        {/* phones: one card per judge */}
        {st.view === 'overview' && <ol className="lb-cards" aria-label={`Judges by accuracy, ${BUDGET_NAME[st.budget]}`}>
          {rows.map((j) => {
            const a = accAt(j, st.budget)
            const bb = budgetOf(j, st.budget)
            const open = openCard === j.key
            return (
              <li key={j.key} data-key={j.key} className={`lb-card${open ? ' open' : ''}`}>
                <button type="button" className="lc-main" aria-expanded={open} onClick={() => setOpenCard(open ? null : j.key)}>
                  <span className="lc-rank tnum">{rankAt(j, st.budget) ?? '·'}</span>
                  <span className="lc-name"><span data-vt="mark" className="vt-mark"><MarkIcon fam={j.family} size={13} open={bb !== QUICK} ring={j.kind === 'ladder'} /></span> <span data-vt="name">{j.name}</span> <span className="faint">{j.size}</span></span>
                  <span className="lc-acc"><span className="tnum">{a == null ? '—' : `${fmt(a)}%`}</span><small>accuracy{bb != null ? `, B = ${count(bb)}` : ''}</small></span>
                  <span className="lc-strip" aria-hidden="true">
                    {axTicks.map((t) => <i key={t} style={{ left: pos(t) }}><em>{t}</em></i>)}
                    {a != null && <b style={{ left: pos(a) }}><MarkIcon fam={j.family} size={11} open={bb !== QUICK} /></b>}
                  </span>
                </button>
                {open && (
                  <div className="lc-more">
                    <dl>
                      <div><dt>B = {QUICK}</dt><dd>{j.acc10 == null ? '—' : `${fmt(j.acc10)}%`}</dd></div>
                      <div><dt>B = {R}</dt><dd>{fmt(j.acc8192)}%</dd></div>
                      <div><dt>IFR</dt><dd>{ifrAt(j, st.budget) == null ? '—' : `${fmt(ifrAt(j, st.budget))}%`}</dd></div>
                      <div><dt>SummEval ρ</dt><dd>{fmt(j.se, 3)}</dd></div>
                      <div><dt>MT-Bench ρ</dt><dd>{fmt(j.mt, 3)}</dd></div>
                      <div><dt>$ per 1k{bb != null ? `, B = ${count(bb)}` : ''}</dt><dd>{costAt(j, st.budget) != null ? `$${fmt(costAt(j, st.budget))}` : '—'}</dd></div>
                    </dl>
                    <div className="row">
                      <a className="btn btn-sm" href={`#/judge/${j.key}`} onClick={(e) => { e.preventDefault(); openJudge(e, j.key) }}>Judge page <Icon.arrow /></a>
                      <label className="chip"><input type="checkbox" checked={st.compare.includes(j.key)} onChange={() => toggleCompare(j.key)} /> Compare</label>
                    </div>
                  </div>
                )}
              </li>
            )
          })}
        </ol>}

        {st.view === 'overview' ? (
          <p className="lb-note">
            Accuracy is agreement with the oracle (%). SummEval and MT-Bench are Spearman’s ρ, and MT-Bench uses one reference
            scorer (Qwen3.5-397B-A17B). Bold is the best in a column. † always reasons, so it runs at B = {R} only. Open marks
            are B = {R}. The gold bracket marks the {claims.tostGroup.length} judges that are pairwise equivalent within
            ±{claims.margin} points. Cost is an estimate per 1,000 verdicts, at the same budget as the accuracy.
          </p>
        ) : (
          <p className="lb-note">
            Each cell is a judge’s accuracy minus the reference’s on that dataset, in points. The reference row shows its
            accuracy (%). Each judge is at the {BUDGET_NAME[st.budget]}{st.budget === QUICK ? ` (B = ${R} for judges that always reason)` : ''}.
            Shading starts at {SHADE_FROM} points and is full at {SHADE_FULL}. Gap is math minus general accuracy, in points.
          </p>
        )}
        </>)}
      </section>

      {st.tab === 'judges' && st.compare.length > 0 && !drawer && (
        <div className="compare-tray" role="region" aria-label="Judges to compare">
          <span className="ct-label">Compare</span>
          {st.compare.map((k) => {
            const j = judges.find((x) => x.key === k)
            return <button key={k} type="button" className="chip" onClick={() => toggleCompare(k)} aria-label={`Remove ${j.name}`}><MarkIcon fam={j.family} size={11} /> {j.name} <Icon.close /></button>
          })}
          <span className="ct-hint faint">{st.compare.length < 2 ? 'Pick one more' : `${st.compare.length} of ${MAX_COMPARE}`}</span>
          <button type="button" className="btn btn-primary btn-sm" disabled={st.compare.length < 2} onClick={() => setDrawer(true)}>Compare <Icon.arrow /></button>
          <button type="button" className="icon-btn" onClick={() => update({ compare: [] })} aria-label="Clear the comparison"><Icon.close /></button>
        </div>
      )}
      {st.tab === 'judges' && drawer && st.compare.length >= 2 && (
        <CompareDrawer keys={st.compare} res={res} onClose={() => setDrawer(false)} onRemove={(k) => update({ compare: st.compare.filter((x) => x !== k) })} />
      )}
    </>
  )
}
