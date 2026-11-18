import { useEffect, useMemo, useRef, useState } from 'react'
import ChartFrame from '../charts/Frame'
import Stage from '../charts/Stage'
import { TipHead, TipRow } from '../charts/kit'
import { DODGE } from '../charts/SizeScatter'
import { Icon, MarkIcon, Segmented } from '../components/ui'
import { claims, facts, famLabel, judges, useResults } from '../lib/data'
import { count, fmt } from '../lib/format'
import { replaceHash } from '../lib/router'
import '../styles/content.css'
import '../styles/pick.css'

const [QUICK, REASONED] = [facts.budgets[0], facts.budgets[facts.budgets.length - 1]]
const SIZES = [...new Set(judges.map((j) => j.params))].sort((a, b) => a - b)
const sizeLabel = (p) => judges.find((j) => j.params === p).size
const SIZE_TICKS = [0.5, 1, 2, 5, 10, 20, 50, 100]

const TASKS = [
  { value: 'verify', label: 'Answers' },
  { value: 'summeval', label: 'Summaries' },
  { value: 'mtbench', label: 'Chats' },
]
const TASK_NOTE = {
  verify: 'Math, science or multiple choice, against a reference answer.',
  summeval: `Rate summaries from 1 to ${facts.scoreScale}, checked against human scores (SummEval).`,
  mtbench: 'Rate multi-turn answers, checked against a large reference model (MT-Bench).',
}
const ADVICE = {
  verify: 'With a reference answer to check against, a small judge giving a quick verdict is enough.',
  summeval: 'On SummEval the strongest small judges match or beat most large ones. Validate on human scores where you can.',
  mtbench: `MT-Bench is scored against one large reference model over ${facts.nConversations} conversations per student, and the top judges’ intervals overlap.`,
}
const DOMAINS = [
  { value: 'all', label: 'All' },
  { value: 'math', label: 'Math' },
  { value: 'science', label: 'Science' },
  { value: 'general', label: 'General' },
]

export default function Pick({ route }) {
  const q = route.query
  const res = useResults()
  const [task, setTask] = useState(TASKS.some((t) => t.value === q.get('task')) ? q.get('task') : 'verify')
  const [maxI, setMaxI] = useState(() => {
    const n = SIZES.indexOf(Number(q.get('max')))
    const d = SIZES.indexOf(parseFloat(claims.phiSize))
    return n >= 0 ? n : d >= 0 ? d : SIZES.length - 1
  })
  const [budget, setBudget] = useState(q.get('budget') === 'any' ? 'any' : 'quick')
  const [domain, setDomain] = useState(DOMAINS.some((d) => d.value === q.get('domain')) ? q.get('domain') : 'all')
  const first = useRef(true)
  useEffect(() => {
    if (first.current) { first.current = false; return }
    if (TASKS.some((t) => t.value === q.get('task'))) setTask(q.get('task'))
    const n = SIZES.indexOf(Number(q.get('max')))
    if (n >= 0) setMaxI(n)
    setBudget(q.get('budget') === 'any' ? 'any' : 'quick')
    setDomain(DOMAINS.some((d) => d.value === q.get('domain')) ? q.get('domain') : 'all')
  }, [route.raw])
  const sync = (patch) => replaceHash('/pick', { task, max: SIZES[maxI], budget, domain, ...patch })
  const maxP = SIZES[maxI]

  // every judge's score for the chosen task; the ranking keeps the ones that fit
  const scored = useMemo(() => judges.map((j) => {
    if (task !== 'verify') {
      const v = j[task === 'summeval' ? 'se' : 'mt']
      return v == null ? null : { j, v }
    }
    const opts = budget === 'quick' ? [QUICK] : facts.budgets
    const cands = opts.map((b) => {
      const v = domain === 'all' ? (b === QUICK ? j.acc10 : j.acc8192) : res?.domainMeans[j.key]?.[String(b)]?.[domain]
      return v == null ? null : { b, v }
    }).filter(Boolean)
    if (!cands.length) return null
    const best = cands.reduce((a, c) => (c.v > a.v ? c : a))
    return { j, v: best.v, b: best.b, cost: res?.cost.find((c) => c.key === j.key && c.budget === best.b) }
  }).filter(Boolean), [task, budget, domain, res])
  const picks = scored.filter((p) => p.j.params <= maxP).sort((a, b) => b.v - a.v)
  const top = picks.slice(0, 3)
  const unit = task === 'verify' ? '%' : ''
  const nd = task === 'verify' ? 2 : 3
  const metric = task === 'verify' ? (domain === 'all' ? 'Accuracy' : `${DOMAINS.find((d) => d.value === domain).label} accuracy`) : task === 'summeval' ? 'SummEval ρ' : 'MT-Bench ρ'
  const needsRes = task === 'verify' && domain !== 'all' && !res
  const rankOf = Object.fromEntries(top.map((p, n) => [p.j.key, n + 1]))

  const phiPick = scored.find((p) => p.j.key === 'phi4-14b')
  const phiBand = task === 'verify' && domain === 'all' && phiPick ? [phiPick.v - parseFloat(claims.margin), phiPick.v + parseFloat(claims.margin)] : null
  const spec = useMemo(() => {
    if (!scored.length) return null
    const vals = scored.map((p) => p.v)
    const lo = Math.min(...vals)
    const hi = Math.max(...vals)
    const padv = (hi - lo) * 0.08 || 0.05
    const step = task === 'verify' ? (hi - lo > 30 ? 10 : 5) : 0.1
    const ticks = []
    for (let v = Math.ceil((lo - padv) / step) * step; v <= hi + padv + 1e-9; v += step) ticks.push(+v.toFixed(4))
    return {
      key: `${task}-${budget}-${domain}`,
      alt: `${metric} against size for every judge; judges above ${sizeLabel(maxP)} are faded.`,
      x: { domain: [0.45, 170], log: true, ticks: SIZE_TICKS, format: (t) => `${t}B`, title: 'Parameters (log scale)' },
      y: { domain: [lo - padv, hi + padv], ticks, format: (v) => fmt(v, task === 'verify' ? 0 : 1), title: metric + (unit ? ' (%)' : '') },
      marks: scored.map((p) => ({
        id: p.j.key, key: p.j.key, x: p.j.params, dx: DODGE[p.j.key] || 0, y: p.v, fam: p.j.family, open: p.b === REASONED, ring: p.j.kind === 'ladder', r: 6.5,
        dim: p.j.params > maxP, strong: !!rankOf[p.j.key], label: rankOf[p.j.key] ? String(rankOf[p.j.key]) : p.j.name,
        tip: () => <><TipHead fam={p.j.family} name={p.j.name} open={p.b === REASONED} /><TipRow k="Size" v={p.j.size} /><TipRow k={metric} v={`${fmt(p.v, nd)}${unit}`} />{p.b && <TipRow k="Budget" v={`B = ${count(p.b)}`} />}</>,
      })),
      areas: maxP < SIZES[SIZES.length - 1] ? [{ id: 'over', x0: maxP * 1.12, className: 'st-area-over', label: 'too large', labelAnchor: 'start' }] : [],
      // the ±margin band around Phi-4, on the accuracy view that has it
      rules: phiBand ? [{ id: 'eq-hi', axis: 'y', v: phiBand[1], className: 'st-rule-eq' }, { id: 'eq-lo', axis: 'y', v: phiBand[0], className: 'st-rule-eq' }] : [],
      table: {
        columns: [{ key: 'judge', label: 'Judge', align: 'left' }, { key: 'size', label: 'Size', align: 'left' }, { key: 'v', label: metric }, { key: 'fits', label: `Up to ${sizeLabel(maxP)}`, align: 'left' }],
        rows: [...scored].sort((a, b) => b.v - a.v).map((p) => ({ judge: p.j.name, size: p.j.size, v: p.v, fits: p.j.params <= maxP ? 'yes' : 'no' })),
      },
    }
  }, [scored, maxP, phiBand?.[0]])

  const t0 = top[0]
  // phones: a bar at the bottom keeps the top pick in view while the inputs change
  const outRef = useRef(null)
  const [outSeen, setOutSeen] = useState(false)
  useEffect(() => {
    if (!outRef.current || !('IntersectionObserver' in window)) return
    const io = new IntersectionObserver(([e]) => setOutSeen(e.isIntersecting), { threshold: 0.25 })
    io.observe(outRef.current)
    return () => io.disconnect()
  }, [])
  return (
    <>
      <section className="page-head pick-head">
        <div className="wrap">
          <h1 className="h-display">Which judge should you run?</h1>
          <p className="lead">Set the task and what you can serve. The ranking uses only the study’s measurements.</p>
        </div>
      </section>
      <section className="wrap pick">
        <div className="pick-controls">
          <div className="pc">
            <div className="pc-l">What are you judging?</div>
            <Segmented label="Task" value={task} onChange={(v) => { setTask(v); sync({ task: v }) }} options={TASKS} />
            <div className="pc-d">{TASK_NOTE[task]}</div>
          </div>
          <div className="pc">
            <label className="pc-l" htmlFor="maxsize">Largest judge you can serve <span className="pc-v tnum">{sizeLabel(maxP)}</span></label>
            <input id="maxsize" type="range" min={0} max={SIZES.length - 1} step={1} value={maxI} aria-valuetext={sizeLabel(maxP)}
              style={{ '--fill': `${(maxI / (SIZES.length - 1)) * 100}%` }}
              onChange={(e) => { const n = Number(e.target.value); setMaxI(n); sync({ max: SIZES[n] }) }} />
            <div className="range-labels"><span>{sizeLabel(SIZES[0])}</span><span>{sizeLabel(SIZES[SIZES.length - 1])}</span></div>
          </div>
          {task === 'verify' && (
            <>
              <div className="pc">
                <div className="pc-l">Token budget</div>
                <Segmented label="Budget" value={budget} onChange={(v) => { setBudget(v); sync({ budget: v }) }}
                  options={[{ value: 'quick', label: `Quick (B = ${QUICK})` }, { value: 'any', label: `Up to ${count(REASONED)}` }]} />
              </div>
              <div className="pc">
                <div className="pc-l">Domain</div>
                <Segmented label="Domain" value={domain} onChange={(v) => { setDomain(v); sync({ domain: v }) }} options={DOMAINS} />
              </div>
            </>
          )}
        </div>

        {t0 && !needsRes && (
          <a className={`pick-bar${outSeen ? ' away' : ''}`} href={`#/judge/${t0.j.key}`} aria-hidden={outSeen || undefined} tabIndex={outSeen ? -1 : undefined}>
            <span className="pb-k">Top pick</span>
            <span className="pb-name"><MarkIcon fam={t0.j.family} size={13} open={t0.b === REASONED} /> {t0.j.name}</span>
            <span className="pb-v tnum">{fmt(t0.v, nd)}{unit}</span>
          </a>
        )}
        <div className="pick-out" aria-live="polite" ref={outRef}>
          {needsRes && <div className="skeleton" style={{ height: 220 }} />}
          {!needsRes && !t0 && <div className="pick-empty">No judge fits these limits. Allow a larger size or the reasoned budget.</div>}
          {!needsRes && t0 && (
            <>
              <a className="pick-top" href={`#/judge/${t0.j.key}`} key={`${task}-${t0.j.key}`}>
                <span className="pt-k">Top pick</span>
                <span className="pt-name"><MarkIcon fam={t0.j.family} size={18} open={t0.b === REASONED} /> {t0.j.name}</span>
                <span className="pt-v tnum">{fmt(t0.v, nd)}<small>{unit}</small></span>
                <span className="pt-meta">
                  {t0.j.size} · {famLabel(t0.j.family)}{t0.b ? ` · B = ${count(t0.b)}` : ''}{t0.cost ? ` · $${fmt(t0.cost.cost)} per 1,000 verdicts` : ''}
                </span>
                <span className="pt-m">{metric}</span>
                <span className="pt-go">Judge page <Icon.arrow /></span>
              </a>
              <ol className="pick-runners">
                {top.slice(1).map((p, n) => (
                  <li key={p.j.key}>
                    <a href={`#/judge/${p.j.key}`}>
                      <span className="pr-n tnum">{n + 2}</span>
                      <span className="pr-name"><MarkIcon fam={p.j.family} size={13} open={p.b === REASONED} /> {p.j.name} <span className="faint">{p.j.size}</span></span>
                      <span className="pr-v tnum">{fmt(p.v, nd)}{unit}</span>
                    </a>
                  </li>
                ))}
              </ol>
              <p className="pick-advice">{ADVICE[task]}</p>
            </>
          )}
        </div>

        <div className="pick-chart" data-reveal>
          {spec && (
            <ChartFrame id="pick" title={`${metric} against size`} sub={`Faded: judges larger than ${sizeLabel(maxP)}. Your top three are numbered.${phiBand ? ` Dashed lines: ±${claims.margin} points of Phi-4.` : ''}`} table={spec.table}>
              <Stage spec={spec} height={400} />
            </ChartFrame>
          )}
        </div>
      </section>
    </>
  )
}
