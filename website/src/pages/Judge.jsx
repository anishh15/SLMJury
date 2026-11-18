import { useEffect, useMemo, useState } from 'react'
import ChartFrame, { Lazy } from '../charts/Frame'
import Stage from '../charts/Stage'
import { LineChart } from '../charts/generic'
import { TipHead, TipRow } from '../charts/kit'
import { Icon, Mark, MarkIcon } from '../components/ui'
import MathText from '../components/MathText'
import { byKey, claims, core, facts, FAMILY_SLUG, famLabel, judges, ranked, statement, useResults } from '../lib/data'
import { cap, count, fmt, signed } from '../lib/format'
import { setLastJudge } from '../lib/vt'
import { sweepDelay, useMarksIn } from '../lib/motion'
import '../styles/content.css'
import '../styles/judge.css'

const [QUICK, REASONED] = [facts.budgets[0], facts.budgets[facts.budgets.length - 1]]
const R = count(REASONED)
const DS = Object.fromEntries(core.datasets.closed.map((d) => [d.key, d]))
const THINKS = { no: 'no thinking mode', reasoned: `thinking mode at B = ${R}`, always: `always reasons (B = ${R} only)` }
const nRanked = ranked.filter((x) => x.rank).length
const PROTO_LABEL = { reference_based: 'With the reference', reference_free: 'Without the reference', geval: 'G-Eval rubric' }
const ORD = ['first', 'second', 'third', 'fourth', 'fifth', 'sixth', 'seventh', 'eighth', 'ninth', 'tenth']
const ordinal = (n) => ORD[n - 1] ?? `${n}th`
const samples = import.meta.glob('../data/samples/*.json', { import: 'default' })

function useSample(key) {
  const [s, setS] = useState(null)
  useEffect(() => {
    setS(null)
    const load = samples[`../data/samples/${key}.json`]
    if (load) load().then(setS)
  }, [key])
  return s
}

// The judge's place among all judges on one metric: a strip of every judge,
// this one lit, its comparison ringed.
function Strip({ value, values, other, fam, lo, hi, band }) {
  const x = (v) => 6 + ((v - lo) / (hi - lo || 1)) * 188
  const [ref, entry] = useMarksIn()
  return (
    <svg className={`strip ${entry}`} viewBox="0 0 200 22" aria-hidden="true" ref={ref}>
      {band && <rect x={x(band[0])} width={x(band[1]) - x(band[0])} y="2" height="18" className="strip-band" />}
      <line x1="6" x2="194" y1="11" y2="11" className="strip-line" />
      {values.map((v, n) => <circle key={n} cx={x(v)} cy="11" r="2.3" className="strip-dot mk" style={{ '--mk-d': sweepDelay(x(v), 200, 360) }} />)}
      {other != null && <circle cx={x(other)} cy="11" r="5.5" className="strip-other mk" style={{ '--mk-d': '380ms' }} />}
      <g transform={`translate(${x(value)},11)`}><Mark fam={fam} r={5} className="mk" style={{ '--mk-d': '440ms' }} /></g>
    </svg>
  )
}

function Tile({ label, value, at, children }) {
  return (
    <div className="jt">
      <div className="jt-l">{label}</div>
      <div className="jt-v tnum">{value}{at && <span className="jt-at">{at}</span>}</div>
      {children && <div className="jt-c">{children}</div>}
    </div>
  )
}

function Clamp({ text, math }) {
  const [open, setOpen] = useState(false)
  const long = text.length > 200
  return (
    <div className={`clamp${open || !long ? ' open' : ''}`}>
      <p>{math ? <MathText text={text} /> : text}</p>
      {long && <button type="button" className="link-btn" onClick={() => setOpen((o) => !o)}>{open ? 'Show less' : 'Show all'}</button>}
    </div>
  )
}

function VerdictCard({ item, row }) {
  const [show, setShow] = useState(null)
  const budgets = Object.keys(row)
  return (
    <article className="vc">
      <header className="vc-head">
        <span className="pill">{item.datasetName}</span>
        <span className="faint tiny">{item.studentName}</span>
      </header>
      <Clamp text={item.question} math={item.dataset === 'math'} />
      <dl className="vc-ans">
        <div><dt>Reference</dt><dd className="mono">{item.dataset === 'math' ? <MathText text={item.reference} whole /> : item.reference}</dd></div>
        <div><dt>Student</dt><dd className="mono">{item.answer}</dd></div>
        <div><dt>Oracle</dt><dd>{item.oracle}</dd></div>
      </dl>
      <div className="vc-verdicts">
        {budgets.map((b) => {
          const v = row[b]
          const ok = v.verdict === item.oracle
          return (
            <div key={b} className={`vc-v${ok ? '' : ' off'}`}>
              <span className={`vc-ok ${ok ? 'yes' : 'no'}`} title={ok ? 'Agrees with the oracle' : 'Disagrees with the oracle'}>
                {ok ? <Icon.check /> : <Icon.close />}<span className="sr-only">{ok ? 'agrees with the oracle' : 'disagrees with the oracle'}</span>
              </span>
              <span className="vc-b tnum">B = {count(Number(b))}</span>
              <span className="vc-said">said {v.verdict}</span>
              <button type="button" className="link-btn vc-more" aria-expanded={show === b} onClick={() => setShow(show === b ? null : b)}>{show === b ? 'Hide' : 'Response'}</button>
            </div>
          )
        })}
      </div>
      {show && <pre className="vc-resp" tabIndex={0}>{row[show].response}</pre>}
    </article>
  )
}

export default function Judge({ route }) {
  const key = route.parts[1]
  const j = byKey[key]
  const res = useResults()
  const sample = useSample(key)
  useEffect(() => { if (j) setLastJudge(key) }, [key])
  const i = ranked.findIndex((x) => x.key === key)
  const prev = ranked[(i - 1 + ranked.length) % ranked.length]
  const next = ranked[(i + 1) % ranked.length]

  useEffect(() => {
    if (!j) return
    const onKey = (e) => {
      if (e.metaKey || e.ctrlKey || e.altKey || /^(INPUT|TEXTAREA|SELECT)$/.test(e.target.tagName) || document.querySelector('.palette')) return
      if (e.key === 'ArrowLeft') window.location.hash = `#/judge/${prev.key}`
      if (e.key === 'ArrowRight') window.location.hash = `#/judge/${next.key}`
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [key])

  const ctx = useMemo(() => {
    if (!j) return null
    // ranks are among the 22 judges; the ladder judge is not ranked
    const pool = judges.filter((x) => x.rank)
    const rankOf = (k) => (j.rank ? [...pool].sort((a, b) => b[k] - a[k]).findIndex((x) => x.key === key) + 1 : null)
    return { near: byKey[core.tiles[key].near], seRank: rankOf('se'), mtRank: rankOf('mt') }
  }, [key])

  if (!j) {
    return (
      <section className="wrap notfound">
        <h1 className="h2" style={{ marginBottom: 16 }}>No judge called <code>{key}</code></h1>
        <a className="btn btn-primary" href="#/leaderboard">Back to the leaderboard <Icon.arrow /></a>
      </section>
    )
  }
  const { near } = ctx
  const st = statement(key, near.key)
  const tile = core.tiles[key]
  const accs = judges.map((x) => x.best)
  const lo = Math.min(...accs)
  const hi = Math.max(...accs)
  const budgetsHere = facts.budgets.filter((b) => (b === QUICK ? j.acc10 != null : true))
  const eq = claims.tostGroup.includes(key)
  const pd = res?.perDataset[key]
  const npd = res?.perDataset[near.key]?.[String(near.bestBudget)]
  const sweep = res?.sweep.judges[key]
  const pers = res?.personas[key]
  const prompts = res?.prompts.judges[key]
  const proto = res?.protocols.accuracy[key]
  const flip = res?.protocols.flip[key]
  const agree = sample ? Object.fromEntries(budgetsHere.map((b) => [b, sample.items.filter((r, k) => r[String(b)]?.verdict === core.items[k].oracle).length])) : null

  const budgetMarks = (rowKey, getV, idp, tipName) => budgetsHere.map((b) => {
    const v = getV(String(b))
    return v == null ? null : {
      id: `${idp}:${b}`, key, x: v, y: rowKey, fam: j.family, open: b === REASONED, r: 6,
      tip: () => <><TipHead fam={j.family} name={tipName} open={b === REASONED} /><TipRow k={`B = ${count(b)}`} v={`${fmt(v)}%`} /></>,
    }
  }).filter(Boolean)
  const axisFor = (vals, pad = 1.5) => {
    const a = Math.floor(Math.min(...vals) - pad)
    const b = Math.min(100, Math.ceil(Math.max(...vals) + pad))
    const step = b - a > 30 ? 10 : b - a > 12 ? 5 : b - a > 5 ? 2 : 1
    const ticks = []
    for (let v = Math.ceil(a / step) * step; v <= b; v += step) ticks.push(v)
    return { domain: [a, b], ticks, format: (v) => fmt(v, 0), title: 'Accuracy (%)' }
  }
  const byBudget = (obj, rows, label) => ({
    columns: [{ key: 'row', label, align: 'left' }, ...Object.keys(obj).map((b) => ({ key: `b${b}`, label: `B = ${count(Number(b))} (%)` }))],
    rows: rows.map((r) => ({ row: r.label, ...Object.fromEntries(Object.keys(obj).map((b) => [`b${b}`, obj[b]?.[r.key]])) })),
  })

  let dsSpec = null
  if (pd) {
    const rows = res.datasetKeys.map((d, n) => ({ key: d, label: DS[d].name, band: n % 2 === 0 }))
    const marks = [
      ...(npd ? res.datasetKeys.map((d, di) => ({
        id: `n:${d}`, x: npd[di], y: d, fam: near.family, open: true, r: 5, color: 'var(--faint)',
        tip: () => <><TipHead fam={near.family} name={near.name} /><TipRow k={DS[d].name} v={`${fmt(npd[di])}%`} /></>,
      })) : []),
      ...res.datasetKeys.flatMap((d, di) => budgetMarks(d, (b) => pd[b]?.[di], d, DS[d].name)),
    ]
    dsSpec = {
      key: 'ds', alt: `Accuracy of ${j.name} on each dataset`, x: axisFor(marks.map((m) => m.x)), y: { rows }, labelW: 116, labelWNarrow: 96, marks,
      table: {
        columns: [{ key: 'row', label: 'Dataset', align: 'left' }, ...budgetsHere.map((b) => ({ key: `b${b}`, label: `B = ${count(b)} (%)` })), { key: 'near', label: `${near.name} (%)` }],
        rows: res.datasetKeys.map((d, di) => ({ row: DS[d].name, ...Object.fromEntries(budgetsHere.map((b) => [`b${b}`, pd[String(b)]?.[di]])), near: npd?.[di] })),
      },
    }
  }
  let persSpec = null
  if (pers) {
    const rows = [{ key: 'base', label: 'No persona', band: true }, ...facts.personas.map((p) => ({ key: p, label: p }))]
    const marks = [...budgetMarks('base', (b) => pers[b]?.base, 'base', 'No persona'), ...facts.personas.flatMap((p) => budgetMarks(p, (b) => pers[b]?.[p], p, p))]
    persSpec = { key: 'pers', alt: `Accuracy of ${j.name} under each persona`, x: axisFor(marks.map((m) => m.x), 1), y: { rows }, labelW: 108, labelWNarrow: 92, marks, table: byBudget(pers, rows, 'Persona') }
  }
  let promptSpec = null
  if (prompts) {
    const rows = res.prompts.variants.map((v) => ({ key: v, label: res.prompts.labels[v] }))
    const marks = res.prompts.variants.flatMap((v) => budgetMarks(v, (b) => prompts[b]?.[v], `pv${v}`, res.prompts.labels[v]))
    promptSpec = { key: 'pv', alt: `Accuracy of ${j.name} under each wording`, x: axisFor(marks.map((m) => m.x), 1), y: { rows }, labelW: 128, labelWNarrow: 100, marks, table: byBudget(prompts, rows, 'Wording') }
  }
  let protoSpec = null
  if (proto) {
    const ps = Object.keys(proto)
    const rows = ps.map((p) => ({ key: p, label: PROTO_LABEL[p] ?? p }))
    const marks = ps.flatMap((p) => budgetMarks(p, (b) => proto[p]?.[b], `pr${p}`, PROTO_LABEL[p]))
    protoSpec = {
      key: 'pr', alt: `Accuracy of ${j.name} with and without the reference`, x: axisFor(marks.map((m) => m.x), 2), y: { rows }, labelW: 150, labelWNarrow: 112, marks,
      table: { columns: [{ key: 'row', label: 'Protocol', align: 'left' }, ...facts.budgets.map((b) => ({ key: `b${b}`, label: `B = ${count(b)} (%)` }))], rows: ps.map((p) => ({ row: PROTO_LABEL[p] ?? p, ...Object.fromEntries(facts.budgets.map((b) => [`b${b}`, proto[p][String(b)]])) })) },
    }
  }
  const budgetKey = budgetsHere.length > 1
    ? <span className="fam-key"><span><MarkIcon fam={j.family} size={11} />B = {QUICK}</span><span><MarkIcon fam={j.family} size={11} open />B = {R}</span></span>
    : <span className="fam-key"><span><MarkIcon fam={j.family} size={11} open />B = {R}</span></span>
  const sweepAll = res ? Object.keys(res.sweep.judges) : []

  return (
    <div className="judge-page">
      <section className="judge-top">
        <div className="wrap">
          <nav className="judge-crumbs" aria-label="Judges">
            <a href="#/leaderboard" className="crumb"><Icon.arrowLeft /> Leaderboard</a>
            <span className="spacer" />
            <a className="icon-btn" href={`#/judge/${prev.key}`} aria-label={`Previous judge: ${prev.name}`} data-tip={`${prev.name} ←`}><Icon.arrowLeft /></a>
            <a className="icon-btn" href={`#/judge/${next.key}`} aria-label={`Next judge: ${next.name}`} data-tip={`${next.name} →`}><Icon.arrow /></a>
          </nav>
          <div className="judge-head">
            <div className="judge-mark" data-vt="mark"><svg viewBox="-12 -12 24 24" aria-hidden="true"><Mark fam={j.family} r={8} ring={j.kind === 'ladder'} /></svg></div>
            <div className="judge-id">
              <h1 className="judge-title" data-vt="name">{j.name}</h1>
              <p className="judge-meta">
                {j.kind === 'ladder' ? 'Qwen3 ladder judge' : j.kind === 'large' ? 'Large' : `Small · ${famLabel(j.family)}`}
                {' · '}{j.size}{j.active ? ` (${j.active}B active)` : ''}
                {' · '}{THINKS[j.thinks]}
                {' · '}{j.rank ? `rank ${j.rank} of ${nRanked}` : 'not ranked'}
                {' · '}<a href={`https://huggingface.co/${j.weights}`} target="_blank" rel="noreferrer">{j.weights}<Icon.external className="ext" /></a>
              </p>
            </div>
          </div>
          <a className="btn btn-sm judge-compare" href={`#/leaderboard?compare=${key},${near.key}`}><Icon.swap /> Compare with {near.name}</a>
        </div>
      </section>

      <section className="wrap judge-body">
        <div className="jt-grid">
          <Tile label="Accuracy" value={<>{fmt(j.best)}<small>%</small></>} at={`at B = ${count(j.bestBudget)}`}>
            <Strip value={j.best} values={accs} other={near.best} fam={j.family} lo={lo} hi={hi}
              band={[byKey['phi4-14b'].best - parseFloat(claims.margin), byKey['phi4-14b'].best + parseFloat(claims.margin)]} />
            <span>{budgetsHere.length > 1 ? <>B = {QUICK}: {fmt(j.acc10)}% · B = {R}: {fmt(j.acc8192)}%</> : <>Runs at B = {R} only</>}</span>
            <span className="jt-key"><i aria-hidden="true" />±{claims.margin} points of Phi-4</span>
          </Tile>
          <Tile label={j.kind === 'large' ? 'Nearest small judge' : 'Nearest large judge'} value={<a href={`#/judge/${near.key}`} className="jt-link"><MarkIcon fam={near.family} size={15} /> {near.name}</a>}>
            <span>{fmt(near.best)}% at B = {count(near.bestBudget)}.{st && <> <strong className="jt-claim">{st.text}</strong></>}</span>
          </Tile>
          <Tile label="Cost per 1,000 verdicts (estimated)" value={<>${fmt(tile.cost)}</>} at={`at B = ${count(tile.budget)}`}>
            {tile.otherBudget && <span>${fmt(tile.otherCost)} at B = {count(tile.otherBudget)}</span>}
            <span>{tile.nearCost != null
              ? <>{near.name}: ${fmt(tile.nearCost)} at B = {count(tile.budget)}</>
              : <>{near.name} runs at B = {count(tile.nearOnly)} only</>}</span>
          </Tile>
          <Tile label="SummEval agreement" value={<>ρ = {fmt(j.se, 3)}</>}>
            <span>{ctx.seRank
              ? <>{cap(ordinal(ctx.seRank))} of {nRanked} judges. On MT-Bench, ρ = {fmt(j.mt, 3)} ({ordinal(ctx.mtRank)}).</>
              : <>Ladder judge, not ranked. On MT-Bench, ρ = {fmt(j.mt, 3)}.</>}</span>
          </Tile>
        </div>
        <div className="jt-facts">
          {j.delta != null && <span>Quick minus reasoned <strong className="tnum">{signed(j.delta)} points</strong></span>}
          <span>Parseable verdicts <strong className="tnum">{budgetsHere.map((b) => `${fmt(b === QUICK ? j.ifr10 : j.ifr8192)}%`).join(' · ')}</strong></span>
          {flip && <span>Answer-order flip rate <strong className="tnum">{Object.values(flip).map((v) => fmt(v, 2)).join(' · ')}</strong></span>}
        </div>
        {eq && (
          <p className="jt-eq">
            <span className="eq-dot" aria-hidden="true" />
            <span>
              One of {claims.tostGroup.length} leading judges that are pairwise equivalent within ±{claims.margin} points:{' '}
              {claims.tostGroup.map((k, n) => <span key={k}>{n ? ', ' : ''}{k === key ? <strong>{byKey[k].name}</strong> : <a href={`#/judge/${k}`}>{byKey[k].name}</a>}</span>)}.
            </span>
          </p>
        )}

        {!res && <div className="skeleton" style={{ height: 380 }} />}
        {res && (
          <div className="judge-grid" data-reveal="stagger">
            {dsSpec && (
              <ChartFrame id={`judge-${key}-datasets`} title="Accuracy per dataset" sub={`Grey: ${near.name}, the nearest ${j.kind === 'large' ? 'small' : 'large'} judge.`} legend={budgetKey} table={dsSpec.table}>
                <Lazy height={340}><Stage spec={dsSpec} height={340} /></Lazy>
              </ChartFrame>
            )}
            {sweep && (
              <ChartFrame id={`judge-${key}-sweep`} title="Budget sweep" sub="Reasoned prompt. The other judges in grey."
                table={{ columns: [{ key: 'b', label: 'Budget', align: 'left' }, { key: 'acc', label: 'Accuracy (%)' }], rows: res.sweep.budgets.map((b, n) => ({ b: count(b), acc: sweep[n] })) }}>
                <Lazy height={340}><LineChart series={sweepAll.map((k) => ({ key: k, label: byKey[k].name, fam: byKey[k].family, slug: FAMILY_SLUG[byKey[k].family], values: res.sweep.judges[k] }))}
                  xs={res.sweep.budgets} active={[key]}
                  yDomain={[Math.floor(Math.min(...sweep)) - 3, Math.min(100, Math.ceil(Math.max(...sweep)) + 2)]}
                  xFormat={(v) => count(v)} height={340} xTitle="Token budget (log scale)" yTitle="Accuracy (%)" ariaLabel={`Budget sweep for ${j.name}`} /></Lazy>
              </ChartFrame>
            )}
            {persSpec && (
              <ChartFrame id={`judge-${key}-personas`} title="Adversarial personas" sub="Accuracy under each system prompt" legend={budgetKey} table={persSpec.table}>
                <Lazy height={300}><Stage spec={persSpec} height={300} /></Lazy>
              </ChartFrame>
            )}
            {promptSpec && (
              <ChartFrame id={`judge-${key}-prompts`} title="Prompt wordings" sub="Accuracy under each wording of the instruction" legend={budgetKey} table={promptSpec.table}>
                <Lazy height={260}><Stage spec={promptSpec} height={260} /></Lazy>
              </ChartFrame>
            )}
            {protoSpec && (
              <ChartFrame id={`judge-${key}-protocols`} title="Judging protocols" sub="With and without the reference answer" legend={budgetKey} table={protoSpec.table}>
                <Lazy height={210}><Stage spec={protoSpec} height={210} /></Lazy>
              </ChartFrame>
            )}
          </div>
        )}

        <section className="verdicts-sec" aria-labelledby="vs-h">
          <div className="vs-head" data-reveal>
            <h2 className="h3" id="vs-h">Real verdicts</h2>
            <p className="muted small">
              The same {core.items.length} items for every judge.
              {agree && <> {j.name} agrees with the oracle on {budgetsHere.map((b) => `${agree[b]} of ${core.items.length} at B = ${count(b)}`).join(' and ')}.</>}
            </p>
          </div>
          {!sample && <div className="skeleton" style={{ height: 260 }} />}
          {sample && (
            <div className="vc-grid" data-reveal="stagger">
              {core.items.map((it, k) => <VerdictCard key={`${key}-${k}`} item={it} row={sample.items[k]} />)}
            </div>
          )}
        </section>

        <nav className="judge-next" aria-label="Other judges" data-reveal="stagger">
          <a href={`#/judge/${prev.key}`} className="jn prev">
            <span className="jn-l"><Icon.arrowLeft /> Previous</span>
            <span className="jn-n"><MarkIcon fam={prev.family} size={13} ring={prev.kind === 'ladder'} /> {prev.name}</span>
            <span className="jn-s tnum">{fmt(prev.best)}%</span>
          </a>
          <a href={`#/judge/${next.key}`} className="jn next">
            <span className="jn-l">Next <Icon.arrow /></span>
            <span className="jn-n"><MarkIcon fam={next.family} size={13} ring={next.kind === 'ladder'} /> {next.name}</span>
            <span className="jn-s tnum">{fmt(next.best)}%</span>
          </a>
        </nav>
      </section>
    </div>
  )
}
