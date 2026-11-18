import { useMemo, useRef, useState } from 'react'
import { Icon, Mark, Segmented } from '../components/ui'
import { claims, judges, QUICK, REASONED, setHovered, useHovered } from '../lib/data'
import { count, fmt } from '../lib/format'
import { useFlip } from '../lib/flip'
import { useInView } from '../lib/motion'

const N = 10
const accAt = (j, b) => (b === 'best' ? j.best : b === QUICK ? j.acc10 : j.acc8192)
const budgetAt = (j, b) => (b === 'best' ? j.bestBudget : b)
const costAt = (j, b) => (budgetAt(j, b) === QUICK ? j.cost10 : j.cost8192)

// The top of the table: each judge a mark on one labelled accuracy axis, with
// its estimated cost at the same budget as its accuracy.
export default function LeaderTeaser() {
  const [b, setB] = useState('best')
  const list = useRef(null)
  const [ref, seen] = useInView({ threshold: 0.2 })
  const hovered = useHovered()
  const rows = useMemo(() => judges.filter((j) => j.kind !== 'ladder' && accAt(j, b) != null)
    .sort((x, y) => accAt(y, b) - accAt(x, b)).slice(0, N), [b])
  useFlip(list, [b])
  const vals = rows.map((j) => accAt(j, b))
  const span = Math.max(...vals) - Math.min(...vals)
  const step = span > 3 ? 1 : 0.5
  const lo = Math.floor((Math.min(...vals) - step * 0.4) / step) * step
  const hi = Math.ceil((Math.max(...vals) + step * 0.4) / step) * step
  const ticks = []
  for (let t = lo; t <= hi + 1e-9; t += step) ticks.push(+t.toFixed(2))
  const pos = (v) => (v - lo) / (hi - lo)
  const group = new Set(claims.tostGroup)
  const inGroup = rows.map((j, i) => (group.has(j.key) && b === 'best' ? i : -1)).filter((i) => i >= 0)
  const contiguous = inGroup.length === claims.tostGroup.length && inGroup[inGroup.length - 1] - inGroup[0] === inGroup.length - 1
  return (
    <section className="section teaser" aria-labelledby="lt-title" ref={ref}>
      <div className="wrap-wide">
        <div className="sec-head row-head" data-reveal="stagger">
          <h2 className="h2" id="lt-title">The top of the table</h2>
          <Segmented label="Budget" value={b} onChange={setB}
            options={[{ value: 'best', label: 'Better budget' }, { value: QUICK, label: `B = ${QUICK}` }, { value: REASONED, label: `B = ${count(REASONED)}` }]} />
        </div>
        <div className={`lt${seen ? ' in' : ''}`} data-reveal>
          <div className="lt-head" aria-hidden="true">
            <span />
            <span />
            <span className="lt-axis">{ticks.map((t) => <i key={t} style={{ left: `${pos(t) * 100}%` }}>{step < 1 ? fmt(t, 1) : fmt(t, 0)}</i>)}</span>
            <span className="lt-h-cost">$ / 1k</span>
            <span className="lt-h-acc">Accuracy</span>
          </div>
          <ol className="lt-list" ref={list}>
            {rows.map((j, i) => {
              const v = accAt(j, b)
              const bb = budgetAt(j, b)
              return (
                <li key={j.key} data-flip={j.key} className={`lt-row${hovered === j.key ? ' hot' : ''}`}
                  onPointerEnter={() => setHovered(j.key)} onPointerLeave={() => setHovered(null)}>
                  <a href={`#/judge/${j.key}`}>
                    <span className="lt-rank tnum">{i + 1}</span>
                    <span className="lt-name">{j.name}<span className="lt-size">{j.size}</span></span>
                    <span className="lt-track" aria-hidden="true">
                      {ticks.map((t) => <i key={t} style={{ left: `${pos(t) * 100}%` }} />)}
                      <svg className="lt-dot" viewBox="-8 -8 16 16" style={{ left: `${pos(v) * 100}%`, '--i': i }}><Mark fam={j.family} r={5.2} open={bb !== QUICK} /></svg>
                    </span>
                    <span className="lt-cost tnum" title={`Estimated serving cost per 1,000 verdicts at B = ${count(bb)}`}>${fmt(costAt(j, b))}</span>
                    <span className="lt-acc tnum">{fmt(v)}<span>%</span></span>
                  </a>
                </li>
              )
            })}
          </ol>
          {contiguous && (
            <div className="lt-bracket" style={{ '--from': inGroup[0], '--n': inGroup.length }} aria-hidden="true">
              <span>pairwise equivalent within ±{claims.margin} points</span>
            </div>
          )}
        </div>
        <div className="teaser-foot" data-reveal>
          <a className="btn" href="#/leaderboard">Full leaderboard <Icon.arrow className="arrow" /></a>
          <a className="btn btn-ghost" href={`#/leaderboard?compare=${claims.tostGroup.slice(0, 2).join(',')}`}>Compare two judges</a>
          <span className="faint small">Open marks: B = {count(REASONED)}. Cost is an estimate per 1,000 verdicts, at the same budget as the accuracy.</span>
        </div>
      </div>
    </section>
  )
}
