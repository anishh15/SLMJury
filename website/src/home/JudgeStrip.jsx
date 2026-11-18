import { useMemo, useState } from 'react'
import { Mark } from '../components/ui'
import { useWidth } from '../charts/kit'
import { byKey, claims, points, QUICK, REASONED, setHovered, useHovered } from '../lib/data'
import { count, fmt } from '../lib/format'

// Every judge on one accuracy axis: its quick verdict, or B = 8,192 for a
// judge that always reasons (open mark). The gold band is ±margin around Phi-4.
// The axis has two linear segments with a marked break, so the crowded top
// gets most of the width: 40–84% on the left, 84–91% on the right.
const LO = 40
const MID = 84
const HI = 91
const SPLIT = 0.3
const GAP = 0.035
const TICKS_LO = [40, 60, 80]
const TICKS_HI = [84, 85, 86, 87, 88, 89, 90, 91]
const rows = points.filter((p) => byKey[p.key].kind !== 'ladder').map((p) => ({ ...p, j: byKey[p.key] }))

// Stack marks upward wherever two would overlap, so every judge stays visible.
function swarm(items, pitch) {
  const placed = []
  return [...items].sort((a, b) => a.x - b.x).map((it) => {
    let level = 0
    while (placed.some((p) => p.level === level && Math.abs(p.x - it.x) < pitch)) level += 1
    const out = { ...it, level }
    placed.push(out)
    return out
  })
}

export default function JudgeStrip({ pair }) {
  const [box, width] = useWidth(880)
  const [focus, setFocus] = useState(null)
  const linked = useHovered()
  const W = Math.max(300, width)
  const narrow = W < 560
  const pitch = narrow ? 12 : 15
  const r = narrow ? 4.4 : 5.6
  const pad = { l: narrow ? 14 : 18, r: narrow ? 12 : 16 }
  const span = W - pad.l - pad.r
  const xBreak = pad.l + span * SPLIT
  const xHi0 = pad.l + span * (SPLIT + GAP)
  const x = (v) => (v < MID
    ? pad.l + ((v - LO) / (MID - LO)) * (xBreak - pad.l)
    : xHi0 + ((v - MID) / (HI - MID)) * (W - pad.r - xHi0))
  const marks = useMemo(() => swarm(rows.map((p) => ({ ...p, x: x(p.accExact) })), pitch), [W])
  const top = Math.max(...marks.map((m) => m.level))
  const LABEL_H = 26
  const axisY = LABEL_H + 14 + (top + 1) * pitch
  const H = axisY + 24
  const yOf = (m) => axisY - pitch * 0.8 - m.level * pitch
  const phi = rows.find((p) => p.key === 'phi4-14b')
  const margin = parseFloat(claims.margin)
  const bx0 = x(phi.accExact - margin)
  const bx1 = x(phi.accExact + margin)
  const at = Object.fromEntries(marks.map((m) => [m.key, m]))
  const shown = focus || linked
  const hot = shown && at[shown] ? at[shown] : null

  // the two compared judges get names; the right one ends at the strip's edge
  const [a, b] = [...pair].filter((k) => at[k]).sort((p, q) => at[p].x - at[q].x)
  const charW = 7.4
  const callouts = []
  if (a && b) {
    const wB = byKey[b].name.length * charW
    const endB = Math.min(W - pad.r, at[b].x + wB / 2)
    const wA = byKey[a].name.length * charW
    let endA = Math.min(at[a].x + wA / 2, endB - wB - 16)
    callouts.push(endA - wA < 2 ? { k: a, tx: 2, anchor: 'start' } : { k: a, tx: endA, anchor: 'end' })
    callouts.push({ k: b, tx: endB, anchor: 'end' })
  }
  const leave = () => { setFocus(null); setHovered(null) }

  return (
    <div className="jstrip" ref={box}>
      <svg viewBox={`0 0 ${W} ${H}`} role="group" aria-label={`All ${rows.length} judges on one accuracy axis`} onPointerLeave={leave}>
        <rect className="js-band" x={bx0} y={LABEL_H + 4} width={bx1 - bx0} height={axisY - LABEL_H - 4} />
        <line className="js-band-edge" x1={bx0} x2={bx0} y1={LABEL_H + 4} y2={axisY} />
        <line className="js-band-edge" x1={bx1} x2={bx1} y1={LABEL_H + 4} y2={axisY} />
        <line className="js-axis js-axis-lo" x1={pad.l} x2={xBreak} y1={axisY} y2={axisY} />
        <line className="js-axis" x1={xHi0} x2={W - pad.r} y1={axisY} y2={axisY} />
        <path className="js-break" d={`M${(xBreak + xHi0) / 2 - 5},${axisY + 5}l4,-10M${(xBreak + xHi0) / 2 + 1},${axisY + 5}l4,-10`} />
        {[...TICKS_LO, ...TICKS_HI.filter((t) => !narrow || t % 2 === 0)].map((t) => (
          <g key={t} transform={`translate(${x(t)},${axisY})`}>
            <line className="js-tick" y1={0} y2={4} />
            <text className="js-tick-l" y={17} textAnchor="middle">{t === 84 && narrow ? '' : `${t}%`}</text>
          </g>
        ))}
        {callouts.map((c) => {
          const m = at[c.k]
          const ly = LABEL_H - 9
          return (
            <g key={`c-${c.k}`} className="js-callout" aria-hidden="true">
              <path d={`M${m.x},${yOf(m) - r - 4}V${ly + 5}H${c.anchor === 'end' ? Math.max(c.tx, m.x) : Math.min(c.tx, m.x)}`} className="js-lead" />
              <text x={c.tx} y={ly} textAnchor={c.anchor} className="js-name">{byKey[c.k].name}</text>
            </g>
          )
        })}
        {marks.map((m) => {
          const on = hot && hot.key === m.key
          const picked = pair.includes(m.key)
          return (
            <a key={m.key} href={`#/judge/${m.key}`} className={`js-mark${hot && !on ? ' dim' : ''}`}
              aria-label={`${m.j.name}, ${fmt(m.acc)}% at B = ${count(m.budget)}`}
              onPointerEnter={() => { setFocus(m.key); setHovered(m.key) }} onFocus={() => setFocus(m.key)} onBlur={() => setFocus(null)}>
              <g transform={`translate(${m.x},${yOf(m)})`}>
                <circle r={pitch * 0.62} className="js-hit" />
                {(picked || on) && <circle r={r + 4} className={`js-ring${on ? ' on' : ''}`} />}
                <Mark fam={m.j.family} r={r} open={m.budget === REASONED && m.budget !== QUICK} />
              </g>
            </a>
          )
        })}
      </svg>
      {hot && (
        <div className="js-tip" role="status"
          style={{ transform: `translate(${Math.min(W - 110, Math.max(110, hot.x))}px, ${yOf(hot) - r - 8}px) translate(-50%, -100%)` }}>
          <strong>{hot.j.name}</strong> <span className="tnum">{fmt(hot.acc)}%</span>
          <span className="js-tip-sub"> · {hot.j.size}{hot.budget !== QUICK ? ` · B = ${count(hot.budget)}` : ''}</span>
        </div>
      )}
      <div className="js-key">
        <span><span className="js-key-band" aria-hidden="true" />±{claims.margin} points of Phi-4</span>
        <span className="faint">Open marks: B = {count(REASONED)}, judges that always reason. The axis changes scale at 84%.</span>
      </div>
    </div>
  )
}
