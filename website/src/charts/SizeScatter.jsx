import { useId, useMemo, useState } from 'react'
import { Mark } from '../components/ui'
import { judges, byKey, claims, famLabel } from '../lib/data'
import { fmt } from '../lib/format'
import { useInView, useTween } from '../lib/motion'
import { linear, log, ticks, useWidth, YAxis, XAxis, Tooltip, TipHead, TipRow, useTip } from './kit'

const valueAt = (j, mode) => (mode === 'best' ? j.best : mode === 10 ? j.acc10 : j.acc8192)
const budgetAt = (j, mode) => (mode === 'best' ? j.bestBudget : mode)
const SIZE_TICKS = [0.5, 1, 2, 5, 10, 20, 50, 100]
const FAM_ORDER = ['Llama', 'Qwen2.5', 'Qwen3', 'Phi-4', 'Large']

// Judges of the same size would sit on top of each other; spread them a few
// pixels apart, in family order, as the paper's figures do.
export const DODGE = (() => {
  const groups = {}
  judges.forEach((j) => { (groups[j.params] ||= []).push(j) })
  const out = {}
  Object.values(groups).forEach((g) => {
    g.sort((a, b) => FAM_ORDER.indexOf(a.family) - FAM_ORDER.indexOf(b.family) || a.name.localeCompare(b.name))
    g.forEach((j, i) => { out[j.key] = (i - (g.length - 1) / 2) * 9 })
  })
  return out
})()

// Accuracy against judge size (log scale), one mark per judge, with the
// equivalence band around Phi-4 at the quick verdict.
export default function SizeScatter({ mode = 'best', zoom = 'all', height = 380, highlight = ['phi4-14b', claims.topLarge], compact = false }) {
  const [box, width] = useWidth()
  const [inRef, seen] = useInView()
  const [tip, showTip, hideTip] = useTip(width)
  const [hover, setHover] = useState(null)
  const clip = `plot${useId().replace(/[^a-zA-Z0-9]/g, '')}`
  const m ={ t: 30, r: compact ? 14 : 22, b: 46, l: 40 }
  const W = width
  const H = height
  const margin = parseFloat(claims.margin)
  const phi = byKey['phi4-14b']

  const vals = judges.map((j) => valueAt(j, mode))
  const present = vals.filter((v) => v != null)
  const top = Math.max(...present)
  const domAll = [Math.floor(Math.min(...present) / 5) * 5 - 1, Math.ceil(top + 0.6)]
  const domTop = [Math.floor(phi.acc10 - 3), Math.ceil(top + 0.6)]
  const dom = zoom === 'top' ? domTop : domAll

  // fall back to the judge's best accuracy for position when it has no value at this budget
  const target = [dom[0], dom[1], ...judges.map((j, i) => vals[i] ?? j.best)]
  const from = [dom[0], dom[1], ...judges.map(() => dom[0])]
  const tw = useTween(target, { from, run: seen, duration: 900 })
  const [y0, y1] = [tw[0], tw[1]]
  const ys = tw.slice(2)

  const x = log([0.45, 170], [m.l, W - m.r])
  const y = linear([y0, y1], [H - m.b, m.t])
  const yt = ticks(y0, y1, zoom === 'top' ? 5 : 6).filter((t) => t >= y0 - 1e-6 && t <= y1 + 1e-6)

  const pts = judges.map((j, i) => {
    const v = vals[i]
    const vy = ys[i]
    const visible = v != null && v >= dom[0] && v <= dom[1]
    return { j, v, px: x(j.params) + (DODGE[j.key] || 0), py: Math.max(m.t - 4, Math.min(H - m.b, y(vy))), visible }
  })

  const ladder = useMemo(() => judges.filter((j) => j.family === 'Qwen3').sort((a, b) => a.params - b.params), [])
  const ladderPath = ladder
    .map((j) => pts.find((p) => p.j.key === j.key))
    .filter((p) => p.v != null)
    .map((p, i) => `${i ? 'L' : 'M'}${p.px.toFixed(1)},${p.py.toFixed(1)}`)
    .join('')

  const showBand = mode !== 8192
  const bandLo = y(phi.acc10 - margin)
  const bandHi = y(phi.acc10 + margin)

  const onMove = (e) => {
    const r = e.currentTarget.getBoundingClientRect()
    const mx = ((e.clientX - r.left) / r.width) * W
    const my = ((e.clientY - r.top) / r.height) * H
    let best = null
    let bd = 26 * 26
    for (const p of pts) {
      if (!p.visible) continue
      const d = (p.px - mx) ** 2 + (p.py - my) ** 2
      if (d < bd) { bd = d; best = p }
    }
    if (!best) { setHover(null); hideTip(); return }
    setHover(best.j.key)
    const j = best.j
    showTip((best.px / W) * r.width, (best.py / H) * r.height, (
      <>
        <TipHead fam={j.family} name={j.name} open={budgetAt(j, mode) === 8192} ring={j.kind === 'ladder'} />
        <TipRow k="Size" v={`${j.size}${j.active ? ` (${j.active}B active)` : ''}`} />
        <TipRow k="Family" v={famLabel(j.family)} />
        <TipRow k="Accuracy, B = 10" v={j.acc10 == null ? 'reasons only' : `${fmt(j.acc10)}%`} />
        <TipRow k="Accuracy, B = 8,192" v={`${fmt(j.acc8192)}%`} />
      </>
    ))
  }
  const onLeave = () => { setHover(null); hideTip() }

  const labelled = new Set([...highlight, hover].filter(Boolean))

  return (
    <div className="chart" ref={(el) => { box.current = el; inRef.current = el }}>
      <svg
        viewBox={`0 0 ${W} ${H}`}
        role="img"
        aria-label={`Accuracy against judge size for all ${judges.length} judges. ${phi.name} at ${fmt(phi.acc10)}% sits within ±${claims.margin} points of ${claims.topLargeName} at ${claims.topLargeAcc10}%.`}
        onPointerMove={onMove}
        onPointerLeave={onLeave}
        onPointerDown={onMove}
      >
        <defs>
          <clipPath id={clip}>
            <rect x={m.l - 8} y={m.t - 10} width={W - m.l - m.r + 16} height={H - m.t - m.b + 10} />
          </clipPath>
        </defs>
        <YAxis scale={y} ticks={yt} x0={m.l} x1={W - m.r} format={(t) => fmt(t, 0)} />
        <text className="ax-title" x={m.l - 8} y={m.t - 16}>Accuracy (%)</text>
        <XAxis scale={x} ticks={SIZE_TICKS.filter((t) => W > 520 || [1, 10, 100].includes(t))} y={H - m.b} x0={m.l} x1={W - m.r} format={(t) => `${t}B`} title="Parameters (log scale)" />

        <g clipPath={`url(#${clip})`}>
          {showBand && (
            <g style={{ opacity: seen ? 1 : 0, transition: 'opacity .8s .5s' }}>
              <rect className="band-eq" x={m.l} width={W - m.l - m.r} y={bandHi} height={Math.max(0, bandLo - bandHi)} />
              <line className="band-eq-line" x1={m.l} x2={W - m.r} y1={bandHi} y2={bandHi} />
              <line className="band-eq-line" x1={m.l} x2={W - m.r} y1={bandLo} y2={bandLo} />
            </g>
          )}
          {mode === 'best' && ladderPath && (
            <path d={ladderPath} fill="none" style={{ stroke: 'var(--fam-qwen3)', opacity: seen ? 0.45 : 0, transition: 'opacity .6s .6s' }} strokeWidth="1.3" strokeDasharray="3 3" />
          )}
          {pts.map((p) => (
            <g key={p.j.key} transform={`translate(${p.px},${p.py})`}>
              <Mark
                fam={p.j.family}
                r={compact ? 5.5 : 6.5}
                open={budgetAt(p.j, mode) === 8192}
                ring={p.j.kind === 'ladder'}
                className={`mark${hover && hover !== p.j.key ? ' dim' : ''}`}
                style={{ opacity: p.visible ? undefined : 0 }}
              />
            </g>
          ))}
        </g>

        {showBand && zoom === 'top' && W > 380 && (
          <text className="lbl muted" x={W - m.r - 4} y={bandHi - 6} textAnchor="end" style={{ fontSize: 11 }}>
            ±{claims.margin} points of {phi.name}
          </text>
        )}

        {pts.filter((p) => p.visible && labelled.has(p.j.key)).map((p) => {
          const nearRight = p.px > W * 0.72
          const below = p.j.key === 'phi4-14b'
          const x0 = nearRight ? p.px + 8 : p.px + 12
          const y0 = below ? p.py + 20 : p.py - 14
          return (
            <g key={p.j.key} style={{ pointerEvents: 'none' }}>
              <circle cx={p.px} cy={p.py} r={11} fill="none" style={{ stroke: 'var(--ink)', opacity: 0.35 }} strokeWidth="1" />
              <text className="lbl" x={x0} y={y0} textAnchor={nearRight ? 'end' : 'start'}>{p.j.name}</text>
            </g>
          )
        })}
      </svg>
      <Tooltip tip={tip} />
    </div>
  )
}
