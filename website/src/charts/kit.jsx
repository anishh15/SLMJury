import { useEffect, useLayoutEffect, useRef, useState } from 'react'
import { MarkIcon } from '../components/ui'
import { FAMILIES } from '../lib/data'

// ── Scales ──────────────────────────────────────────────────────────────────
export function linear([d0, d1], [r0, r1]) {
  const k = d1 === d0 ? 0 : (r1 - r0) / (d1 - d0)
  const f = (v) => r0 + (v - d0) * k
  f.invert = (p) => d0 + (p - r0) / (k || 1)
  f.domain = [d0, d1]
  return f
}

export function log([d0, d1], [r0, r1]) {
  const l0 = Math.log(d0)
  const l1 = Math.log(d1)
  const f = (v) => r0 + ((Math.log(v) - l0) / (l1 - l0)) * (r1 - r0)
  f.domain = [d0, d1]
  return f
}

export function band(keys, [r0, r1], pad = 0.2) {
  const n = keys.length
  const step = (r1 - r0) / Math.max(1, n)
  const bw = step * (1 - pad)
  const f = (k) => r0 + keys.indexOf(k) * step + (step - bw) / 2
  f.bw = bw
  f.step = step
  return f
}

export function ticks(d0, d1, count = 5) {
  const span = d1 - d0
  if (!(span > 0)) return [d0]
  const raw = span / count
  const mag = Math.pow(10, Math.floor(Math.log10(raw)))
  const norm = raw / mag
  const step = (norm >= 5 ? 10 : norm >= 2 ? 5 : norm >= 1 ? 2 : 1) * mag
  const out = []
  for (let v = Math.ceil(d0 / step - 1e-9) * step; v <= d1 + step * 1e-9; v += step) out.push(+v.toFixed(10))
  return out
}

export const decimalsFor = (ts) => {
  const step = ts.length > 1 ? Math.abs(ts[1] - ts[0]) : 1
  return step >= 1 ? 0 : step >= 0.1 ? 1 : step >= 0.01 ? 2 : 3
}

// ── Size ────────────────────────────────────────────────────────────────────
// Width of the chart container; charts draw at that width so text stays crisp.
export function useWidth(initial = 640) {
  const ref = useRef(null)
  const [w, setW] = useState(initial)
  useLayoutEffect(() => {
    if (ref.current) setW(ref.current.clientWidth || initial)
  }, [])
  useEffect(() => {
    const el = ref.current
    if (!el || !('ResizeObserver' in window)) return
    let raf = 0
    const ro = new ResizeObserver(([e]) => {
      cancelAnimationFrame(raf)
      const next = Math.round(e.contentRect.width)
      if (next > 0) raf = requestAnimationFrame(() => setW(next))
    })
    ro.observe(el)
    return () => { ro.disconnect(); cancelAnimationFrame(raf) }
  }, [])
  return [ref, w]
}

// ── Axes ────────────────────────────────────────────────────────────────────
export function YAxis({ scale, ticks: ts, x0, x1, format = (v) => v, title, titleX }) {
  return (
    <g>
      {ts.map((t) => (
        <g key={t} transform={`translate(0,${scale(t)})`}>
          <line className="grid-line" x1={x0} x2={x1} />
          <text className="ax-text" x={x0 - 8} dy="0.32em" textAnchor="end">{format(t)}</text>
        </g>
      ))}
      {title && (
        <text className="ax-title" x={titleX ?? x0} y={Math.min(scale.domain ? scale(scale.domain[1]) : 0, 0) - 12} textAnchor="start">{title}</text>
      )}
    </g>
  )
}

export function XAxis({ scale, ticks: ts, y, format = (v) => v, title, x0, x1, grid = false, y0 }) {
  return (
    <g>
      <line className="ax-line" x1={x0} x2={x1} y1={y} y2={y} />
      {ts.map((t) => (
        <g key={t} transform={`translate(${scale(t)},0)`}>
          {grid && <line className="grid-line" y1={y0} y2={y} />}
          <line className="ax-line" y1={y} y2={y + 4} />
          <text className="ax-text" y={y + 17} textAnchor="middle">{format(t)}</text>
        </g>
      ))}
      {title && <text className="ax-title" x={x1} y={y + 36} textAnchor="end">{title}</text>}
    </g>
  )
}

// ── Tooltip ─────────────────────────────────────────────────────────────────
export function Tooltip({ tip }) {
  if (!tip) return null
  return (
    <div className={`tooltip${tip.below ? ' below' : ''}`} style={{ left: tip.x, top: tip.y }} role="status">
      {tip.content}
    </div>
  )
}

export function TipHead({ fam, name, open, ring }) {
  return (
    <div className="tt-head">
      {fam && <MarkIcon fam={fam} size={12} open={open} ring={ring} />}
      <span>{name}</span>
    </div>
  )
}

export function TipRow({ k, v }) {
  return <div className="tt-row"><span>{k}</span><span>{v}</span></div>
}

// Tooltip state with sensible placement inside the chart box.
export function useTip(width) {
  const [tip, setTip] = useState(null)
  const show = (x, y, content) => {
    const cx = Math.max(90, Math.min(width - 90, x))
    setTip({ x: cx, y, below: y < 90, content })
  }
  return [tip, show, () => setTip(null)]
}

// ── Legend ──────────────────────────────────────────────────────────────────
export function FamilyLegend({ families = FAMILIES.map((f) => f.key), labels, extra }) {
  return (
    <div className="legend" aria-label="Judge families">
      {families.map((f) => (
        <span className="legend-item" key={f}>
          <MarkIcon fam={f} size={13} />
          {labels?.[f] ?? FAMILIES.find((x) => x.key === f)?.label ?? f}
        </span>
      ))}
      {extra}
    </div>
  )
}

export function BudgetLegend({ fam = 'Qwen3' }) {
  return (
    <div className="legend">
      <span className="legend-item"><MarkIcon fam={fam} size={13} /> B = 10</span>
      <span className="legend-item"><MarkIcon fam={fam} size={13} open /> B = 8,192</span>
    </div>
  )
}

// Collision-avoiding vertical placement for end labels.
export function spread(items, minGap, lo, hi) {
  const sorted = [...items].sort((a, b) => a.y - b.y)
  for (let i = 1; i < sorted.length; i++) {
    if (sorted[i].y - sorted[i - 1].y < minGap) sorted[i].y = sorted[i - 1].y + minGap
  }
  const over = sorted.length ? sorted[sorted.length - 1].y - hi : 0
  if (over > 0) {
    sorted[sorted.length - 1].y -= over
    for (let i = sorted.length - 2; i >= 0; i--) {
      if (sorted[i + 1].y - sorted[i].y < minGap) sorted[i].y = sorted[i + 1].y - minGap
    }
  }
  if (sorted.length && sorted[0].y < lo) {
    const d = lo - sorted[0].y
    sorted.forEach((s) => { s.y += d })
  }
  return sorted
}

// Interpolate between two theme colours read from CSS variables, for heatmaps.
export function useCssVars(names, theme) {
  const [vals, setVals] = useState(() => names.map(() => '#888'))
  useLayoutEffect(() => {
    const cs = getComputedStyle(document.documentElement)
    setVals(names.map((n) => cs.getPropertyValue(n).trim() || '#888'))
  }, [theme])
  return vals
}

const hex = (c) => {
  const m = c.replace('#', '')
  const n = parseInt(m.length === 3 ? m.split('').map((x) => x + x).join('') : m, 16)
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255]
}

export function ramp(stops) {
  const rgb = stops.map(hex)
  return (t) => {
    const x = Math.max(0, Math.min(1, t)) * (rgb.length - 1)
    const i = Math.min(rgb.length - 2, Math.floor(x))
    const k = x - i
    const c = rgb[i].map((v, j) => Math.round(v + (rgb[i + 1][j] - v) * k))
    return `rgb(${c[0]},${c[1]},${c[2]})`
  }
}

export function luminance(css) {
  const m = css.match(/\d+/g)
  if (!m) return 1
  const [r, g, b] = m.map((v) => +v / 255)
  return 0.2126 * r + 0.7152 * g + 0.0722 * b
}
