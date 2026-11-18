import { useId, useMemo, useState } from 'react'
import { Mark } from '../components/ui'
import { useInView, useTween } from '../lib/motion'
import { useTheme } from '../lib/theme'
import { fmt } from '../lib/format'
import {
  band, decimalsFor, linear, log, luminance, ramp, spread, ticks,
  Tooltip, useCssVars, useTip, useWidth, XAxis, YAxis,
} from './kit'

const mergeRefs = (...refs) => (el) => refs.forEach((r) => { r.current = el })
const narrow = (w) => w < 520

// ── Dot plot: one row per item, dots along a numeric axis ───────────────────
// rows: [{ key, label, sub?, dots: [{ id, v, fam, open, color, r, tip }], range?: [lo, hi] }]
export function DotPlot({ rows, domain, format = (v) => fmt(v, 1), xTitle, refs = [], rowH = 28, labelW, height, ariaLabel }) {
  const [box, width] = useWidth()
  const [inRef, seen] = useInView()
  const [tip, showTip, hideTip] = useTip(width)
  const [hot, setHot] = useState(null)
  const LW = labelW ?? (narrow(width) ? 104 : 168)
  const hasNote = rows.some((r) => r.note)
  const m = { t: 10, r: hasNote ? (narrow(width) ? 82 : 104) : 18, b: xTitle ? 44 : 28, l: LW }
  const H = height ?? m.t + m.b + rows.length * rowH
  const flat = rows.flatMap((r) => r.dots.map((d) => d.v ?? NaN))
  const target = [domain[0], domain[1], ...flat]
  const from = [domain[0], domain[1], ...flat.map(() => domain[0])]
  const tw = useTween(target, { from, run: seen, duration: 800 })
  const x = linear([tw[0], tw[1]], [m.l, width - m.r])
  const xt = ticks(tw[0], tw[1], narrow(width) ? 4 : 6).filter((t) => t >= tw[0] - 1e-9 && t <= tw[1] + 1e-9)
  const nd = decimalsFor(ticks(domain[0], domain[1], narrow(width) ? 4 : 6))
  let k = 2
  return (
    <div className="chart" ref={mergeRefs(box, inRef)}>
      <svg viewBox={`0 0 ${width} ${H}`} role="img" aria-label={ariaLabel} onPointerLeave={() => { hideTip(); setHot(null) }}>
        {xt.map((t) => (
          <line key={t} className="grid-line" x1={x(t)} x2={x(t)} y1={m.t} y2={H - m.b} />
        ))}
        <XAxis scale={x} ticks={xt} y={H - m.b} x0={m.l} x1={width - m.r} format={(t) => fmt(t, nd)} title={xTitle} />
        {refs.map((r, i) => (
          <g key={i}>
            <line className="ref-line" x1={x(r.v)} x2={x(r.v)} y1={m.t - 4} y2={H - m.b} />
            {r.label && <text className="lbl muted" x={x(r.v) + 5} y={m.t + 8} style={{ fontSize: 11 }}>{r.label}</text>}
          </g>
        ))}
        {rows.map((row, i) => {
          const cy = m.t + i * rowH + rowH / 2
          const vs = row.dots.map(() => tw[k++])
          const valid = vs.filter((v) => !Number.isNaN(v))
          const lo = row.range ? Math.min(...valid) : null
          const hi = row.range ? Math.max(...valid) : null
          return (
            <g key={row.key} opacity={hot && hot !== row.key ? 0.45 : 1} style={{ transition: 'opacity .25s' }}>
              {i % 2 === 0 && <rect x={0} y={cy - rowH / 2} width={width} height={rowH} style={{ fill: 'var(--surface-2)', opacity: 0.55 }} />}
              <text x={LW - 12} y={cy} dy="0.33em" textAnchor="end" className="ax-text" style={{ fill: 'var(--ink-2)', fontSize: narrow(width) ? 11 : 12 }}>
                {row.label}
              </text>
              {row.note && <text x={width - 6} y={cy} dy="0.33em" textAnchor="end" className="ax-text" style={{ fill: 'var(--ink-2)' }}>{row.note}</text>}
              {row.range && valid.length > 1 && (
                <line x1={x(lo)} x2={x(hi)} y1={cy} y2={cy} style={{ stroke: row.rangeColor || 'var(--line-2)' }} strokeWidth="3" strokeLinecap="round" />
              )}
              {row.dots.map((d, di) => {
                const v = vs[di]
                if (Number.isNaN(v) || d.v == null) return null
                return (
                  <g key={d.id ?? di} transform={`translate(${x(v)},${cy})`}
                    onPointerEnter={() => {
                      setHot(row.key)
                      if (d.tip) showTip(((x(v)) / width) * (box.current?.clientWidth || width), (cy / H) * (box.current?.clientHeight || H), d.tip)
                    }}
                    onPointerLeave={hideTip}
                  >
                    {d.fam ? <Mark fam={d.fam} r={d.r ?? 5} open={d.open} color={d.color} /> : <circle r={d.r ?? 4.5} style={{ fill: d.color || 'var(--navy-mid)', stroke: 'var(--surface)' }} strokeWidth="1.2" />}
                    <circle r={10} className="hit" />
                  </g>
                )
              })}
            </g>
          )
        })}
      </svg>
      <Tooltip tip={tip} />
    </div>
  )
}

// ── Scatter ─────────────────────────────────────────────────────────────────
// points: [{ key, x, y, fam, open, ring, label, tip, hidden }]
export function Scatter({
  points, xDomain, yDomain, xLog = false, xFormat = (v) => fmt(v, 0), yFormat = (v) => fmt(v, 0),
  xTitle, yTitle, height = 360, labels = [], hLines = [], vLines = [], quadrants, xTicks, line, ariaLabel,
}) {
  const [box, width] = useWidth()
  const [inRef, seen] = useInView()
  const [tip, showTip, hideTip] = useTip(width)
  const [hover, setHover] = useState(null)
  const m = { t: 30, r: 20, b: 48, l: 46 }
  const H = height
  const target = [...xDomain, ...yDomain, ...points.flatMap((p) => [p.x, p.y])]
  const from = [...xDomain, ...yDomain, ...points.flatMap((p) => [p.x, yDomain[0]])]
  const tw = useTween(target, { from, run: seen, duration: 850 })
  const x = (xLog ? log : linear)([tw[0], tw[1]], [m.l, width - m.r])
  const y = linear([tw[2], tw[3]], [H - m.b, m.t])
  const yt = ticks(tw[2], tw[3], 5).filter((t) => t >= tw[2] - 1e-9 && t <= tw[3] + 1e-9)
  const xt = xTicks ?? ticks(tw[0], tw[1], narrow(width) ? 4 : 6).filter((t) => t >= tw[0] - 1e-9 && t <= tw[1] + 1e-9)
  const pts = points.map((p, i) => ({ ...p, px: x(tw[4 + i * 2]), py: y(tw[5 + i * 2]) }))
  const onMove = (e) => {
    const r = e.currentTarget.getBoundingClientRect()
    const mx = ((e.clientX - r.left) / r.width) * width
    const my = ((e.clientY - r.top) / r.height) * H
    let best = null
    let bd = 24 * 24
    for (const p of pts) {
      if (p.hidden) continue
      const d = (p.px - mx) ** 2 + (p.py - my) ** 2
      if (d < bd) { bd = d; best = p }
    }
    if (!best) { hideTip(); setHover(null); return }
    setHover(best.key)
    if (best.tip) showTip((best.px / width) * r.width, (best.py / H) * r.height, best.tip)
  }
  const shown = new Set([...labels, hover])
  const lineD = line && pts.filter((p) => line.keys.includes(p.key)).sort((a, b) => line.keys.indexOf(a.key) - line.keys.indexOf(b.key))
    .map((p, i) => `${i ? 'L' : 'M'}${p.px.toFixed(1)},${p.py.toFixed(1)}`).join('')
  return (
    <div className="chart" ref={mergeRefs(box, inRef)}>
      <svg viewBox={`0 0 ${width} ${H}`} role="img" aria-label={ariaLabel} onPointerMove={onMove} onPointerDown={onMove} onPointerLeave={() => { hideTip(); setHover(null) }}>
        <YAxis scale={y} ticks={yt} x0={m.l} x1={width - m.r} format={yFormat} />
        {yTitle && <text className="ax-title" x={m.l - 8} y={m.t - 16}>{yTitle}</text>}
        <XAxis scale={x} ticks={xt} y={H - m.b} x0={m.l} x1={width - m.r} format={xFormat} title={xTitle} />
        {quadrants && quadrants.map((q, i) => (
          <text key={i} className="lbl muted" x={q.right ? width - m.r - 6 : m.l + 6} y={q.top ? m.t + 4 : H - m.b - 8} textAnchor={q.right ? 'end' : 'start'} style={{ fontSize: 11 }}>{q.text}</text>
        ))}
        {hLines.map((l, i) => <line key={`h${i}`} className="ref-line" x1={m.l} x2={width - m.r} y1={y(l)} y2={y(l)} />)}
        {vLines.map((l, i) => <line key={`v${i}`} className="ref-line" y1={m.t} y2={H - m.b} x1={x(l)} x2={x(l)} />)}
        {lineD && <path d={lineD} fill="none" style={{ stroke: line.color, opacity: seen ? 0.5 : 0, transition: 'opacity .6s .5s' }} strokeWidth="1.3" strokeDasharray="3 3" />}
        {pts.map((p) => (
          <g key={p.key} transform={`translate(${p.px},${p.py})`}>
            <Mark fam={p.fam} r={6} open={p.open} ring={p.ring} className={`mark${hover && hover !== p.key ? ' dim' : ''}`} style={{ opacity: p.hidden ? 0 : undefined }} />
          </g>
        ))}
        {pts.filter((p) => !p.hidden && shown.has(p.key)).map((p) => {
          const right = p.px < width * 0.7
          return <text key={`l${p.key}`} className="lbl" x={p.px + (right ? 10 : -10)} y={p.py - 9} textAnchor={right ? 'start' : 'end'}>{p.label ?? p.key}</text>
        })}
      </svg>
      <Tooltip tip={tip} />
    </div>
  )
}

// ── Lines over a (log) x axis ───────────────────────────────────────────────
// series: [{ key, label, fam, values: [] }], xs: []
export function LineChart({ series, xs, yDomain, active, height = 360, xTitle, yTitle, xFormat = (v) => v, yFormat = (v) => fmt(v, 0), ariaLabel, onPick }) {
  const [box, width] = useWidth()
  const [inRef, seen] = useInView()
  const [tip, showTip, hideTip] = useTip(width)
  const [idx, setIdx] = useState(null)
  const clip = `reveal${useId().replace(/[^a-zA-Z0-9]/g, '')}`
  const m = { t: 30, r: narrow(width) ? 14 : 132, b: 48, l: 44 }
  const H = height
  const ty = useTween([...yDomain], { run: true })
  const x = log([xs[0], xs[xs.length - 1]], [m.l, width - m.r])
  const y = linear([ty[0], ty[1]], [H - m.b, m.t])
  const yt = ticks(ty[0], ty[1], 5).filter((t) => t >= ty[0] - 1e-9 && t <= ty[1] + 1e-9)
  const on = (s) => !active || active.includes(s.key)
  const path = (vals) => vals.map((v, i) => `${i ? 'L' : 'M'}${x(xs[i]).toFixed(1)},${y(Math.max(ty[0], v)).toFixed(1)}`).join('')
  const onMove = (e) => {
    const r = e.currentTarget.getBoundingClientRect()
    const mx = ((e.clientX - r.left) / r.width) * width
    let bi = 0
    let bd = Infinity
    xs.forEach((v, i) => { const d = Math.abs(x(v) - mx); if (d < bd) { bd = d; bi = i } })
    setIdx(bi)
    const act = series.filter(on).sort((a, b) => b.values[bi] - a.values[bi])
    showTip((x(xs[bi]) / width) * r.width, (m.t + 20) / H * r.height, (
      <>
        <div className="tt-head">B = {xFormat(xs[bi])} tokens</div>
        {act.slice(0, 8).map((s) => <div className="tt-row" key={s.key}><span>{s.label}</span><span>{fmt(s.values[bi])}%</span></div>)}
      </>
    ))
  }
  const ends = spread(series.filter(on).map((s) => ({ key: s.key, s, y: y(Math.max(ty[0], s.values[s.values.length - 1])) })), 14, m.t, H - m.b)
  return (
    <div className="chart" ref={mergeRefs(box, inRef)}>
      <svg viewBox={`0 0 ${width} ${H}`} role="img" aria-label={ariaLabel} onPointerMove={onMove} onPointerLeave={() => { setIdx(null); hideTip() }}>
        <defs>
          <clipPath id={clip}><rect x={0} y={0} height={H} width={seen ? width : 0} style={{ transition: 'width 1.4s cubic-bezier(.2,.7,.2,1)' }} /></clipPath>
        </defs>
        <YAxis scale={y} ticks={yt} x0={m.l} x1={width - m.r} format={yFormat} />
        {yTitle && <text className="ax-title" x={m.l - 8} y={m.t - 16}>{yTitle}</text>}
        <XAxis scale={x} ticks={xs.filter((v, i) => i === 0 || i === xs.length - 1 || (width - m.l - m.r) / xs.length > 52 || i % 2 === 0)} y={H - m.b} x0={m.l} x1={width - m.r} format={xFormat} title={xTitle} />
        {idx !== null && <line x1={x(xs[idx])} x2={x(xs[idx])} y1={m.t} y2={H - m.b} style={{ stroke: 'var(--line-2)' }} />}
        <g clipPath={`url(#${clip})`}>
          {series.filter((s) => !on(s)).map((s) => (
            <path key={s.key} d={path(s.values)} fill="none" style={{ stroke: 'var(--line-2)' }} strokeWidth="1.2" />
          ))}
          {series.filter(on).map((s) => (
            <g key={s.key}>
              <path d={path(s.values)} fill="none" style={{ stroke: `var(--fam-${s.slug})` }} strokeWidth="2" strokeLinejoin="round" />
              {s.values.map((v, i) => (
                <g key={i} transform={`translate(${x(xs[i])},${y(Math.max(ty[0], v))})`}><Mark fam={s.fam} r={idx === i ? 4.6 : 3.4} /></g>
              ))}
            </g>
          ))}
        </g>
        {!narrow(width) && ends.map((e) => (
          <text key={e.key} className="lbl" x={width - m.r + 10} y={e.y} dy="0.33em" style={{ fontSize: 11.5, cursor: onPick ? 'pointer' : 'default' }} onClick={() => onPick?.(e.key)}>{e.s.label}</text>
        ))}
      </svg>
      <Tooltip tip={tip} />
    </div>
  )
}

// ── Horizontal bars ─────────────────────────────────────────────────────────
// rows: [{ key, label, v, color, fam, tip, open }]
export function BarList({ rows, domain, format = (v) => fmt(v, 3), rowH = 22, labelW, xTitle, ariaLabel, valueW = 52 }) {
  const [box, width] = useWidth()
  const [inRef, seen] = useInView()
  const [tip, showTip, hideTip] = useTip(width)
  const LW = labelW ?? (narrow(width) ? 116 : 176)
  const m = { t: 6, r: valueW, b: xTitle ? 40 : 24, l: LW }
  const H = m.t + m.b + rows.length * rowH
  const vals = useTween([domain[0], domain[1], ...rows.map((r) => r.v ?? 0)], { from: [domain[0], domain[1], ...rows.map(() => 0)], run: seen, duration: 900, ease: 'out' })
  const x = linear([vals[0], vals[1]], [m.l, width - m.r])
  const zero = x(Math.max(vals[0], Math.min(0, vals[1])))
  const xt = ticks(vals[0], vals[1], narrow(width) ? 3 : 5).filter((t) => t >= vals[0] - 1e-9 && t <= vals[1] + 1e-9)
  const nd = decimalsFor(xt)
  return (
    <div className="chart" ref={mergeRefs(box, inRef)}>
      <svg viewBox={`0 0 ${width} ${H}`} role="img" aria-label={ariaLabel} onPointerLeave={hideTip}>
        {xt.map((t) => <line key={t} className="grid-line" x1={x(t)} x2={x(t)} y1={m.t} y2={H - m.b} />)}
        <XAxis scale={x} ticks={xt} y={H - m.b} x0={m.l} x1={width - m.r} format={(t) => fmt(t, nd)} title={xTitle} />
        <line x1={zero} x2={zero} y1={m.t} y2={H - m.b} className="ax-line" />
        {rows.map((r, i) => {
          const v = vals[i + 2]
          const cy = m.t + i * rowH
          const x1 = x(v)
          const bx = Math.min(zero, x1)
          const bw = Math.abs(x1 - zero)
          return (
            <g key={r.key}
              onPointerEnter={() => r.tip && showTip(((bx + bw / 2) / width) * (box.current?.clientWidth || width), (cy / H) * (box.current?.clientHeight || H), r.tip)}
              onPointerLeave={hideTip}>
              <rect x={0} y={cy} width={width} height={rowH} className="hit" />
              <text x={LW - 10} y={cy + rowH / 2} dy="0.33em" textAnchor="end" className="ax-text" style={{ fill: r.strong ? 'var(--ink)' : 'var(--ink-2)', fontWeight: r.strong ? 600 : 400, fontSize: narrow(width) ? 11 : 12 }}>{r.label}</text>
              <rect x={bx} y={cy + 4} width={bw} height={rowH - 8} rx={3}
                style={{ fill: r.open ? 'var(--surface)' : (r.color || 'var(--navy-mid)'), stroke: r.color || 'var(--navy-mid)', strokeWidth: r.open ? 1.4 : 0 }} />
              {r.v != null && <text x={(r.v >= 0 ? Math.max(x1, zero) + 6 : Math.max(x1, zero) + 6)} y={cy + rowH / 2} dy="0.33em" className="ax-text" style={{ fill: 'var(--ink-2)' }}>{format(r.v)}</text>}
            </g>
          )
        })}
      </svg>
      <Tooltip tip={tip} />
    </div>
  )
}

// ── Heatmap ─────────────────────────────────────────────────────────────────
// rows: [{ key, label, fam }], cols: [{ key, label }], value(rowKey, colKey) → number|null
export function Heatmap({ rows, cols, value, domain, format = (v) => fmt(v, 1), tip, extraCol, ariaLabel }) {
  const [box, width] = useWidth()
  const [inRef, seen] = useInView()
  const theme = useTheme()
  const [tipS, showTip, hideTip] = useTip(width)
  const [hot, setHot] = useState(null)
  const stops = useCssVars(['--seq-0', '--seq-1', '--seq-2', '--seq-3', '--seq-4', '--seq-5'], theme)
  const color = useMemo(() => ramp(stops), [stops.join()])
  const small = narrow(width)
  const LW = small ? 108 : 172
  const EW = extraCol ? (small ? 44 : 64) : 0
  const m = { t: small ? 64 : 34, r: EW + 4, b: 8, l: LW }
  const cw = (width - m.l - m.r) / cols.length
  const rh = small ? 22 : 26
  const H = m.t + m.b + rows.length * rh
  const x = band(cols.map((c) => c.key), [m.l, width - m.r], 0.06)
  const showText = cw > 46
  return (
    <div className="chart" ref={mergeRefs(box, inRef)}>
      <svg viewBox={`0 0 ${width} ${H}`} role="img" aria-label={ariaLabel} onPointerLeave={() => { hideTip(); setHot(null) }}>
        {cols.map((c) => (
          <text key={c.key} className="ax-text" x={x(c.key) + x.bw / 2} y={m.t - 10}
            textAnchor={small ? 'start' : 'middle'} transform={small ? `rotate(-50 ${x(c.key) + x.bw / 2} ${m.t - 10})` : undefined}
            style={{ fontWeight: hot?.c === c.key ? 600 : 400, fill: hot?.c === c.key ? 'var(--ink)' : undefined }}>{c.label}</text>
        ))}
        {extraCol && <text className="ax-text" x={width - EW / 2} y={m.t - 10} textAnchor="middle" style={{ fontWeight: 600 }}>{extraCol.label}</text>}
        {rows.map((r, i) => {
          const cy = m.t + i * rh
          return (
            <g key={r.key}>
              <text x={LW - 10} y={cy + rh / 2} dy="0.33em" textAnchor="end" className="ax-text" style={{ fill: hot?.r === r.key ? 'var(--ink)' : 'var(--ink-2)', fontWeight: hot?.r === r.key ? 600 : 400, fontSize: small ? 11 : 12 }}>{r.label}</text>
              {cols.map((c, ci) => {
                const v = value(r.key, c.key)
                const t = v == null ? 0 : (v - domain[0]) / (domain[1] - domain[0])
                const fill = v == null ? 'var(--surface-2)' : color(t)
                const dark = luminance(fill) < 0.5
                return (
                  <g key={c.key}
                    onPointerEnter={() => {
                      setHot({ r: r.key, c: c.key })
                      if (tip && v != null) showTip(((x(c.key) + x.bw / 2) / width) * (box.current?.clientWidth || width), (cy / H) * (box.current?.clientHeight || H), tip(r, c, v))
                    }}>
                    <rect className="heat-cell" x={x(c.key)} y={cy + 1} width={x.bw} height={rh - 2} rx={3}
                      style={{ fill, opacity: seen ? 1 : 0, transition: `fill .5s, opacity .5s ${(i * 0.02 + ci * 0.03).toFixed(2)}s`, stroke: hot?.r === r.key && hot?.c === c.key ? 'var(--ink)' : 'none', strokeWidth: 1.5 }} />
                    {showText && v != null && (
                      <text className="heat-txt" x={x(c.key) + x.bw / 2} y={cy + rh / 2} dy="0.34em" textAnchor="middle" style={{ fill: dark ? '#fff' : '#13213a', opacity: seen ? 1 : 0, transition: 'opacity .5s .4s' }}>{format(v)}</text>
                    )}
                  </g>
                )
              })}
              {extraCol && (
                <text x={width - EW / 2} y={cy + rh / 2} dy="0.33em" textAnchor="middle" className="ax-text" style={{ fill: 'var(--ink)', fontWeight: 500 }}>{extraCol.value(r.key)}</text>
              )}
            </g>
          )
        })}
      </svg>
      <Tooltip tip={tipS} />
    </div>
  )
}

// ── Bars on a log axis, with two value columns ──────────────────────────────
// rows: [{ key, label, fam, slug, open, v, right, right2, strong, tip }]
export function LogBars({ rows, domain, xTitle, format = (v) => v, rowH = 24, ariaLabel }) {
  const [box, width] = useWidth()
  const [inRef, seen] = useInView()
  const [tip, showTip, hideTip] = useTip(width)
  const small = narrow(width)
  const LW = small ? 120 : 176
  const RW = small ? 64 : 150
  const m = { t: 6, r: RW, b: xTitle ? 42 : 26, l: LW }
  const H = m.t + m.b + rows.length * rowH
  const tw = useTween([domain[0], domain[1], ...rows.map((r) => r.v)], { from: [domain[0], domain[1], ...rows.map(() => domain[0])], run: seen, duration: 950, ease: 'out' })
  const x = log([tw[0], tw[1]], [m.l, width - m.r])
  const ts = []
  for (let v = domain[0]; v <= domain[1] * 1.0001; v *= 10) ts.push(v)
  return (
    <div className="chart" ref={mergeRefs(box, inRef)}>
      <svg viewBox={`0 0 ${width} ${H}`} role="img" aria-label={ariaLabel} onPointerLeave={hideTip}>
        {ts.map((t) => <line key={t} className="grid-line" x1={x(t)} x2={x(t)} y1={m.t} y2={H - m.b} />)}
        <XAxis scale={x} ticks={ts} y={H - m.b} x0={m.l} x1={width - m.r} format={format} title={xTitle} />
        {rows.map((r, i) => {
          const cy = m.t + i * rowH
          const v = tw[i + 2]
          return (
            <g key={r.key}
              onPointerEnter={() => r.tip && showTip((x(v) / width) * (box.current?.clientWidth || width), (cy / H) * (box.current?.clientHeight || H), r.tip)}
              onPointerLeave={hideTip}>
              <rect x={0} y={cy} width={width} height={rowH} className="hit" />
              {r.strong && <rect x={0} y={cy + 1} width={width} height={rowH - 2} rx={6} style={{ fill: 'var(--tint)', opacity: 0.55 }} />}
              <text x={LW - 12} y={cy + rowH / 2} dy="0.33em" textAnchor="end" className="ax-text" style={{ fill: r.strong ? 'var(--ink)' : 'var(--ink-2)', fontWeight: r.strong ? 600 : 400, fontSize: small ? 11 : 12 }}>{r.label}</text>
              <line x1={m.l} x2={width - m.r} y1={cy + rowH / 2} y2={cy + rowH / 2} className="row-guide" />
              <g transform={`translate(${x(v)},${cy + rowH / 2})`}><Mark fam={r.fam} r={r.strong ? 5 : 4.4} open={r.open} /></g>
              <text x={width - RW + 10} y={cy + rowH / 2} dy="0.33em" className="ax-text" style={{ fill: 'var(--ink)', fontWeight: 500 }}>{r.right}</text>
              {!small && <text x={width - 6} y={cy + rowH / 2} dy="0.33em" textAnchor="end" className="ax-text">{r.right2}</text>}
            </g>
          )
        })}
      </svg>
      <Tooltip tip={tip} />
    </div>
  )
}
