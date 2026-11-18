import { useRef, useState } from 'react'
import { Mark } from '../components/ui'
import { setHovered, useHovered } from '../lib/data'
import { linear, log, Tooltip, useTip, useWidth } from './kit'

// A chart whose marks keep their identity between states. Give it a new spec
// and every mark with the same id glides to its new place (CSS transitions on
// transform), marks that leave fade out and new ones fade in. The Findings
// stories, the Pick a judge view and the judge pages draw with it.
//
// spec: {
//   x: { domain, log?, ticks: [v | {v, label}], format, title, hide?, rotate? }
//   y: same as x, or { rows: [{ key, label }] } for one row per category
//   marks: [{ id, x, y, fam?, color?, open?, ring?, r?, shape?: 'dot'|'cell', cw?, ch?, text?, label?, key?, tip?, dim?, strong? }]
//   areas: [{ id, x0?, x1?, y0?, y1?, className, label?, labelAt? }]
//   rules: [{ id, axis: 'x'|'y', v, className?, label? }]
//   paths: [{ id, ids: [markId], step?: 'hv', className }]
//   notes: [{ id, x, y, text, anchor?, dx?, dy?, className? }]
//   alt, height, labelW
// }

const scaleOf = (ax, r0, r1) => (ax.log ? log : linear)(ax.domain, [r0, r1])
const tickList = (ax) => (ax.ticks || []).map((t) => (typeof t === 'object' ? t : { v: t, label: ax.format ? ax.format(t) : String(t) }))

function placeLabels(items, W, H, pad, narrow) {
  const boxes = []
  const marks = items.map((p) => ({ x0: p.px - 7, y0: p.py - 7, x1: p.px + 7, y1: p.py + 7 }))
  const hit = (b) => [...boxes, ...marks].some((o) => b.x0 < o.x1 && b.x1 > o.x0 && b.y0 < o.y1 && b.y1 > o.y0)
  const out = {}
  for (const p of items) {
    if (!p.label) continue
    const w = p.label.length * 6.7 + 4
    const side = p.labelSide
    const cands = [
      [11, 4, 'start'], [-11, 4, 'end'], [0, -12, 'middle'], [0, 21, 'middle'],
      [10, -9, 'start'], [-10, -9, 'end'], [10, 17, 'start'], [-10, 17, 'end'],
    ]
    if (side === 'l') cands.unshift([-11, 4, 'end'])
    if (side === 't') cands.unshift([0, -12, 'middle'])
    if (side === 'b') cands.unshift([0, 21, 'middle'])
    let placed = null
    for (const [dx, dy, anchor] of cands) {
      const x = p.px + dx
      const y = p.py + dy
      const x0 = anchor === 'start' ? x : anchor === 'end' ? x - w : x - w / 2
      const b = { x0, y0: y - 11, x1: x0 + w, y1: y + 3 }
      if (b.x0 < pad.l - 30 || b.x1 > W - 2 || b.y0 < 0 || b.y1 > H - pad.b + 6) continue
      if (!hit(b)) { placed = { dx, dy, anchor }; boxes.push(b); break }
    }
    if (!placed && p.force && !narrow) placed = { dx: 11, dy: 4, anchor: p.px > W * 0.7 ? 'end' : 'start' }
    if (placed) out[p.id] = placed
  }
  return out
}

function Shape({ mk }) {
  if (mk.shape === 'cell') {
    return (
      <g>
        <rect x={-mk.w / 2} y={-mk.h / 2} width={mk.w} height={mk.h} rx={5} className={mk.cls} style={mk.color ? { fill: mk.color } : undefined} />
        {mk.text && mk.w > 34 && <text className="st-cell-text" dy="0.34em" textAnchor="middle" style={mk.ink ? { fill: mk.ink } : undefined}>{mk.text}</text>}
      </g>
    )
  }
  if (mk.shape === 'dot' || !mk.fam) {
    return <circle r={mk.r ?? 4.5} className="st-dot" style={{ fill: mk.open ? 'var(--surface)' : mk.color || 'var(--accent-2)', stroke: mk.open ? mk.color || 'var(--accent-2)' : 'var(--surface)', strokeWidth: mk.open ? 1.6 : 1 }} />
  }
  return <Mark fam={mk.fam} r={mk.r ?? 6} open={mk.open} ring={mk.ring} color={mk.color} />
}

export default function Stage({ spec, height, className = '' }) {
  const [box, width] = useWidth(720)
  const [tip, showTip, hideTip] = useTip(width)
  const [hot, setHot] = useState(null)
  const linked = useHovered()
  const reg = useRef(new Map())
  const W = Math.max(280, width)
  const narrow = W < 560
  const H = height ?? spec.height ?? (narrow ? 330 : 440)
  const rows = spec.y.rows
  const xtAll = spec.x.hide ? [] : tickList(spec.x)
  const m = {
    t: spec.top ?? 26,
    r: spec.right ?? (narrow ? 14 : 26),
    b: spec.x.hide ? 10 : spec.x.rotate && narrow ? 70 : 46,
    l: rows ? (narrow ? spec.labelWNarrow ?? 92 : spec.labelW ?? 150) : narrow ? 40 : 50,
  }
  const sx = scaleOf(spec.x, m.l, W - m.r)
  // drop x ticks whose labels would touch the previous one
  const xt = []
  if ((spec.x.rotate && narrow) || spec.x.cat) xt.push(...xtAll)
  else {
    let lastEnd = -Infinity
    for (const t of xtAll) {
      const half = (String(t.label).length * 6.6 + 10) / 2
      const px = sx(t.v)
      if (px - half >= lastEnd) { xt.push(t); lastEnd = px + half }
    }
  }
  let sy
  let rowH = 0
  if (rows) {
    rowH = (H - m.t - m.b) / rows.length
    const idx = Object.fromEntries(rows.map((r, i) => [r.key, i]))
    sy = (k) => m.t + (idx[k] ?? 0) * rowH + rowH / 2
  } else sy = scaleOf(spec.y, H - m.b, m.t)
  const yt = rows ? [] : tickList(spec.y)

  const live = new Set()
  const pos = spec.marks.map((mk) => {
    let px = sx(mk.x) + (mk.dx || 0)
    let py = sy(mk.y) + (mk.dy || 0)
    px = Math.max(m.l - 10, Math.min(W - m.r + 10, px))
    py = Math.max(m.t - 12, Math.min(H - m.b + 6, py))
    const out = { ...mk, px, py }
    if (mk.shape === 'cell') {
      const cw = mk.cw ?? 0.9
      out.w = Math.abs(sx(mk.x + cw / 2) - sx(mk.x - cw / 2))
      out.h = rows ? rowH * (mk.ch ?? 0.82) : Math.abs(sy(mk.y + (mk.ch ?? 0.9) / 2) - sy(mk.y - (mk.ch ?? 0.9) / 2))
    }
    live.add(mk.id)
    reg.current.set(mk.id, out)
    return out
  })
  const all = [...reg.current.values()]
  const focus = hot || (linked && pos.some((p) => p.key === linked) ? linked : null)
  const labelled = pos.filter((p) => p.label && (p.strong || p.showLabel || (focus && (p.key === focus || p.id === focus))))
  const placed = placeLabels(labelled.sort((a, b) => (b.strong ? 1 : 0) - (a.strong ? 1 : 0)), W, H, m, narrow)
  const at = Object.fromEntries(pos.map((p) => [p.id, p]))

  const onMove = (e) => {
    const r = e.currentTarget.getBoundingClientRect()
    const mx = ((e.clientX - r.left) / r.width) * W
    const my = ((e.clientY - r.top) / r.height) * H
    let best = null
    let bd = 24 * 24
    for (const p of pos) {
      if (!p.tip) continue
      const d = p.shape === 'cell'
        ? (Math.abs(p.px - mx) <= p.w / 2 && Math.abs(p.py - my) <= p.h / 2 ? 0 : Infinity)
        : (p.px - mx) ** 2 + (p.py - my) ** 2
      if (d < bd) { bd = d; best = p }
    }
    if (!best) { setHot(null); hideTip(); setHovered(null); return }
    setHot(best.key || best.id)
    if (best.key) setHovered(best.key)
    showTip((best.px / W) * r.width, (best.py / H) * r.height, typeof best.tip === 'function' ? best.tip() : best.tip)
  }
  const onLeave = () => { setHot(null); hideTip(); setHovered(null) }

  const area = (a) => {
    const x0 = a.x0 == null ? m.l : sx(a.x0)
    const x1 = a.x1 == null ? W - m.r : sx(a.x1)
    const yA = a.y0 == null ? H - m.b : rows ? sy(a.y0) - rowH / 2 : sy(a.y0)
    const yB = a.y1 == null ? m.t : rows ? sy(a.y1) + rowH / 2 : sy(a.y1)
    return { x: Math.min(x0, x1), y: Math.min(yA, yB), w: Math.abs(x1 - x0), h: Math.abs(yB - yA) }
  }

  return (
    <div className={`chart stage ${className}`} ref={box}>
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={spec.alt} onPointerMove={onMove} onPointerDown={onMove} onPointerLeave={onLeave}>
        {(spec.areas || []).map((a) => {
          const g = area(a)
          return (
            <g key={a.id} className="st-area-g">
              <rect width="1" height="1" className={`st-area ${a.className || ''}`} style={{ transform: `translate(${g.x}px,${g.y}px) scale(${Math.max(0.01, g.w)},${Math.max(0.01, g.h)})` }} />
              {a.label && !narrow && (
                <text className="st-note muted st-move" textAnchor={a.labelAnchor || 'end'}
                  style={{ transform: `translate(${a.labelAnchor === 'start' ? g.x + 8 : g.x + g.w - 8}px,${a.labelInside ? g.y + g.h - 9 : g.y - 7}px)` }}>{a.label}</text>
              )}
            </g>
          )
        })}
        {yt.map((t) => (
          <g key={`y${t.v}`} className="st-tick st-move" style={{ transform: `translate(0px,${sy(t.v)}px)` }}>
            <line className="grid-line" x1={m.l} x2={W - m.r} />
            <text className="ax-text" x={m.l - 8} dy="0.32em" textAnchor="end">{t.label}</text>
          </g>
        ))}
        {!rows && spec.y.title && <text key={`yt-${spec.y.title}`} className="ax-title st-fade" x={m.l - 8} y={m.t - 12}>{spec.y.title}</text>}
        {rows && rows.map((r) => (
          <g key={`r${r.key}`} className="st-tick st-move" style={{ transform: `translate(0px,${sy(r.key)}px)` }}>
            {r.band && <rect x={0} y={-rowH / 2} width={W} height={rowH} className="st-rowband" />}
            <text className="ax-text st-rowlabel" x={m.l - 12} dy="0.33em" textAnchor="end">{r.label}</text>
          </g>
        ))}
        {!spec.x.hide && <line className="ax-line" x1={m.l} x2={W - m.r} y1={H - m.b} y2={H - m.b} />}
        {xt.map((t) => (
          <g key={`x${t.v}`} className="st-tick st-move" style={{ transform: `translate(${sx(t.v)}px,0px)` }}>
            {spec.x.grid !== false && <line className="grid-line" y1={m.t} y2={H - m.b} />}
            <text className="ax-text" y={H - m.b + 18} textAnchor={spec.x.rotate && narrow ? 'end' : 'middle'}
              transform={spec.x.rotate && narrow ? `rotate(-35 0 ${H - m.b + 18})` : undefined}>{t.label}</text>
          </g>
        ))}
        {!spec.x.hide && spec.x.title && <text key={`xt-${spec.x.title}`} className="ax-title st-fade" x={W - m.r} y={H - 6} textAnchor="end">{spec.x.title}</text>}
        {(spec.rules || []).map((r) => (
          <g key={r.id} className="st-move" style={{ transform: r.axis === 'x' ? `translate(${sx(r.v)}px,0px)` : `translate(0px,${rows ? 0 : sy(r.v)}px)` }}>
            {r.axis === 'x'
              ? <line className={`st-rule ${r.className || ''}`} y1={m.t - 6} y2={H - m.b} />
              : <line className={`st-rule ${r.className || ''}`} x1={m.l} x2={W - m.r} />}
            {r.label && (() => {
              const flip = r.axis === 'x' && sx(r.v) > m.l + (W - m.l - m.r) * 0.62
              return (
                <text className={`st-note ${r.labelClass || 'muted'}`} x={r.axis === 'x' ? (flip ? -6 : 6) : W - m.r - 4} y={r.axis === 'x' ? m.t + 4 : -7}
                  textAnchor={r.axis === 'x' && !flip ? 'start' : 'end'}>{r.label}</text>
              )
            })()}
          </g>
        ))}
        {(spec.paths || []).map((p) => {
          const pts = p.ids.map((id) => at[id]).filter(Boolean)
          if (pts.length < 2) return null
          const d = pts.map((q, i) => (i === 0 ? `M${q.px.toFixed(1)},${q.py.toFixed(1)}` : p.step === 'hv' ? `H${q.px.toFixed(1)}V${q.py.toFixed(1)}` : `L${q.px.toFixed(1)},${q.py.toFixed(1)}`)).join('')
          return <path key={`${p.id}-${spec.key}-${W}`} d={d} className={`st-path ${p.className || ''}`} />
        })}
        <g>
          {all.map((mk) => {
            const on = live.has(mk.id)
            const cur = on ? at[mk.id] : mk
            const isFocus = focus && (cur.key === focus || cur.id === focus)
            const dim = on && (cur.dim || (focus && !isFocus && cur.key !== undefined))
            return (
              <g key={mk.id} className={`st-mark${on ? '' : ' out'}${dim ? ' dim' : ''}${isFocus ? ' focus' : ''}`}
                style={{ transform: `translate(${cur.px}px,${cur.py}px)` }} aria-hidden="true">
                {isFocus && cur.shape !== 'cell' && <circle r={(cur.r ?? 6) + 7} className="st-ring" />}
                <Shape mk={cur} />
              </g>
            )
          })}
        </g>
        {(spec.notes || []).map((n) => (
          <text key={n.id} className={`st-note st-move ${n.className || 'muted'}`} textAnchor={n.anchor || 'start'}
            style={{ transform: `translate(${(n.px ?? sx(n.x)) + (n.dx || 0)}px,${(n.py ?? (rows ? sy(n.y) : sy(n.y))) + (n.dy || 0)}px)` }}>{n.text}</text>
        ))}
        {labelled.filter((p) => placed[p.id]).map((p) => {
          const L = placed[p.id]
          return (
            <text key={`l${p.id}`} className={`lbl st-move st-label${p.strong ? ' strong' : ''}`} textAnchor={L.anchor}
              style={{ transform: `translate(${p.px + L.dx}px,${p.py + L.dy}px)` }}>{p.label}</text>
          )
        })}
      </svg>
      <Tooltip tip={tip} />
    </div>
  )
}
