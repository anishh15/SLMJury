import { useMemo, useRef, useState } from 'react'
import { Mark, Segmented } from '../components/ui'
import { log, linear, Tooltip, TipHead, TipRow, useTip, useWidth } from '../charts/kit'
import { DataTable, Toolbar } from '../charts/Frame'
import { byKey, claims, core, famColor, onFront, points, QUICK, REASONED, setHovered, useHovered } from '../lib/data'
import { count, fmt } from '../lib/format'
import { useInView, useTween } from '../lib/motion'

const COST_TICKS = [0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300]
const SIZE_TICKS = [0.5, 1, 2, 5, 10, 20, 50, 100]
const LABELLED = ['phi4-14b', claims.topLarge, 'qwen2.5-72b', 'llama3.3-70b', 'gpt-oss-120b', 'glm-4.5-air', 'phi4mi-3.8b', 'qwen3-8b']
const money = (v) => (v >= 1 ? `$${fmt(v, 0)}` : `$${v >= 0.1 ? fmt(v, 1) : fmt(v, 2)}`)

// Place labels greedily: right, left, above, below of each mark, avoiding
// marks and labels already placed.
function placeLabels(items, W, H, pad) {
  const boxes = []
  const marks = items.map((p) => ({ x0: p.x - 8, y0: p.y - 8, x1: p.x + 8, y1: p.y + 8 }))
  const hit = (b) => [...boxes, ...marks].some((o) => b.x0 < o.x1 && b.x1 > o.x0 && b.y0 < o.y1 && b.y1 > o.y0)
  const out = []
  for (const p of items.filter((x) => x.label)) {
    const w = p.label.length * 6.9 + 6
    const cands = [
      [p.x + 11, p.y + 4, 'start'], [p.x - 11, p.y + 4, 'end'],
      [p.x, p.y - 13, 'middle'], [p.x, p.y + 21, 'middle'],
      [p.x + 11, p.y - 10, 'start'], [p.x - 11, p.y - 10, 'end'], [p.x + 11, p.y + 18, 'start'], [p.x - 11, p.y + 18, 'end'],
    ]
    for (const [x, y, anchor] of cands) {
      const x0 = anchor === 'start' ? x : anchor === 'end' ? x - w : x - w / 2
      const b = { x0, y0: y - 12, x1: x0 + w, y1: y + 3 }
      if (b.x0 < pad.l || b.x1 > W - 4 || b.y0 < 4 || b.y1 > H - pad.b) continue
      if (!hit(b)) { boxes.push(b); out.push({ key: p.key, x, y, anchor, text: p.label, strong: p.strong }); break }
    }
  }
  return out
}

export default function CostChart() {
  const [mode, setMode] = useState('cost')
  const [budgets, setBudgets] = useState(false)
  const [view, setView] = useState('chart')
  const body = useRef(null)
  const [box, width] = useWidth(1100)
  const [inRef, seen] = useInView()
  const [tip, showTip, hideTip] = useTip(width)
  const hovered = useHovered()
  const small = width < 640
  const H = small ? 440 : 540
  const FLOOR = 84
  const top = true
  const pad = { l: small ? 40 : 54, r: small ? 12 : 28, t: 24, b: top ? 84 : 52 }
  const plotBottom = H - pad.b
  const floorY = plotBottom + 30

  const rows = useMemo(() => points.map((p) => ({ ...p, j: byKey[p.key] })), [])
  const extra = rows.filter((p) => p.other)
  const xOf = (p, o) => (mode === 'cost' ? (o ? o.costExact : p.costExact) : p.j.params)
  const yOf = (acc) => acc
  const xd = mode === 'cost' ? [0.016, 700] : [0.42, 180]
  const yd = top ? [FLOOR, 91.2] : [40, 92.5]
  const target = [...xd, ...yd, ...rows.flatMap((p) => [xOf(p), yOf(p.acc)]), ...extra.flatMap((p) => [xOf(p, p.other), yOf(p.other.acc)])]
  const from = [...xd, ...yd, ...rows.flatMap((p) => [xOf(p), yd[0]]), ...extra.flatMap((p) => [xOf(p, p.other), yd[0]])]
  const tw = useTween(target, { from, run: seen, duration: 720 })
  const x = log([tw[0], tw[1]], [pad.l, width - pad.r])
  const y = linear([tw[2], tw[3]], [plotBottom, pad.t])
  const yClamp = (acc) => (top && acc < FLOOR ? floorY : y(acc))
  let k = 4
  const pos = rows.map((p) => ({ ...p, x: x(tw[k++]), y: yClamp(tw[k++]) }))
  const pos2 = extra.map((p) => ({ key: p.key, x: x(tw[k++]), y: yClamp(tw[k++]) }))
  const at = Object.fromEntries(pos.map((p) => [p.key, p]))

  const phi = at['phi4-14b']
  const margin = parseFloat(claims.margin)
  const bandHi = y(phi.acc + margin)
  const bandLo = y(phi.acc - margin)
  const yTicks = top ? [84, 85, 86, 87, 88, 89, 90, 91] : [40, 50, 60, 70, 80, 90]
  const xTicks = (mode === 'cost' ? COST_TICKS : SIZE_TICKS).filter((t) => !small || [0.1, 1, 10, 100, 0.5, 5, 50].includes(t))

  const frontPts = core.front.map((key) => at[key]).filter(Boolean)
  const frontPath = frontPts.map((p, i) => (i ? `H${p.x.toFixed(1)}V${p.y.toFixed(1)}` : `M${p.x.toFixed(1)},${p.y.toFixed(1)}`)).join('')
  const lastFront = frontPts[frontPts.length - 1]

  const keep = small ? ['phi4-14b', claims.topLarge, 'qwen3-4b', 'gpt-oss-120b'] : LABELLED
  const labelled = pos.filter((p) => keep.includes(p.key) && !(top && p.acc < FLOOR))
    .map((p) => ({ ...p, label: p.j.name, strong: p.key === 'phi4-14b' || p.key === claims.topLarge }))
  const labels = placeLabels(labelled.sort((a, b) => (b.strong ? 1 : 0) - (a.strong ? 1 : 0)), width, H, pad)
  const shownLabel = new Set(labels.map((l) => l.key))

  const onMove = (e) => {
    const r = e.currentTarget.getBoundingClientRect()
    const mx = ((e.clientX - r.left) / r.width) * width
    const my = ((e.clientY - r.top) / r.height) * H
    let best = null
    let bd = 26 * 26
    const cands = budgets ? [...pos, ...pos2.map((q) => ({ ...at[q.key], x: q.x, y: q.y }))] : pos
    for (const p of cands) {
      const dd = (p.x - mx) ** 2 + (p.y - my) ** 2
      if (dd < bd) { bd = dd; best = p }
    }
    if (!best) { hideTip(); setHovered(null); return }
    setHovered(best.key)
    const j = best.j
    showTip((best.x / width) * r.width, (best.y / H) * r.height, (
      <>
        <TipHead fam={j.family} name={j.name} open={best.budget !== QUICK} ring={j.kind === 'ladder'} />
        <TipRow k="Size" v={`${j.size}${j.active ? ` (${j.active}B active)` : ''}`} />
        <TipRow k={`Accuracy, B = ${count(best.budget)}`} v={`${fmt(best.acc)}%`} />
        <TipRow k="Est. cost per 1,000 verdicts" v={`$${fmt(best.cost)}`} />
        {best.other && <TipRow k={`At B = ${count(REASONED)}`} v={`${fmt(best.other.acc)}% · $${fmt(best.other.cost)}`} />}
        {onFront.has(best.key) && <div className="tt-note">On the cost–accuracy front</div>}
      </>
    ))
  }
  const leave = () => { hideTip(); setHovered(null) }
  const table = () => ({
    columns: [{ key: 'judge', label: 'Judge', align: 'left' }, { key: 'size', label: 'Size', align: 'left' }, { key: 'budget', label: 'Budget' }, { key: 'acc', label: 'Accuracy (%)' }, { key: 'cost', label: 'Cost per 1,000 verdicts ($)' }, { key: 'front', label: 'On the front', align: 'left' }],
    rows: [...rows].sort((a, b) => a.cost - b.cost).map((p) => ({ judge: p.j.name, size: p.j.size, budget: p.budget, acc: p.acc, cost: p.cost, front: onFront.has(p.key) ? 'yes' : '' })),
  })
  const anno = mode === 'cost' && top && !small && at[claims.topLarge]

  return (
    <section className="section cost" aria-labelledby="cost-title">
      <div className="wrap-wide">
        <div className="chart-frame" id="chart-cost" data-reveal>
          <div className="cf-head">
            <div>
              <h2 className="h2" id="cost-title">Accuracy against the cost of 1,000 verdicts</h2>
              <p className="dek">One point per judge. The stepped line is the cost–accuracy front.</p>
            </div>
            <Toolbar id="cost" title="Accuracy against the cost of 1,000 verdicts" sub="One point per judge. Estimated serving cost, US$." table={table} view={view} setView={setView} body={body} />
          </div>
          <div className="chart-bar">
            <Segmented label="Horizontal axis" value={mode} onChange={setMode} options={[{ value: 'cost', label: 'Cost' }, { value: 'size', label: 'Size' }]} />
            <button type="button" className="chip" aria-pressed={budgets} onClick={() => setBudgets((v) => !v)}>
              <svg viewBox="0 0 22 10" width="22" height="10" aria-hidden="true"><circle cx="4" cy="5" r="3.2" fill="currentColor" /><path d="M7 5h8" stroke="currentColor" strokeDasharray="2 2" /><circle cx="18" cy="5" r="3" fill="none" stroke="currentColor" /></svg>
              Show B = {count(REASONED)} too
            </button>
          </div>
          <div ref={body}>
          {view === 'table' && <DataTable {...table()} caption="Accuracy against the cost of 1,000 verdicts" />}
          <div className="chart" ref={(el) => { box.current = el; inRef.current = el }} style={view === 'table' ? { display: 'none' } : undefined}>
            <svg viewBox={`0 0 ${width} ${H}`} role="img" onPointerMove={onMove} onPointerDown={onMove} onPointerLeave={leave}
              aria-label={`Accuracy against estimated serving cost for ${rows.length} judges. Phi-4 reaches ${fmt(phi.acc)}% for $${fmt(phi.cost)} per 1,000 verdicts; ${byKey[claims.topLarge].name} ${fmt(at[claims.topLarge].acc)}% for $${fmt(at[claims.topLarge].cost)}.`}>
              {yTicks.map((t) => (
                <g key={t}>
                  <line className="grid-line" x1={pad.l} x2={width - pad.r} y1={y(t)} y2={y(t)} />
                  <text className="ax-text" x={pad.l - 10} y={y(t)} dy="0.32em" textAnchor="end">{t}%</text>
                </g>
              ))}
              {xTicks.map((t) => (
                <g key={t}>
                  <line className="grid-line" y1={pad.t} y2={plotBottom} x1={x(t)} x2={x(t)} />
                  <text className="ax-text" x={x(t)} y={H - (top ? 18 : 22)} textAnchor="middle">{mode === 'cost' ? money(t) : `${t}B`}</text>
                </g>
              ))}
              <text className="ax-title" x={width - pad.r} y={H - 2} textAnchor="end">
                {mode === 'cost' ? 'Estimated serving cost per 1,000 verdicts, US$ (log scale)' : 'Parameters (log scale)'}
              </text>
              {top && (
                <g>
                  <rect x={pad.l} y={floorY - 13} width={width - pad.l - pad.r} height={26} rx={6} className="floor" />
                  <text className="ax-text" x={pad.l - 10} y={floorY} dy="0.32em" textAnchor="end">{'<'}{FLOOR}</text>
                </g>
              )}
              <g className={`eq-band${seen ? ' in' : ''}`}>
                <rect x={pad.l} width={width - pad.l - pad.r} y={bandHi} height={Math.max(0, bandLo - bandHi)} className="band-eq" />
                <line className="band-eq-line" x1={pad.l} x2={width - pad.r} y1={bandHi} y2={bandHi} />
                <line className="band-eq-line" x1={pad.l} x2={width - pad.r} y1={bandLo} y2={bandLo} />
                {!small && <text className="lbl muted band-label" x={width - pad.r - 6} y={bandHi - 7} textAnchor="end">±{claims.margin} points of Phi-4</text>}
              </g>
              {mode === 'cost' && frontPts.length > 1 && (
                <g className={`pareto${seen ? ' in' : ''}`}>
                  <path d={frontPath} className="front-line" />
                  {!small && lastFront && <text className="lbl front-label" x={x(0.05)} y={y(top ? 86.6 : 70)}>cost–accuracy front</text>}
                </g>
              )}
              {anno && (
                <g className="anno">
                  <path d={`M${phi.x + 10},${phi.y - 30}H${at[claims.topLarge].x - 12}`} className="anno-line" markerEnd="url(#arrow)" />
                  <path d={`M${phi.x},${phi.y - 12}V${phi.y - 30}`} className="anno-line" />
                  <text className="lbl anno-text" x={(phi.x + at[claims.topLarge].x) / 2} y={phi.y - 38} textAnchor="middle">
                    a large judge’s quick verdict costs an estimated {claims.costLo} to {claims.costHi}× as much
                  </text>
                </g>
              )}
              <defs>
                <marker id="arrow" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7" markerHeight="7" orient="auto">
                  <path d="M0,0L8,4L0,8" fill="none" className="anno-head" />
                </marker>
              </defs>
              {budgets && pos2.map((q) => {
                const p = at[q.key]
                return (
                  <g key={`b-${q.key}`} className={`budget-pair${hovered && hovered !== q.key ? ' dim' : ''}`}>
                    {hovered === q.key && <line x1={p.x} y1={p.y} x2={q.x} y2={q.y} className="budget-line" style={{ stroke: famColor(p.j.family) }} />}
                    <g transform={`translate(${q.x},${q.y})`}><Mark fam={p.j.family} r={4.6} open /></g>
                  </g>
                )
              })}
              {pos.map((p) => {
                const hot = hovered === p.key
                return (
                  <g key={p.key} transform={`translate(${p.x},${p.y})`} className={`pt${hovered && !hot ? ' dim' : ''}`}>
                    {hot && <circle r={13} className="pt-ring" />}
                    <Mark fam={p.j.family} r={p.key === 'phi4-14b' || p.key === claims.topLarge ? 7.5 : 6} open={p.budget !== QUICK} ring={p.j.kind === 'ladder'} />
                  </g>
                )
              })}
              {labels.map((l) => (
                <text key={`l-${l.key}`} className={`lbl${l.strong ? ' strong' : ''}`} x={l.x} y={l.y} textAnchor={l.anchor}>{l.text}</text>
              ))}
              {hovered && at[hovered] && !shownLabel.has(hovered) && (
                <text className="lbl strong" x={at[hovered].x + 11} y={at[hovered].y - 10}>{at[hovered].j.name}</text>
              )}
            </svg>
            <Tooltip tip={tip} />
          </div>
          </div>
          <div className="chart-foot">
            <span className="legend-inline">
              {core.families.map((f) => (
                <span key={f.key}><svg viewBox="-7 -7 14 14" width="13" height="13" aria-hidden="true"><Mark fam={f.key} r={4.6} /></svg>{f.label}</span>
              ))}
              <span><svg viewBox="-7 -7 14 14" width="13" height="13" aria-hidden="true"><Mark fam="Qwen3" r={4.6} open /></svg>B = {count(REASONED)}</span>
            </span>
            <span className="source" title="An approximate decode rate for the judge's size, times its GPUs, at one price per GPU-hour">Cost is estimated from each judge’s approximate decode rate and GPU count at one price per GPU-hour. Judges that always reason are shown at B = {count(REASONED)}.</span>
          </div>
        </div>
      </div>
    </section>
  )
}
