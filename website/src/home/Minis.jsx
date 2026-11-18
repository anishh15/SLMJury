import { Mark } from '../components/ui'
import { byKey, claims, core, judges } from '../lib/data'

// Six small charts, one per finding, each drawn from core.mini.
const W = 260
const H = 92
const mini = core.mini
const lin = (d0, d1, r0, r1) => (v) => r0 + ((v - d0) / (d1 - d0)) * (r1 - r0)
const lg = (d0, d1, r0, r1) => (v) => r0 + ((Math.log(v) - Math.log(d0)) / (Math.log(d1) - Math.log(d0))) * (r1 - r0)
const famSlug = { Llama: 'llama', 'Qwen2.5': 'qwen25', Qwen3: 'qwen3', 'Phi-4': 'phi4', Large: 'large' }

function Frame({ children, label }) {
  return <svg className="mini" viewBox={`0 0 ${W} ${H}`} role="img" aria-label={label}>{children}</svg>
}

export function ScaleMini() {
  const x = lg(0.45, 160, 10, W - 12)
  const y = lin(74, 91.6, H - 10, 8)
  const lad = mini.ladder
  const phi = byKey['phi4-14b']
  const m = parseFloat(claims.margin)
  const large = judges.filter((j) => j.kind === 'large')
  return (
    <Frame label="Accuracy rises along the Qwen3 ladder, then flattens; Phi-4 sits level with the large judges">
      <rect x={6} width={W - 12} y={y(phi.acc10 + m)} height={y(phi.acc10 - m) - y(phi.acc10 + m)} className="band-eq" />
      <path className="draw mini-line" pathLength="1" d={lad.map((p, i) => `${i ? 'L' : 'M'}${x(p.params).toFixed(1)},${y(p.acc).toFixed(1)}`).join('')} style={{ stroke: 'var(--fam-qwen3)' }} />
      {lad.map((p, i) => <g key={p.key} className="pop" style={{ '--i': i }} transform={`translate(${x(p.params)},${y(p.acc)})`}><Mark fam="Qwen3" r={3.2} ring={p.key === 'qwen3-32b'} /></g>)}
      {large.map((j, i) => <g key={j.key} className="pop" style={{ '--i': i + 6 }} transform={`translate(${x(j.params) + (i % 2 ? 3 : -3)},${y(Math.max(74.5, j.best))})`}><Mark fam="Large" r={3.4} open={j.bestBudget !== 10} /></g>)}
      <g className="pop" style={{ '--i': 4 }} transform={`translate(${x(14)},${y(phi.acc10)})`}><Mark fam="Phi-4" r={4.6} /></g>
    </Frame>
  )
}

export function BudgetMini() {
  const all = [...mini.budgetOnly, ...mini.promptOnly]
  const x = lin(Math.min(...all) - 4, Math.max(...all) + 4, 74, W - 8)
  return (
    <Frame label="Raising only the budget leaves accuracy unchanged; changing only the prompt moves it by tens of points">
      <line x1={x(0)} x2={x(0)} y1={8} y2={H - 8} className="mini-zero" />
      <text x={0} y={30} className="mini-label">budget only</text>
      <text x={0} y={68} className="mini-label">prompt only</text>
      {mini.budgetOnly.map((v, i) => <circle key={`b${i}`} className="pop" style={{ '--i': i }} cx={x(v)} cy={26 + ((i % 3) - 1) * 4} r={3.2} fill="var(--navy-2)" />)}
      {mini.promptOnly.map((v, i) => <circle key={`p${i}`} className="pop" style={{ '--i': i + 4 }} cx={x(v)} cy={64 + ((i % 3) - 1) * 4} r={3.2} fill="var(--fam-llama)" />)}
    </Frame>
  )
}

export function DomainMini() {
  const y = lin(48, 100, H - 8, 8)
  const x0 = 70
  const x1 = W - 70
  return (
    <Frame label="Math and general accuracy for every judge, and the gap between them">
      <text x={x0 - 10} y={H - 2} textAnchor="end" className="mini-label">math</text>
      <text x={x1 + 10} y={H - 2} className="mini-label">general</text>
      <line x1={x0} x2={x0} y1={6} y2={H - 8} className="mini-zero" />
      <line x1={x1} x2={x1} y1={6} y2={H - 8} className="mini-zero" />
      {mini.domain.map((d, i) => {
        const j = byKey[d.key]
        return <line key={d.key} className="draw-fade" style={{ '--i': i, stroke: `var(--fam-${famSlug[j.family]})` }} x1={x0} x2={x1} y1={y(d.math)} y2={y(d.general)} strokeWidth="1.4" />
      })}
    </Frame>
  )
}

export function OpenMini() {
  const slms = [...new Set(mini.pairs.map((p) => p.slm))]
  const larges = [...new Set(mini.pairs.map((p) => p.large))]
  const cw = (W - 20) / larges.length
  const ch = (H - 8) / slms.length
  const fill = { slm: 'var(--navy-2)', tie: 'var(--tint-2)', large: 'var(--fam-large)' }
  return (
    <Frame label="Paired comparisons on SummEval: small judges higher, tied, or lower than each large judge">
      {mini.pairs.map((p, i) => (
        <rect key={i} className="pop" style={{ '--i': i % 12, fill: fill[p.v] }} x={10 + larges.indexOf(p.large) * cw + 2} y={4 + slms.indexOf(p.slm) * ch + 2} width={cw - 4} height={ch - 4} rx={3} />
      ))}
    </Frame>
  )
}

export function StrategyMini() {
  const s = mini.strategies
  const lo = 80
  const x = lin(lo, 91, 74, W - 8)
  const rowsY = [14, 37, 60, 83]
  const rows = [['singles', 'judges', 'var(--navy-2)'], ['personas', 'personas', 'var(--navy-2)'], ['juries', 'juries', 'var(--navy-2)'], ['debates', 'debate', 'var(--fam-llama)']]
  const best = Number(claims.bestSingle)
  return (
    <Frame label="Single judges, personas and juries end level; every debate falls below the best single judge">
      <line x1={x(best)} x2={x(best)} y1={4} y2={H - 2} className="mini-ref" />
      {rows.map(([k, label, c], r) => (
        <g key={k}>
          <text x={0} y={rowsY[r] + 4} className="mini-label">{label}</text>
          {s[k].map((v, i) => <circle key={i} className="pop" style={{ '--i': i % 10 }} cx={x(Math.max(lo, v))} cy={rowsY[r]} r={2.6} fill={c} opacity={v < lo ? 0.45 : 0.85} />)}
        </g>
      ))}
    </Frame>
  )
}

export function StudentsMini() {
  const v = mini.solver
  const bw = (W - 20) / v.length
  const y = lin(0, 100, H - 6, 6)
  return (
    <Frame label="Five students, ranked by their own solver accuracy">
      {v.map((a, i) => (
        <rect key={i} className="grow" style={{ '--i': i }} x={10 + i * bw + 6} width={bw - 12} y={y(a)} height={y(0) - y(a)} rx={3} fill="var(--navy-2)" opacity={1 - i * 0.12} />
      ))}
    </Frame>
  )
}

export const MINIS = { scale: ScaleMini, budget: BudgetMini, domain: DomainMini, 'open-ended': OpenMini, strategies: StrategyMini, students: StudentsMini }
