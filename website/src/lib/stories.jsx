import { TipHead, TipRow } from '../charts/kit'
import { DODGE } from '../charts/SizeScatter'
import { byKey, claims, core, facts, judges, name, points } from './data'
import { count, fmt, signed, word } from './format'

// The six findings as scroll stories. Each step is one number, one line and
// one state of a chart; consecutive states share mark ids, so the judges move
// between them. Every value comes from core.json and results.json.

const QUICK = facts.budgets[0]
const REASONED = facts.budgets[facts.budgets.length - 1]
const R = count(REASONED)
const margin = parseFloat(claims.margin)
const famOf = (k) => byKey[k]?.family
const pct = (v) => `${fmt(v)}%`
const money = (v) => (v >= 1 ? `$${fmt(v, 0)}` : `$${v >= 0.1 ? fmt(v, 1) : fmt(v, 2)}`)
const cap = (s) => s.charAt(0).toUpperCase() + s.slice(1)

const judgeTip = (j, rows) => () => (
  <><TipHead fam={j.family} name={j.name} ring={j.kind === 'ladder'} />{rows.map(([k, v]) => <TipRow key={k} k={k} v={v} />)}</>
)
const span = (vals, pad = 0.08) => {
  const lo = Math.min(...vals)
  const hi = Math.max(...vals)
  const p = (hi - lo) * pad || 1
  return [lo - p, hi + p]
}
const niceTicks = (lo, hi, step) => {
  const out = []
  for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) out.push(+v.toFixed(6))
  return out
}
const stepFor = (lo, hi, n = 5) => {
  const raw = (hi - lo) / n
  const mag = Math.pow(10, Math.floor(Math.log10(raw)))
  const k = raw / mag
  return (k >= 5 ? 10 : k >= 2 ? 5 : k >= 1 ? 2 : 1) * mag
}
const axis = (domain, opts = {}) => {
  const st = opts.step ?? stepFor(domain[0], domain[1], opts.n ?? 5)
  const nd = st >= 1 ? 0 : st >= 0.1 ? 1 : 2
  return { domain, ticks: niceTicks(domain[0], domain[1], st), format: opts.format ?? ((v) => fmt(v, nd)), title: opts.title }
}

// ── 1. Scale ────────────────────────────────────────────────────────────────
const SIZE_TICKS = [0.5, 1, 2, 5, 10, 20, 50, 100]
const COST_TICKS = [0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300]
const sizeTable = (marks) => ({
  columns: [{ key: 'judge', label: 'Judge', align: 'left' }, { key: 'size', label: 'Size', align: 'left' }, { key: 'acc', label: 'Accuracy (%)' }, { key: 'budget', label: 'Budget' }],
  rows: marks.map((mk) => { const j = byKey[mk.key]; return { judge: j.name, size: j.size, acc: mk.y, budget: j.bestBudget } }),
})

function scale() {
  const phi = byKey['phi4-14b']
  const top = byKey[claims.topLarge]
  const ladder = judges.filter((j) => j.family === 'Qwen3').sort((a, b) => a.params - b.params)
  const large = judges.filter((j) => j.kind === 'large')
  const hi = Math.ceil(Math.max(...judges.map((j) => j.best)) + 0.6)
  const allLo = Math.floor(Math.min(...judges.map((j) => j.best)) / 5) * 5 - 1
  const zoomLo = Math.floor(Math.min(phi.best - 3, ...large.map((j) => j.best)) - 1)
  const sizeMark = (j, extra = {}) => ({
    id: j.key, key: j.key, x: j.params, y: j.best, dx: DODGE[j.key] || 0, fam: j.family, open: j.bestBudget === REASONED,
    ring: j.kind === 'ladder', label: j.name, r: 6.5,
    tip: judgeTip(j, [['Size', `${j.size}${j.active ? ` (${j.active}B active)` : ''}`], [`Accuracy, B = ${count(j.bestBudget)}`, pct(j.best)]]),
    ...extra,
  })
  const xSize = { domain: [0.45, 170], log: true, ticks: SIZE_TICKS, format: (t) => `${t}B`, title: 'Parameters (log scale)' }
  const yAll = axis([allLo, hi], { step: 10, title: 'Accuracy (%)' })
  const yZoom = axis([zoomLo, hi], { step: hi - zoomLo > 12 ? 4 : 2, title: 'Accuracy (%)' })
  const band = { id: 'eq', y0: phi.best - margin, y1: phi.best + margin, className: 'st-band-eq', label: `±${claims.margin} points of Phi-4` }
  const bandRules = [
    { id: 'eq-hi', axis: 'y', v: phi.best + margin, className: 'st-rule-eq' },
    { id: 'eq-lo', axis: 'y', v: phi.best - margin, className: 'st-rule-eq' },
  ]
  const inZoom = judges.filter((j) => j.best >= zoomLo)
  const ds = 'deepseek-r1-distill-llama-70b'
  const lowLo = Math.floor(Math.min(...large.filter((j) => j.key !== ds).map((j) => j.best)) - 1.5)
  const yLow = axis([lowLo, hi + 0.4], { step: 1, title: 'Accuracy (%)' })
  const inLow = judges.filter((j) => j.best >= lowLo || j.key === ds)
  const cost = points.map((p) => ({ ...p, j: byKey[p.key] }))
  const g = claims.ladderGain
  return {
    slug: 'scale',
    steps: [
      {
        key: 'ladder',
        big: { v: <>+{g['qwen3-1.7b']} <span className="arrow">→</span> +{g['qwen3-32b']}</>, note: 'Qwen3’s gain from 0.6B to 1.7B, then from 14B to 32B' },
        spec: {
          key: 'ladder', title: 'Accuracy against judge size', sub: `Each judge at its better budget. The dashed line joins the Qwen3 family.`,
          alt: `Accuracy against size. Qwen3 gains ${g['qwen3-1.7b']} points from 0.6B to 1.7B and ${g['qwen3-32b']} from 14B to 32B.`,
          x: xSize, y: yAll,
          marks: judges.map((j) => sizeMark(j, { dim: j.family !== 'Qwen3', strong: ['qwen3-0.6b', 'qwen3-1.7b', 'qwen3-32b'].includes(j.key) })),
          paths: [{ id: 'ladder', ids: ladder.map((j) => j.key), className: 'st-path-ladder' }],
          table: sizeTable(judges.map((j) => ({ key: j.key, y: j.best }))),
        },
      },
      {
        key: 'top',
        big: { v: <>{claims.phiAcc10} <span className="approx">≈</span> {claims.topLargeAcc10}</>, u: '%', note: `Phi-4 and ${top.name} are equivalent within ±${claims.margin} points. The other ${claims.nBelowWord} large judges are significantly less accurate.` },
        spec: {
          key: 'top', title: 'The top of the field', sub: `Shaded: ±${claims.margin} points around Phi-4. Hexagons: large judges.`,
          alt: `Phi-4 at ${claims.phiAcc10}% and ${top.name} at ${claims.topLargeAcc10}% sit inside the ±${claims.margin}-point band; the other large judges sit below Phi-4.`,
          x: xSize, y: yLow,
          marks: inLow.map((j) => sizeMark(j, {
            dim: j.kind !== 'large' && j.key !== 'phi4-14b', strong: j.kind === 'large' || j.key === 'phi4-14b', force: j.key === 'phi4-14b' || j.key === ds || j.key === claims.topLarge,
            label: j.key === ds ? `${j.name} · ${fmt(j.best)}% ↓` : j.name,
          })),
          areas: [band], rules: bandRules,
          table: sizeTable(inLow.map((j) => ({ key: j.key, y: j.best }))),
        },
      },
      {
        key: 'cost',
        big: { v: <>{claims.costLo}–{claims.costHi}<small>×</small></>, note: 'how much more a large judge’s quick verdict costs to serve than Phi-4 (estimated)' },
        spec: {
          key: 'cost', title: 'Accuracy against serving cost', sub: 'Stepped line: the cost–accuracy front.',
          alt: `Phi-4 serves 1,000 verdicts for $${claims.phiCost}; large judges cost ${claims.costLo} to ${claims.costHi} times more.`,
          x: { domain: [0.016, 700], log: true, ticks: COST_TICKS, format: money, title: 'Estimated $ per 1,000 verdicts (log scale)' },
          y: yZoom,
          marks: cost.filter((p) => p.acc >= zoomLo).map((p) => ({
            ...sizeMark(p.j), x: p.costExact, y: p.acc, dx: 0, open: p.budget !== QUICK,
            strong: ['phi4-14b', claims.topLarge, 'qwen2.5-72b', 'llama3.3-70b'].includes(p.key), force: ['phi4-14b', claims.topLarge].includes(p.key),
            tip: judgeTip(p.j, [[`Accuracy, B = ${count(p.budget)}`, pct(p.acc)], ['Cost per 1,000 verdicts', `$${fmt(p.cost)}`]]),
          })),
          paths: [{ id: 'front', ids: core.front.filter((k) => cost.find((p) => p.key === k && p.acc >= zoomLo)), step: 'hv', className: 'st-path-front' }],
          areas: [band], rules: bandRules,
          table: {
            columns: [{ key: 'judge', label: 'Judge', align: 'left' }, { key: 'budget', label: 'Budget' }, { key: 'acc', label: 'Accuracy (%)' }, { key: 'cost', label: 'Cost per 1,000 verdicts ($)' }],
            rows: [...cost].sort((a, b) => a.cost - b.cost).map((p) => ({ judge: p.j.name, budget: p.budget, acc: p.acc, cost: p.cost })),
          },
        },
      },
    ],
  }
}

// ── 2. Budget ───────────────────────────────────────────────────────────────
function budget(res) {
  const ot = res.overthink
  const ax = Math.ceil(Math.max(...ot.flatMap((o) => [Math.abs(o.math), Math.abs(o.general)])) * 1.1)
  const sq = { domain: [-ax, ax], ticks: niceTicks(-ax, ax, stepFor(-ax, ax, 4)), format: (v) => signed(v, 0) }
  const otMark = (o, extra = {}) => {
    const j = byKey[o.key]
    return {
      id: o.key, key: o.key, x: o.math, y: o.general, fam: j.family, label: j.name, r: 6,
      tip: judgeTip(j, [['Math, quick − reasoned', `${signed(o.math)} points`], ['General, quick − reasoned', `${signed(o.general)} points`]]),
      ...extra,
    }
  }
  const otTable = {
    columns: [{ key: 'judge', label: 'Judge', align: 'left' }, { key: 'math', label: 'Math, quick − reasoned (points)' }, { key: 'general', label: 'General, quick − reasoned (points)' }],
    rows: ot.map((o) => ({ judge: name(o.key), math: o.math, general: o.general })),
  }
  const sf = res.singleFactor.judges
  const keys = Object.keys(sf)
  const bud = keys.filter((k) => sf[k].factor_budget?.delta != null)
  const pro = keys.filter((k) => sf[k].factor_prompt?.delta != null)
  const bAbs = Math.max(...bud.map((k) => Math.abs(sf[k].factor_budget.delta)), 0.5) * 2.2
  const all = [...bud.map((k) => sf[k].factor_budget.delta), ...pro.map((k) => sf[k].factor_prompt.delta)]
  const rows = [{ key: 'budget', label: 'Budget only', band: true }, { key: 'prompt', label: 'Prompt only' }]
  const sfMark = (k, cond, extra = {}) => {
    const j = byKey[k]
    const x = sf[k][cond]
    return {
      id: cond === 'factor_budget' ? k : `${k}:p`, key: k, x: x.delta, y: cond === 'factor_budget' ? 'budget' : 'prompt',
      fam: j.family, r: 5.5, label: j.name,
      tip: judgeTip(j, [[`Change from B = ${QUICK}`, `${signed(x.delta)} points`], ['Accuracy', pct(x.acc)], ['Parseable verdicts', pct(x.ifr)]]),
      ...extra,
    }
  }
  const jitter = (k, i) => ((i % 7) - 3) * 7
  const sfTable = {
    columns: [{ key: 'judge', label: 'Judge', align: 'left' }, { key: 'budget', label: 'Budget only (points)' }, { key: 'prompt', label: 'Prompt only (points)' }],
    rows: keys.map((k) => ({ judge: name(k), budget: sf[k].factor_budget?.delta, prompt: sf[k].factor_prompt?.delta })),
  }
  const full = span(all, 0.06)
  return {
    slug: 'budget',
    steps: [
      {
        key: 'math',
        big: { v: <>{claims.quickMath}<small> of {claims.nDual}</small></>, note: `judges score higher on math with the quick verdict, while reasoning wins on general tasks for ${claims.reasonGeneral} of ${claims.nDual}` },
        spec: {
          key: 'math', title: 'Quick minus reasoned accuracy', sub: 'Per judge, in points.',
          alt: `Quick minus reasoned accuracy on math against general tasks for ${claims.nDual} judges.`,
          x: { ...sq, title: 'Math (points)' }, y: { ...sq, title: 'General (points)' },
          marks: ot.map((o) => otMark(o, { strong: o.key === 'phi4-14b' || o.key === 'qwen3-4b' })),
          areas: [{ id: 'q-right', x0: 0, className: 'st-area-good', label: 'quick wins on math', labelAnchor: 'end' }, { id: 'q-below', y1: 0, className: 'st-area-warm', label: 'reasoning wins on general tasks', labelInside: true }],
          rules: [{ id: 'x0', axis: 'x', v: 0 }, { id: 'y0', axis: 'y', v: 0 }],
          table: otTable,
        },
      },
      {
        key: 'budget-only',
        big: { v: <>{claims.largestBudgetMove}</>, u: 'points at most', note: `Raising the budget alone leaves accuracy unchanged for ${claims.unchanged} of ${claims.nBudget} judges.` },
        spec: {
          key: 'budget-only', title: 'Change one factor at a time', sub: `Change from B = ${QUICK}.`,
          alt: `Raising only the budget leaves ${claims.unchanged} of ${claims.nBudget} judges unchanged.`,
          x: axis([-bAbs, bAbs], { title: 'Change in accuracy (points)', format: (v) => signed(v, Math.abs(bAbs) < 2 ? 1 : 0) }),
          y: { rows }, labelW: 120, labelWNarrow: 92,
          marks: bud.map((k, i) => sfMark(k, 'factor_budget', { dy: jitter(k, i) })),
          rules: [{ id: 'x0', axis: 'x', v: 0 }],
          table: sfTable,
        },
      },
      {
        key: 'prompt-only',
        big: { v: <>r = {claims.rIfr}</>, note: 'Changing only the prompt moves accuracy by tens of points, tracking the share of parseable verdicts.' },
        spec: {
          key: 'prompt-only', title: 'Change one factor at a time', sub: `Change from B = ${QUICK}.`,
          alt: 'The prompt alone moves accuracy by tens of points while the budget alone barely moves it.',
          x: axis(full, { title: 'Change in accuracy (points)', format: (v) => signed(v, 0) }),
          y: { rows }, labelW: 120, labelWNarrow: 92,
          marks: [...bud.map((k, i) => sfMark(k, 'factor_budget', { dy: jitter(k, i) })), ...pro.map((k, i) => sfMark(k, 'factor_prompt', { dy: jitter(k, i) }))],
          rules: [{ id: 'x0', axis: 'x', v: 0 }],
          table: sfTable,
        },
      },
    ],
  }
}

// ── 3. Domain ───────────────────────────────────────────────────────────────
function domain(res) {
  const HS = res.datasetKeys.indexOf('hellaswag')
  // the quick verdict, as in the paper's domain table; B = 8,192 for judges that always reason
  const qb = (j) => (j.acc10 != null ? QUICK : REASONED)
  const rowsJ = judges.map((j) => ({ j, d: res.domainMeans[j.key]?.[String(qb(j))] })).filter((x) => x.d)
    .sort((a, b) => b.d.gap - a.d.gap)
  const rows = rowsJ.map(({ j }) => ({ key: j.key, label: j.name }))
  const strong3 = ['phi4-14b', 'qwen3-8b', 'qwen3-14b']
  const largeNoDs = judges.filter((j) => j.kind === 'large' && j.key !== 'deepseek-r1-distill-llama-70b').map((j) => j.key)
  const vals = rowsJ.flatMap(({ d }) => [d.math, d.general])
  const dom = axis(span([...vals, 100], 0.03), { step: 10, title: 'Accuracy (%)' })
  const tip = (j, d) => judgeTip(j, [['Math', pct(d.math)], ['General', pct(d.general)], ['Gap', `${fmt(d.gap, 1)} points`], ['Budget', `B = ${count(qb(j))}`]])
  const pair = (focus, label) => rowsJ.flatMap(({ j, d }) => {
    const on = focus(j.key)
    return [
      { id: `${j.key}:m`, key: j.key, x: d.math, y: j.key, fam: j.family, r: 4.6, dim: !on, tip: tip(j, d), label: label && on ? `${fmt(d.gap, 1)}` : undefined, showLabel: label && on, labelSide: 'r' },
      { id: `${j.key}:g`, key: j.key, x: d.general, y: j.key, fam: j.family, open: true, r: 4.4, dim: !on, tip: tip(j, d) },
    ]
  })
  const paths = rowsJ.map(({ j }) => ({ id: `gap-${j.key}`, ids: [`${j.key}:g`, `${j.key}:m`], className: 'st-path-gap' }))
  const table = {
    columns: [{ key: 'judge', label: 'Judge', align: 'left' }, { key: 'budget', label: 'Budget' }, { key: 'math', label: 'Math (%)' }, { key: 'science', label: 'Science (%)' }, { key: 'general', label: 'General (%)' }, { key: 'gap', label: 'Gap (points)' }],
    rows: rowsJ.map(({ j, d }) => ({ judge: j.name, budget: qb(j), math: d.math, science: d.science, general: d.general, gap: d.gap })),
  }
  const base = { title: 'Math against general accuracy', sub: 'Filled: math. Open: general. Quick verdicts.', openNote: true, x: dom, y: { rows }, labelW: 168, labelWNarrow: 112, height: 520, heightNarrow: 470, paths, table }
  const hs = rowsJ.map(({ j }) => ({ j, v: res.perDataset[j.key]?.[String(qb(j))]?.[HS] })).filter((x) => x.v != null)
  return {
    slug: 'domain',
    steps: [
      {
        key: 'widest',
        big: { v: <>{claims.gapHi}</>, u: 'points', note: `${name(claims.gapHiJudge)}’s math-to-general gap (${claims.gapHiMath}% on math, ${claims.gapHiGeneral}% on general reasoning)` },
        spec: { ...base, key: 'widest', alt: `The math-to-general gap reaches ${claims.gapHi} points.`, marks: pair((k) => k === claims.gapHiJudge, true) },
      },
      {
        key: 'strong',
        big: { v: <>{claims.strong3Gap.join('–')}</>, u: 'points', note: `the gap for Phi-4, Qwen3-8B and Qwen3-14B, and ${claims.largeGap1.join(' to ')} for the large judges other than DeepSeek-R1-Distill` },
        spec: { ...base, key: 'strong', alt: `The strongest small judges have a gap of ${claims.strong3Gap.join('–')} points, the large judges ${claims.largeGap1.join('–')}.`, marks: pair((k) => strong3.includes(k) || largeNoDs.includes(k), true) },
      },
      {
        key: 'hellaswag',
        big: { v: <>&lt;{claims.hsCeil}<small>%</small></>, note: 'No judge at any scale clears this on HellaSwag.' },
        spec: {
          ...base, key: 'hellaswag', title: 'Accuracy on HellaSwag', sub: 'Quick verdicts.', paths: [],
          alt: `No judge clears ${claims.hsCeil}% on HellaSwag.`,
          marks: hs.map(({ j, v }) => ({ id: `${j.key}:m`, key: j.key, x: v, y: j.key, fam: j.family, r: 4.8, tip: judgeTip(j, [['HellaSwag', pct(v)]]) })),
          rules: [{ id: 'hs', axis: 'x', v: Number(claims.hsCeil), className: 'st-rule-ref', label: `${claims.hsCeil}%` }],
          table: { columns: [{ key: 'judge', label: 'Judge', align: 'left' }, { key: 'hs', label: 'HellaSwag (%)' }], rows: hs.map(({ j, v }) => ({ judge: j.name, hs: v })) },
        },
      },
    ],
  }
}

// ── 4. Open-ended ───────────────────────────────────────────────────────────
const VERDICT_FILL = { slm: 'var(--accent)', tie: 'var(--tint-2)', large: 'var(--fam-large)' }
const VERDICT_INK = { slm: '#fff', tie: 'var(--ink)', large: '#fff' }
const VERDICT_LABEL = { slm: 'small judge higher', tie: 'not distinguishable', large: 'large judge higher' }

function openEnded(res) {
  const pairs = res.summevalPairs
  const slms = [...new Set(pairs.map((p) => p.slm))]
  const larges = [...new Set(pairs.map((p) => p.large))]
  const grid = pairs.map((p) => ({
    id: `pair:${p.slm}:${p.large}`, x: larges.indexOf(p.large), y: p.slm, shape: 'cell', cw: 0.92, ch: 0.84,
    color: VERDICT_FILL[p.verdict], ink: VERDICT_INK[p.verdict], text: signed(p.diff, 3),
    tip: () => <><div className="tt-head">{name(p.slm)} vs {name(p.large)}</div><TipRow k="Difference in ρ" v={signed(p.diff, 3)} /><TipRow k="95% interval" v={`${fmt(p.lo, 3)} to ${fmt(p.hi, 3)}`} /><TipRow k="Result" v={VERDICT_LABEL[p.verdict]} /></>,
  }))
  const scatter = (metric, strongKeys) => {
    const pts = judges.filter((j) => j[metric] != null)
    const ys = pts.map((j) => j[metric])
    return {
      x: axis(span(pts.map((j) => j.best), 0.05), { step: 5, title: 'Accuracy, better budget (%)' }),
      y: axis(span(ys, 0.1), { step: 0.1, title: metric === 'se' ? 'SummEval ρ' : 'MT-Bench ρ', format: (v) => fmt(v, 1) }),
      marks: pts.map((j) => ({
        id: j.key, key: j.key, x: j.best, y: j[metric], fam: j.family, open: j.bestBudget === REASONED, ring: j.kind === 'ladder', label: j.name, r: 6,
        strong: strongKeys.includes(j.key), dim: !strongKeys.includes(j.key) && j.kind !== 'large',
        tip: judgeTip(j, [['Accuracy, better budget', pct(j.best)], ['SummEval ρ', fmt(j.se, 3)], ['MT-Bench ρ', fmt(j.mt, 3)]]),
      })),
      table: {
        columns: [{ key: 'judge', label: 'Judge', align: 'left' }, { key: 'acc', label: 'Accuracy (%)' }, { key: 'se', label: 'SummEval ρ' }, { key: 'mt', label: 'MT-Bench ρ' }],
        rows: pts.map((j) => ({ judge: j.name, acc: j.best, se: j.se, mt: j.mt })),
      },
    }
  }
  const counts = pairs.reduce((a, p) => ({ ...a, [p.verdict]: (a[p.verdict] || 0) + 1 }), {})
  return {
    slug: 'open-ended',
    steps: [
      {
        key: 'pairs',
        big: { v: <>{claims.seMatch}<small> of {claims.seN}</small></>, note: 'paired comparisons on SummEval where a top small judge matches or beats a large one' },
        spec: {
          key: 'pairs', title: 'Small against large on SummEval', sub: 'Difference in Spearman’s ρ, small minus large.',
          alt: `${claims.seMatch} of ${claims.seN} paired comparisons are matched or won by the small judge.`,
          x: { domain: [-0.5, larges.length - 0.5], ticks: larges.map((k, i) => ({ v: i, label: name(k).split('-').slice(0, 2).join('-') })), grid: false, rotate: true, cat: true },
          y: { rows: slms.map((k) => ({ key: k, label: name(k) })) }, labelW: 130, labelWNarrow: 96,
          marks: grid,
          notes: [],
          legend: ['slm', 'tie', 'large'].map((v) => ({ color: VERDICT_FILL[v], label: `${VERDICT_LABEL[v]} (${counts[v] || 0})` })),
          table: {
            columns: [{ key: 'slm', label: 'Small judge', align: 'left' }, { key: 'large', label: 'Large judge', align: 'left' }, { key: 'diff', label: 'Difference in ρ' }, { key: 'lo', label: 'Lower' }, { key: 'hi', label: 'Upper' }, { key: 'v', label: 'Result', align: 'left' }],
            rows: pairs.map((p) => ({ slm: name(p.slm), large: name(p.large), diff: p.diff, lo: p.lo, hi: p.hi, v: VERDICT_LABEL[p.verdict] })),
          },
        },
      },
      {
        key: 'summeval',
        big: { v: <>ρ = {claims.phiSe}</>, note: `Phi-4 on SummEval, significantly above ${word(claims.phiSeAbove)} of the ${word(facts.nLarge)} large judges` },
        spec: { key: 'summeval', title: 'Accuracy against SummEval agreement', alt: `Phi-4 reaches ρ = ${claims.phiSe} on SummEval.`, ...scatter('se', ['phi4-14b']) },
      },
      {
        key: 'mtbench',
        big: { v: <>{cap(claims.phiMtRank)}</>, note: `Phi-4 among small judges on MT-Bench. Phi-4-Reasoning is ${claims.phirMtRank}.` },
        spec: { key: 'mtbench', title: 'Accuracy against MT-Bench agreement', alt: `On MT-Bench Phi-4 ranks ${claims.phiMtRank} among small judges.`, ...scatter('mt', ['phi4-14b', 'phi4r-14b']) },
      },
    ],
  }
}

// ── 5. Strategies ───────────────────────────────────────────────────────────
function strategies(res) {
  const small = judges.filter((j) => j.kind === 'small')
  const st = res.strategies
  const members = (ms) => ms.map((m) => name(m.key)).join(' + ')
  const pers = (k) => {
    const b = String(byKey[k].bestBudget)
    const x = res.personas[k]?.[b]
    return x ? { b, x } : null
  }
  const rowsAll = [
    { key: 'single', label: 'Single judges', band: true },
    { key: 'jury', label: 'Majority-vote juries' },
    { key: 'debate', label: 'Multi-agent debate', band: true },
    { key: 'p-phi4-14b', label: 'Phi-4, personas' },
    { key: 'p-qwen3-14b', label: 'Qwen3-14B, personas', band: true },
  ]
  const jit = (i, n = 3) => ((i % n) - (n - 1) / 2) * 5
  const singles = small.map((j, i) => ({
    id: `s:${j.key}`, key: j.key, x: j.best, y: 'single', fam: j.family, open: j.bestBudget === REASONED, r: 5, dy: jit(i),
    tip: judgeTip(j, [[`Accuracy, B = ${count(j.bestBudget)}`, pct(j.best)]]),
  }))
  const juries = st.juries.map((x, i) => ({
    id: `j:${i}`, x: x.acc, y: 'jury', shape: 'dot', color: 'var(--accent-2)', r: 4.6, dy: jit(i),
    tip: () => <><div className="tt-head">Majority-vote jury of three</div><div className="tt-sub">{members(x.members)}</div><TipRow k="Accuracy" v={pct(x.acc)} /></>,
  }))
  const debates = st.debates.map((x, i) => ({
    id: `d:${i}`, x: x.acc, y: 'debate', shape: 'dot', color: 'var(--v-incorrect)', r: 4.6, dy: jit(i),
    tip: () => <><div className="tt-head">{x.variant === 'same' ? 'Same-model debate' : 'Cross-architecture debate'}</div><div className="tt-sub">{members(x.members)}</div><TipRow k={`B = ${count(x.budget)}`} v={pct(x.acc)} /></>,
  }))
  const personaDots = ['phi4-14b', 'qwen3-14b'].flatMap((k) => {
    const p = pers(k)
    if (!p) return []
    const j = byKey[k]
    return [
      { id: `p:${k}:base`, key: k, x: p.x.base, y: `p-${k}`, fam: j.family, open: true, r: 6, tip: judgeTip(j, [['No persona', pct(p.x.base)]]) },
      ...facts.personas.map((q, i) => ({ id: `p:${k}:${q}`, key: k, x: p.x[q], y: `p-${k}`, fam: j.family, r: 4, dy: jit(i, 2) * 0.8, tip: judgeTip(j, [[q, pct(p.x[q])], ['No persona', pct(p.x.base)]]) })),
    ]
  })
  const all = [...singles, ...juries, ...debates, ...personaDots].map((m) => m.x)
  const x = axis(span(all, 0.03), { step: 5, title: 'Accuracy (%)' })
  const best = Number(claims.bestSingle)
  // juries and personas sit within a point of the best single judge, so those steps zoom in
  const zoom = (vals) => {
    const lo = Math.floor((Math.min(...vals) - 0.4) * 2) / 2
    const hi = Math.ceil((Math.max(...vals) + 0.3) * 2) / 2
    return { ...axis([lo, hi], { step: 0.5, title: 'Accuracy (%)', format: (v) => fmt(v, 1) }), lo }
  }
  const xJury = zoom([...juries.map((m) => m.x), best])
  const xPers = zoom([...personaDots.map((m) => m.x), best])
  const within = (ms, ax) => ms.filter((m) => m.x >= ax.lo)
  const rule = { id: 'best', axis: 'x', v: best, className: 'st-rule-ref', label: `best single judge, ${claims.bestSingle}%` }
  const table = {
    columns: [{ key: 'kind', label: 'Configuration', align: 'left' }, { key: 'who', label: 'Judges', align: 'left' }, { key: 'acc', label: 'Accuracy (%)' }],
    rows: [
      ...small.map((j) => ({ kind: 'Single judge', who: j.name, acc: j.best })),
      ...st.juries.map((x2) => ({ kind: 'Majority-vote jury of three', who: members(x2.members), acc: x2.acc })),
      ...st.debates.map((x2) => ({ kind: x2.variant === 'same' ? 'Same-model debate' : 'Cross-architecture debate', who: members(x2.members), acc: x2.acc })),
    ],
  }
  const base = { title: 'Every strategy, side by side', sub: 'Accuracy of each configuration (%). Hover a dot for its members.', x, y: { rows: rowsAll }, labelW: 150, labelWNarrow: 100, rules: [rule], table, heightNarrow: 380 }
  return {
    slug: 'strategies',
    steps: [
      {
        key: 'jury',
        big: { v: <>+{claims.juryGain}</>, u: 'points', note: `the best of ${facts.nJuries} majority-vote juries over the best single small judge` },
        spec: { ...base, key: 'jury', alt: `The best majority-vote jury gains ${claims.juryGain} points.`, x: xJury, sub: 'Zoomed in on the top. Hover a dot for its members.', marks: [...within(singles, xJury).map((m) => ({ ...m, dim: true })), ...juries] },
      },
      {
        key: 'debate',
        big: { v: <>−{claims.debateGap}</>, u: 'points', note: 'Multi-agent debate degrades accuracy in every configuration tested. Shown is the best debate against Qwen3-4B alone.' },
        spec: { ...base, key: 'debate', alt: `Every multi-agent debate falls below the best single judge.`, marks: [...singles.map((m) => ({ ...m, dim: true })), ...juries.map((m) => ({ ...m, dim: true })), ...debates] },
      },
      {
        key: 'personas',
        big: { v: <>{claims.personaRange['qwen3-14b']} · {claims.personaRange['phi4-14b']}</>, u: 'points', note: `Qwen3-14B and Phi-4 stay within a point across ${word(facts.nPersonas)} adversarial personas` },
        spec: { ...base, key: 'personas', alt: `Qwen3-14B and Phi-4 stay within a point across six personas.`, x: xPers, sub: 'Open marks: no persona.', openNote: true, marks: [...within(singles, xPers).map((m) => ({ ...m, dim: true })), ...within(juries, xPers).map((m) => ({ ...m, dim: true })), ...personaDots] },
      },
    ],
  }
}

// ── 6. Students ─────────────────────────────────────────────────────────────
function students(res) {
  const s = res.students
  const solver = [...s.solver].sort((a, b) => b.acc - a.acc)
  const fams = Object.keys(s.lineage)
  const solverStep = {
    x: axis([0, 100], { step: 20, title: 'Solver accuracy (%)' }),
    y: { rows: solver.map((x, i) => ({ key: x.key, label: x.weights.split('/')[1].replace(/-Instruct.*$/, '').replace(/-GPTQ.*$/, ''), band: i % 2 === 0 })) },
    labelW: 190, labelWNarrow: 120,
    marks: solver.map((x) => ({
      id: `st:${x.key}`, x: x.acc, y: x.key, shape: 'dot', r: 6.5, open: x.origin === 'probe',
      color: x.origin === 'probe' ? 'var(--fam-qwen3)' : 'var(--accent-2)', label: `${fmt(x.acc, 1)}%`, showLabel: true, labelSide: 'r',
      tip: () => <><div className="tt-head">{x.weights}</div><TipRow k="Solver accuracy" v={`${fmt(x.acc, 1)}%`} /><TipRow k="Family" v={x.family} /></>,
    })),
    table: { columns: [{ key: 'student', label: 'Student', align: 'left' }, { key: 'family', label: 'Family', align: 'left' }, { key: 'acc', label: 'Solver accuracy (%)' }], rows: solver.map((x) => ({ student: x.weights, family: x.family, acc: x.acc })) },
  }
  const lin = s.lineage
  const vals = fams.flatMap((f) => Object.values(lin[f]))
  const lim = Math.max(...vals.map(Math.abs)) * 1.25
  const linStep = (b) => ({
    x: axis([-lim, lim], { title: 'Lineage effect on the false-accept rate', format: (v) => signed(v, 2) }),
    y: { rows: fams.map((f, i) => ({ key: f, label: f, band: i % 2 === 0 })) },
    labelW: 110, labelWNarrow: 84,
    marks: fams.map((f) => ({
      id: `lin:${f}`, x: lin[f][b], y: f, fam: f, r: 8, open: b === String(REASONED), label: signed(lin[f][b], b === String(QUICK) ? 3 : 2), showLabel: true, labelSide: lin[f][b] < 0 ? 'l' : 'r',
      tip: () => <><TipHead fam={f} name={f} /><TipRow k={`B = ${count(Number(b))}`} v={signed(lin[f][b], 3)} /></>,
    })),
    rules: [{ id: 'x0', axis: 'x', v: 0 }],
    table: { columns: [{ key: 'fam', label: 'Family', align: 'left' }, { key: 'b10', label: `B = ${QUICK}` }, { key: 'b8192', label: `B = ${R}` }], rows: fams.map((f) => ({ fam: f, b10: lin[f][String(QUICK)], b8192: lin[f][String(REASONED)] })) },
  })
  const l10 = fams.map((f) => lin[f][String(QUICK)])
  return {
    slug: 'students',
    steps: [
      {
        key: 'solver',
        big: { v: <>τ = {claims.studentsTauSolver}</>, note: `between how capable judges rank ${word(claims.nStudentsWide)} students and the students’ own accuracy` },
        spec: { key: 'solver', title: 'The students', sub: 'Outlined: a strong Qwen3-family probe student.', alt: `Students span ${claims.solverRange.join('–')}% solver accuracy.`, ...solverStep },
      },
      {
        key: 'quick',
        big: { v: <>{signed(Math.min(...l10), 3)} to {signed(Math.max(...l10), 3)}</>, note: `at B = ${QUICK}, no family favors its own students` },
        spec: { key: 'quick', title: `Lineage effect at B = ${QUICK}`, sub: 'Positive: a family favors its own student.', alt: 'At the quick verdict the lineage effects are close to zero.', ...linStep(String(QUICK)) },
      },
      {
        key: 'reasoned',
        big: { v: <>{claims.lineage['8192'].Qwen3}</>, note: `Qwen3’s lineage effect at B = ${R}. A preference appears only once judges reason.` },
        spec: { key: 'reasoned', title: `Lineage effect at B = ${R}`, sub: 'Positive: a family favors its own student.', alt: `At B = ${R} Qwen3 shows a lineage effect of ${claims.lineage['8192'].Qwen3}.`, ...linStep(String(REASONED)) },
      },
    ],
  }
}

export function buildStories(res) {
  return [scale(), budget(res), domain(res), openEnded(res), strategies(res), students(res)]
}
