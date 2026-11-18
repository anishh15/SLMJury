import { useState } from 'react'
import ChartFrame from '../charts/Frame'
import SizeScatter from '../charts/SizeScatter'
import { BarList, DotPlot, Heatmap, LineChart, LogBars, Scatter } from '../charts/generic'
import { BudgetLegend, FamilyLegend, TipHead, TipRow } from '../charts/kit'
import { MarkIcon, Segmented } from '../components/ui'
import { byKey, claims, core, facts, FAMILY_SLUG, judges, name, useResults } from '../lib/data'
import { count, fmt, signed, word } from '../lib/format'

// The full numbers behind each finding: the paper's wording, and every
// detailed chart. Loaded the first time a reader opens Details.

const [QUICK, REASONED] = [facts.budgets[0], facts.budgets[facts.budgets.length - 1]]
const R = count(REASONED)
const DS_LABEL = Object.fromEntries(core.datasets.closed.map((d) => [d.key, d.name]))
const famOf = (k) => byKey[k]?.family
const small = judges.filter((j) => j.kind === 'small')
const J1 = { key: 'judge', label: 'Judge', align: 'left' }

const span = (vals, pad = 0.08, floor = null) => {
  const lo = Math.min(...vals)
  const hi = Math.max(...vals)
  const p = (hi - lo) * pad || 1
  return [floor ?? lo - p, hi + p]
}

const budgetOpts = (withBest = true) => [
  ...(withBest ? [{ value: 'best', label: 'Better budget' }] : []),
  { value: QUICK, label: `B = ${QUICK}` },
  { value: REASONED, label: `B = ${R}` },
]

function Prose({ children }) {
  return <div className="prose details-prose">{children}</div>
}

// ── 1. Scale ────────────────────────────────────────────────────────────────
function Scale({ res }) {
  const [mode, setMode] = useState('best')
  const [zoom, setZoom] = useState('all')
  const [cb, setCb] = useState(QUICK)
  const cq = Object.fromEntries(core.costQuick.map((c) => [c.key, c]))
  const costRows = res.cost.filter((c) => c.budget === cb).sort((a, b) => a.cost - b.cost).map((c) => {
    const j = byKey[c.key]
    return {
      key: c.key, label: j.name, fam: j.family, slug: FAMILY_SLUG[j.family], open: cb === REASONED, v: c.cost,
      strong: c.key === 'phi4-14b' || j.kind === 'large', right: `$${fmt(c.cost)}`, right2: `${fmt(c.acc)}%`,
      tip: <><TipHead fam={j.family} name={j.name} open={cb === REASONED} /><TipRow k="Cost per 1,000 verdicts" v={`$${fmt(c.cost)}`} /><TipRow k="Accuracy" v={`${fmt(c.acc)}%`} /></>,
    }
  })
  const costs = costRows.map((r) => r.v)
  const lo = Math.pow(10, Math.floor(Math.log10(Math.min(...costs))))
  const hi = Math.pow(10, Math.ceil(Math.log10(Math.max(...costs))))
  const valueAt = (j) => (mode === 'best' ? j.best : mode === QUICK ? j.acc10 : j.acc8192)
  return (
    <>
      <Prose>
        <p>
          Phi-4 reaches {claims.phiAcc10}% at B = {QUICK} and {claims.topLargeName} {claims.topLargeAcc10}%. The two are
          equivalent within ±{claims.margin} points, and the other {claims.nBelowWord} large judges are significantly less
          accurate than Phi-4. All {word(facts.nLarge)} large judges ran through the same pipeline, prompts, students and
          oracle as the small ones, and every comparison is paired on the same items. Within one family the curve flattens:
          Qwen3 gains {claims.ladderGain['qwen3-1.7b']} points from 0.6B to 1.7B, then {claims.ladderGain['qwen3-14b']} from
          8B to 14B and {claims.ladderGain['qwen3-32b']} from 14B to 32B ({claims.ladderTop}%).
        </p>
        <p>
          A quick verdict from a {facts.largeRange.join('–')} judge costs {claims.costLo} to {claims.costHi} times more.
          Phi-4 serves 1,000 verdicts for ${claims.phiCost}, while Qwen2.5-72B needs ${fmt(cq['qwen2.5-72b'].cost)},
          Llama-3.3-70B ${fmt(cq['llama3.3-70b'].cost)} and {claims.topLargeName} ${fmt(cq[claims.topLarge].cost)}, at
          accuracy within {claims.costWithin} points of Phi-4 either way. Costs are estimates from each judge’s GPU count and
          approximate decode speed.
        </p>
      </Prose>
      <div className="stack">
        <ChartFrame
          id="scale-size" title="Accuracy against judge size"
          sub="Hover a mark for both budgets. The dashed line joins the Qwen3 family, 0.6B to 32B."
          controls={<>
            <Segmented label="Budget" value={mode} onChange={setMode} options={budgetOpts()} />
            <Segmented label="Range" value={zoom} onChange={setZoom} options={[{ value: 'all', label: 'All' }, { value: 'top', label: 'Top' }]} />
          </>}
          legend={<FamilyLegend />}
          caption={<>Shaded: ±{claims.margin} points around Phi-4 at B = {QUICK}. Open marks are at B = {R}; the ringed circle is the Qwen3-32B ladder judge.</>}
          table={() => ({
            columns: [J1, { key: 'size', label: 'Size', align: 'left' }, { key: 'acc', label: 'Accuracy (%)' }],
            rows: judges.filter((j) => valueAt(j) != null).map((j) => ({ judge: j.name, size: j.size, acc: valueAt(j) })),
          })}
        >
          <SizeScatter mode={mode} zoom={zoom} height={420} />
        </ChartFrame>
        <ChartFrame
          id="scale-cost" title="What 1,000 verdicts cost"
          sub="Estimated serving cost (log scale), cheapest first, with each judge’s accuracy at the same budget"
          controls={<Segmented label="Budget" value={cb} onChange={setCb} options={budgetOpts(false)} />}
          caption="Shaded rows: Phi-4 and the large judges."
          table={() => ({
            columns: [J1, { key: 'cost', label: 'Cost per 1,000 verdicts ($)' }, { key: 'acc', label: 'Accuracy (%)' }],
            rows: costRows.map((r) => ({ judge: r.label, cost: r.v, acc: Number(r.right2.replace('%', '')) })),
          })}
        >
          <LogBars rows={costRows} domain={[lo, hi]} format={(v) => `$${v >= 1 ? fmt(v, 0) : fmt(v, v >= 0.1 ? 1 : 2)}`} xTitle="Dollars per 1,000 verdicts (log scale)" rowH={23} ariaLabel="Serving cost per 1,000 verdicts for every judge at the chosen budget" />
        </ChartFrame>
      </div>
    </>
  )
}

// ── 2. Budget ───────────────────────────────────────────────────────────────
const CONDITION_LABEL = {
  factor_budget: 'Budget only',
  factor_prompt: 'Prompt only',
  factor_temperature: 'Temperature only',
  factor_thinking: 'Thinking mode',
}

function Budget({ res }) {
  const fac = res.singleFactor
  const facRows = fac.conditions.map((c) => ({
    key: c,
    label: CONDITION_LABEL[c],
    dots: Object.entries(fac.judges).filter(([, v]) => v[c] && v[c].delta != null).map(([k, v]) => {
      const j = byKey[k]
      return {
        id: k, v: v[c].delta, fam: j.family,
        tip: <><TipHead fam={j.family} name={j.name} /><TipRow k={`Change from B = ${QUICK}`} v={`${signed(v[c].delta)} points`} /><TipRow k="Accuracy" v={`${fmt(v[c].acc)}%`} /><TipRow k="Parseable verdicts" v={`${fmt(v[c].ifr)}%`} /></>,
      }
    }),
  }))
  const facVals = facRows.flatMap((r) => r.dots.map((d) => d.v))
  const sw = res.sweep
  const swKeys = Object.keys(sw.judges)
  const [active, setActive] = useState(['qwen3-4b', 'phi4-14b', 'llama3.1-8b', 'qwen2.5-7b'].filter((k) => swKeys.includes(k)))
  const series = swKeys.map((k) => ({ key: k, label: name(k), fam: famOf(k), slug: FAMILY_SLUG[famOf(k)], values: sw.judges[k] }))
  const allV = swKeys.flatMap((k) => sw.judges[k])
  const toggle = (k) => setActive((a) => (a.includes(k) ? a.filter((x) => x !== k) : [...a, k]))
  return (
    <>
      <Prose>
        <p>
          The quick verdict beats extended reasoning on math for {claims.quickMath} of the {claims.nDual} judges that run at
          both budgets, while reasoning wins on general tasks for {claims.reasonGeneral} of {claims.nDual}, so one average
          hides both. Reasoning lifts Qwen3-4B by {claims.wino.gain} points on WinoGrande
          ({claims.wino.from}% to {claims.wino.to}%), where commonsense questions are hard to settle in {QUICK} tokens.
        </p>
        <p>
          The difference comes from the prompt rather than the budget. With the quick-verdict prompt held fixed, raising the
          budget leaves accuracy unchanged for {claims.unchanged} of {claims.nBudget} judges, and the largest change is{' '}
          {claims.largestBudgetMove} points. The reasoned prompt at {QUICK} tokens is cut off mid-explanation, and accuracy
          falls with the share of parseable verdicts (r = {claims.rIfr}).
        </p>
      </Prose>
      <div className="stack">
        <ChartFrame
          id="budget-factors" title="Change one factor at a time"
          sub={`Accuracy change from the B = ${QUICK} baseline (points)`}
          legend={<FamilyLegend families={['Llama', 'Qwen2.5', 'Qwen3', 'Phi-4']} />}
          caption={<>Budget alone barely moves any judge. The reasoned prompt at {QUICK} tokens moves accuracy by tens of points, because the verdict is cut off.</>}
          table={() => ({
            columns: [J1, ...fac.conditions.map((c) => ({ key: c, label: `${CONDITION_LABEL[c]} (points)` }))],
            rows: Object.entries(fac.judges).map(([k, v]) => ({ judge: name(k), ...Object.fromEntries(fac.conditions.map((c) => [c, v[c]?.delta])) })),
          })}
        >
          <DotPlot rows={facRows} domain={span(facVals, 0.05)} format={(v) => signed(v, 0)} xTitle="Points" rowH={44} labelW={124} ariaLabel="Accuracy change under each single-factor condition" refs={[{ v: 0 }]} />
        </ChartFrame>
        <ChartFrame
          id="budget-sweep" title="The budget sweep"
          sub={`Reasoned prompt, ${facts.nSweep} budgets from ${QUICK} to ${R} tokens. Pick judges to compare.`}
          caption="Accuracy dips where the explanation is cut short, then levels off."
          controls={
            <div className="chips" role="group" aria-label="Judges in the sweep">
              {swKeys.map((k) => (
                <button key={k} type="button" className="chip" aria-pressed={active.includes(k)} onClick={() => toggle(k)}>
                  <MarkIcon fam={famOf(k)} size={11} /> {name(k)}
                </button>
              ))}
            </div>
          }
          table={() => ({
            columns: [J1, ...sw.budgets.map((b) => ({ key: `b${b}`, label: `B = ${count(b)}` }))],
            rows: swKeys.map((k) => ({ judge: name(k), ...Object.fromEntries(sw.budgets.map((b, i) => [`b${b}`, sw.judges[k][i]])) })),
          })}
        >
          <LineChart series={series} xs={sw.budgets} active={active} yDomain={span(active.length ? active.flatMap((k) => sw.judges[k]) : allV, 0.08)}
            xFormat={(v) => count(v)} yTitle="Accuracy (%)" xTitle="Token budget (log scale)" height={380} onPick={toggle}
            ariaLabel="Accuracy against token budget for the selected judges" />
        </ChartFrame>
      </div>
    </>
  )
}

// ── 3. Domain ───────────────────────────────────────────────────────────────
function Domain({ res }) {
  const [b, setB] = useState('best')
  const keys = res.datasetKeys
  const rowsJ = [...judges].sort((a, c) => c.best - a.best)
  const budgetOf = (j) => String(b === 'best' ? j.bestBudget : b)
  const val = (k, d) => {
    const row = res.perDataset[k]?.[budgetOf(byKey[k])]
    return row ? row[keys.indexOf(d)] : null
  }
  const rows = rowsJ.filter((j) => res.perDataset[j.key]?.[budgetOf(j)]).map((j) => ({ key: j.key, label: j.name, fam: j.family }))
  const all = rows.flatMap((r) => keys.map((d) => val(r.key, d))).filter((v) => v != null)
  const cols = keys.map((k) => ({ key: k, label: DS_LABEL[k] }))
  return (
    <>
      <Prose>
        <p>
          The math-to-general gap runs from {claims.gapHi} points for {name(claims.gapHiJudge)} ({claims.gapHiMath}% on math,{' '}
          {claims.gapHiGeneral}% on general reasoning) down to {claims.strong3Gap.join('–')} points for Phi-4, Qwen3-8B and
          Qwen3-14B. The large judges other than DeepSeek-R1-Distill show the same gap ({claims.largeGap1.join('–')} points) as
          the strongest small ones ({claims.strong3Gap1.join('–')}), and no judge at any scale clears {claims.hsCeil}% on
          HellaSwag, so the gap follows the task at every size. Each domain is the unweighted mean of its datasets.
        </p>
      </Prose>
      <ChartFrame
        id="domain-heatmap" title="Accuracy per judge and dataset"
        sub="Rows ordered by accuracy at each judge’s better budget. Hover a cell."
        controls={<Segmented label="Budget" value={b} onChange={setB} options={budgetOpts()} />}
        caption="The last column is math minus general accuracy, in points."
        table={() => ({
          columns: [J1, ...cols.map((c) => ({ key: c.key, label: `${c.label} (%)` })), { key: 'gap', label: 'Gap (points)' }],
          rows: rows.map((r) => ({ judge: r.label, ...Object.fromEntries(keys.map((d) => [d, val(r.key, d)])), gap: res.domainMeans[r.key]?.[budgetOf(byKey[r.key])]?.gap })),
        })}
      >
        <Heatmap rows={rows} cols={cols} value={val} domain={[Math.min(...all), 100]} format={(v) => fmt(v, 2)}
          extraCol={{ label: 'Gap', value: (k) => fmt(res.domainMeans[k]?.[budgetOf(byKey[k])]?.gap, 1) }}
          tip={(r, c, v) => {
            const j = byKey[r.key]
            const dm = res.domainMeans[r.key]?.[budgetOf(j)]
            return <><TipHead fam={j.family} name={j.name} /><TipRow k={c.label} v={`${fmt(v)}%`} /><TipRow k="Math / science / general" v={`${fmt(dm.math, 1)} / ${fmt(dm.science, 1)} / ${fmt(dm.general, 1)}`} /><TipRow k="Budget" v={`B = ${count(Number(budgetOf(j)))}`} /></>
          }}
          ariaLabel="Heatmap of accuracy for every judge on every closed-ended dataset" />
      </ChartFrame>
    </>
  )
}

// ── 4. Open-ended ───────────────────────────────────────────────────────────
function OpenEnded({ res }) {
  const [metric, setMetric] = useState('se')
  const sorted = [...judges].filter((j) => j[metric] != null).sort((a, b) => b[metric] - a[metric])
  const vals = sorted.map((j) => j[metric])
  const bars = sorted.map((j) => ({
    key: j.key, label: j.name, v: j[metric], color: `var(--fam-${FAMILY_SLUG[j.family]})`, strong: ['phi4-14b', 'phi4r-14b'].includes(j.key),
    tip: <><TipHead fam={j.family} name={j.name} /><TipRow k="SummEval ρ" v={fmt(j.se, 3)} /><TipRow k="MT-Bench ρ" v={fmt(j.mt, 3)} /><TipRow k="Accuracy, better budget" v={`${fmt(j.best)}%`} /></>,
  }))
  const pts = judges.map((j) => ({
    key: j.key, x: j.best, y: j[metric], fam: j.family, open: j.bestBudget === REASONED, ring: j.kind === 'ladder', label: j.name,
    tip: <><TipHead fam={j.family} name={j.name} /><TipRow k="Accuracy, better budget" v={`${fmt(j.best)}%`} /><TipRow k={metric === 'se' ? 'SummEval ρ' : 'MT-Bench ρ'} v={fmt(j[metric], 3)} /></>,
  }))
  const mLabel = metric === 'se' ? 'SummEval ρ' : 'MT-Bench ρ'
  const metricCtl = <Segmented label="Benchmark" value={metric} onChange={setMetric} options={[{ value: 'se', label: 'SummEval' }, { value: 'mt', label: 'MT-Bench' }]} />
  const table = () => ({ columns: [J1, { key: 'acc', label: 'Accuracy (%)' }, { key: 'se', label: 'SummEval ρ' }, { key: 'mt', label: 'MT-Bench ρ' }], rows: sorted.map((j) => ({ judge: j.name, acc: j.best, se: j.se, mt: j.mt })) })
  return (
    <>
      <Prose>
        <p>
          On SummEval’s {count(facts.nSummaries)} human-scored summaries, the {claims.seNSlmWord} strongest small judges match or
          beat the large judges in {claims.seMatch} of {claims.seN} paired comparisons ({claims.seHigher} higher,{' '}
          {claims.seTied} not distinguishable), and all {claims.seLossesWord} losses are to {claims.topLargeName}. Phi-4
          reaches ρ = {claims.phiSe}, significantly above {word(claims.phiSeAbove)} of the {word(facts.nLarge)} large judges and{' '}
          {claims.phiSeGap} below {claims.topLargeName}.
        </p>
        <p>
          Closed- and open-ended rankings diverge. Phi-4 leads the small judges on binary judging and on SummEval but ranks{' '}
          {claims.phiMtRank} on MT-Bench (ρ = {claims.phiMt}), while its reasoning-trained sibling Phi-4-Reasoning ranks{' '}
          {claims.phirMtRank} (ρ = {claims.phirMt}). MT-Bench ρ is measured against one reference model (Qwen3.5-397B-A17B) over{' '}
          {facts.nConversations} conversations per student.
        </p>
      </Prose>
      <div className="figs-2">
        <ChartFrame id="open-ended-agreement" title="Agreement with the reference" sub={`${mLabel}, every judge`} controls={metricCtl} legend={<FamilyLegend />} table={table}>
          <BarList rows={bars} domain={[Math.min(0, ...vals), Math.max(...vals)]} format={(v) => fmt(v, 3)} rowH={21} ariaLabel="Open-ended agreement for every judge" />
        </ChartFrame>
        <ChartFrame id="open-ended-scatter" title="Accuracy against agreement" sub="One mark per judge" controls={metricCtl} legend={<FamilyLegend />} table={table}
          caption="A tight relation on SummEval and a looser one on MT-Bench, where Phi-4 drops and Phi-4-Reasoning rises.">
          <Scatter points={pts} xDomain={span(judges.map((j) => j.best), 0.04)} yDomain={span(vals, 0.08)}
            yFormat={(v) => fmt(v, 1)} xTitle="Accuracy, better budget (%)" yTitle="Spearman’s ρ" labels={['phi4-14b', 'phi4r-14b']} height={420}
            ariaLabel="Closed-ended accuracy against open-ended agreement" />
        </ChartFrame>
      </div>
    </>
  )
}

// ── 5. Strategies ───────────────────────────────────────────────────────────
const PROTO = [
  ['reference_based', 'With the reference', 'var(--accent)'],
  ['geval', 'G-Eval rubric', 'var(--teal)'],
  ['reference_free', 'Without the reference', 'var(--fam-llama)'],
]

function Strategies({ res }) {
  const [pb, setPb] = useState(QUICK)
  const [vb, setVb] = useState(QUICK)
  const pers = Object.entries(res.personas).filter(([, v]) => v[String(pb)]).map(([k, v]) => ({ k, x: v[String(pb)] }))
    .sort((a, b) => a.x.range - b.x.range)
  const persDotRows = pers.map(({ k, x }) => ({
    key: k, label: `${name(k)} · ${fmt(x.range, 2)}`, range: true,
    dots: [
      { id: 'base', v: x.base, fam: famOf(k), open: true, r: 6.5, tip: <><TipHead fam={famOf(k)} name={`${name(k)}, no persona`} /><TipRow k="Accuracy" v={`${fmt(x.base)}%`} /></> },
      ...facts.personas.map((p) => ({ id: p, v: x[p], color: p === 'Lenient' || p === 'Strict' ? 'var(--fam-llama)' : 'var(--accent-2)', r: 3.6, tip: <><TipHead fam={famOf(k)} name={`${name(k)}, ${p}`} /><TipRow k="Accuracy" v={`${fmt(x[p])}%`} /><TipRow k="No persona" v={`${fmt(x.base)}%`} /></> })),
    ],
  }))
  const persVals = persDotRows.flatMap((r) => r.dots.map((d) => d.v))
  const pv = res.prompts
  const pvList = Object.entries(pv.judges).filter(([, v]) => v[String(vb)]).map(([k, v]) => ({ k, x: v[String(vb)] })).sort((a, b) => a.x.range - b.x.range)
  const pvRows = pvList.map(({ k, x }) => ({
    key: k, label: `${name(k)} · ${fmt(x.range, 2)}`, range: true,
    dots: pv.variants.map((v) => ({ id: v, v: x[v], fam: famOf(k), r: 4.4, open: v !== 'original', tip: <><TipHead fam={famOf(k)} name={name(k)} /><TipRow k={pv.labels[v]} v={`${fmt(x[v])}%`} /></> })),
  }))
  const pvVals = pvRows.flatMap((r) => r.dots.map((d) => d.v))
  const pr = res.protocols.best
  const prList = Object.entries(pr).sort((a, b) => b[1].reference_based.acc - a[1].reference_based.acc)
  const prRows = prList.map(([k, v]) => ({
    key: k, label: name(k), range: true,
    dots: PROTO.filter(([p]) => v[p]).map(([p, l, c]) => ({ id: p, v: v[p].acc, color: c, r: 4.6, tip: <><TipHead fam={famOf(k)} name={name(k)} /><TipRow k={l} v={`${fmt(v[p].acc)}%`} /><TipRow k="Budget" v={`B = ${count(v[p].budget)}`} /></> })),
  }))
  const prVals = prRows.flatMap((r) => r.dots.map((d) => d.v))
  const fl = res.protocols.flip
  const flList = Object.entries(fl).sort((a, b) => Math.min(...Object.values(a[1])) - Math.min(...Object.values(b[1])))
  const flRows = flList.map(([k, v]) => ({
    key: k, label: name(k), range: true,
    dots: Object.entries(v).map(([b, f]) => ({ id: b, v: f, fam: famOf(k), open: Number(b) === REASONED, r: 5, tip: <><TipHead fam={famOf(k)} name={name(k)} open={Number(b) === REASONED} /><TipRow k={`Flip rate, B = ${count(Number(b))}`} v={fmt(f, 2)} /></> })),
  }))
  return (
    <>
      <Prose>
        <p>
          Multi-agent debate degrades accuracy in every configuration tested. Under the Reflect-Critique-Refine protocol, the best debate (Qwen3-4B, same
          model, {claims.debateBest}%) falls {claims.debateGap} points below Qwen3-4B alone ({claims.debateAlone}%). A binary
          verdict gives the agents little to exchange, so the critique step can pass a confident wrong verdict to an agent
          that was right.
        </p>
        <p>
          The best of the {word(facts.nJuries)} majority-vote juries of three reaches {claims.juryBest}%,{' '}
          {claims.juryGain} points above the best single small judge, because the top judges share most of their mistakes.
        </p>
        <p>
          At their best budget, Qwen3-14B and Phi-4 stay within a point across{' '}
          {word(facts.nPersonas)} adversarial personas, and judge rankings stay stable across {claims.nVariantsWord} wordings
          of the judging instruction (mean pairwise Kendall’s τ = {claims.tau10} at B = {QUICK}).
        </p>
      </Prose>
      <div className="figs-2">
        <ChartFrame id="strategies-personas" title="Adversarial personas" sub="Accuracy under each persona. The label shows the range in points."
          controls={<Segmented label="Budget" value={pb} onChange={setPb} options={budgetOpts(false)} />}
          caption="Open mark: no persona. Red dots: Strict and Lenient, the personas that move judges most."
          table={() => ({ columns: [J1, { key: 'base', label: 'No persona (%)' }, ...facts.personas.map((p) => ({ key: p, label: `${p} (%)` })), { key: 'range', label: 'Range (points)' }], rows: pers.map(({ k, x }) => ({ judge: name(k), ...x })) })}>
          <DotPlot rows={persDotRows} domain={span(persVals, 0.06)} rowH={40} labelW={170} xTitle="Accuracy (%)" ariaLabel="Accuracy under each adversarial persona" />
        </ChartFrame>
        <ChartFrame id="strategies-prompts" title="Rewording the instruction" sub={`${claims.nVariantsWord[0].toUpperCase() + claims.nVariantsWord.slice(1)} wordings. The label shows the range in points.`}
          controls={<Segmented label="Budget" value={vb} onChange={setVb} options={budgetOpts(false)} />}
          caption={<>Filled: the original instruction. Mean pairwise Kendall’s τ between the rankings: {claims.tau10} at B = {QUICK}, {claims.tau8192} at B = {R}.</>}
          table={() => ({ columns: [J1, ...pv.variants.map((v) => ({ key: v, label: `${pv.labels[v]} (%)` })), { key: 'range', label: 'Range (points)' }], rows: pvList.map(({ k, x }) => ({ judge: name(k), ...x })) })}>
          <DotPlot rows={pvRows} domain={span(pvVals, 0.04)} rowH={26} labelW={170} xTitle="Accuracy (%)" ariaLabel="Accuracy under each prompt wording" />
        </ChartFrame>
        <ChartFrame id="strategies-protocols" title="Judging protocols" sub="Each judge at its better budget per protocol (%)"
          legend={<div className="legend">{PROTO.map(([p, l, c]) => <span key={p} className="legend-item"><svg viewBox="-6 -6 12 12" width="12"><circle r="4.5" style={{ fill: c }} /></svg> {l}</span>)}</div>}
          caption="Withholding the reference costs every judge, small and large."
          table={() => ({ columns: [J1, ...PROTO.map(([p, l]) => ({ key: p, label: `${l} (%)` }))], rows: prList.map(([k, v]) => ({ judge: name(k), ...Object.fromEntries(PROTO.map(([p]) => [p, v[p]?.acc])) })) })}>
          <DotPlot rows={prRows} domain={span(prVals, 0.04)} rowH={24} labelW={170} xTitle="Accuracy (%)" ariaLabel="Accuracy with and without the reference and under G-Eval scoring" />
        </ChartFrame>
        <ChartFrame id="strategies-flip" title="Swapping the answers" sub="Mean flip rate when the two answers trade places. Lower is more consistent."
          legend={<BudgetLegend />} caption="Size does not make a judge consistent; thinking mode lowers the flip rate."
          table={() => ({ columns: [J1, { key: 'b10', label: `B = ${QUICK}` }, { key: 'b8192', label: `B = ${R}` }], rows: flList.map(([k, v]) => ({ judge: name(k), b10: v[String(QUICK)], b8192: v[String(REASONED)] })) })}>
          <DotPlot rows={flRows} domain={[0, 1]} rowH={24} labelW={170} xTitle="Flip rate" format={(v) => fmt(v, 1)} ariaLabel="Pairwise flip rate per judge at both budgets" />
        </ChartFrame>
      </div>
    </>
  )
}

// ── 6. Students ─────────────────────────────────────────────────────────────
function Students({ res }) {
  const st = res.students
  const solverList = [...st.solver].sort((a, b) => b.acc - a.acc)
  const solver = solverList.map((s) => ({
    key: s.key, label: s.weights.split('/')[1], v: s.acc, color: s.origin === 'probe' ? 'var(--fam-qwen3)' : 'var(--accent-2)', open: s.origin === 'probe',
    tip: <><div className="tt-head">{s.weights}</div><TipRow k="Solver accuracy" v={`${fmt(s.acc, 1)}%`} /><TipRow k="Family" v={s.family} /></>,
  }))
  const fams = Object.keys(st.lineage)
  const lin = fams.flatMap((f) => facts.budgets.map((b) => ({
    key: `${f}${b}`, label: `${f}, B = ${count(b)}`, v: st.lineage[f][String(b)], color: `var(--fam-${FAMILY_SLUG[f]})`, open: b === REASONED,
    tip: <><TipHead fam={f} name={f} open={b === REASONED} /><TipRow k={`Lineage effect, B = ${count(b)}`} v={signed(st.lineage[f][String(b)], 3)} /></>,
  })))
  const lv = lin.map((x) => x.v)
  return (
    <>
      <Prose>
        <p>
          Given room to reason, capable judges agree on how to rank {word(claims.nStudentsWide)} students from{' '}
          {word(claims.nStudentFamilies)} families (τ = {claims.studentsTau}), and their ranking matches the students’ own solver
          accuracy (τ = {claims.studentsTauSolver}). The students span {claims.solverRange.join('–')}% solver accuracy. To
          separate lineage from quality, a strong Qwen3-family student joins the pool and each judge’s general leniency is
          subtracted out.
        </p>
        <p>
          At B = {QUICK} the lineage effect on the false-accept rate is {claims.lineage['10'].Qwen3} for Qwen3,{' '}
          {claims.lineage['10'].Llama} for Llama and {claims.lineage['10']['Qwen2.5']} for Qwen2.5, so no family favors its own
          lineage at the quick verdict. A preference appears only once judges reason (B = {R}), mainly for Qwen3
          ({claims.lineage['8192'].Qwen3}; Llama {claims.lineage['8192'].Llama}, Qwen2.5 {claims.lineage['8192']['Qwen2.5']}).
        </p>
      </Prose>
      <div className="figs-2">
        <ChartFrame id="students-solver" title="The students" sub={`Solver accuracy over the ${word(facts.nClosed)} closed-ended datasets (%)`} caption="Outlined: the strong Qwen3-family student added as a probe."
          table={() => ({ columns: [{ key: 's', label: 'Student', align: 'left' }, { key: 'f', label: 'Family', align: 'left' }, { key: 'acc', label: 'Solver accuracy (%)' }], rows: solverList.map((s) => ({ s: s.weights, f: s.family, acc: s.acc })) })}>
          <BarList rows={solver} domain={[0, 100]} format={(v) => fmt(v, 1)} rowH={28} labelW={190} ariaLabel="Solver accuracy of each student" />
        </ChartFrame>
        <ChartFrame id="students-lineage" title="Lineage effect on the false-accept rate" sub="How much more a family accepts wrong answers from its own lineage"
          legend={<BudgetLegend fam="Qwen3" />} caption={<>Difference-in-differences of false-accept rates. Positive means a preference for the family’s own student.</>}
          table={() => ({ columns: [{ key: 'f', label: 'Family', align: 'left' }, { key: 'b10', label: `B = ${QUICK}` }, { key: 'b8192', label: `B = ${R}` }], rows: fams.map((f) => ({ f, b10: st.lineage[f][String(QUICK)], b8192: st.lineage[f][String(REASONED)] })) })}>
          <BarList rows={lin} domain={[Math.min(0, ...lv) - 0.02, Math.max(...lv) + 0.02]} format={(v) => signed(v, 3)} rowH={28} labelW={150} valueW={62} ariaLabel="Lineage effect per judge family and budget" />
        </ChartFrame>
      </div>
    </>
  )
}

const SECTIONS = { scale: Scale, budget: Budget, domain: Domain, 'open-ended': OpenEnded, strategies: Strategies, students: Students }

export default function FindingDetails({ slug }) {
  const res = useResults()
  const S = SECTIONS[slug]
  if (!res || !S) return <div className="skeleton" style={{ height: 420 }} />
  return <div className="details-content"><S res={res} /></div>
}
