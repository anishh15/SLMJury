import { useEffect, useState } from 'react'
import { Icon, Mark, MarkIcon } from '../components/ui'
import { claims, core, facts, famLabel, judges } from '../lib/data'
import { count, word } from '../lib/format'
import '../styles/content.css'
import '../styles/method.css'

const [QUICK, REASONED] = [facts.budgets[0], facts.budgets[facts.budgets.length - 1]]
const R = count(REASONED)
const hf = (id) => `https://huggingface.co/${id}`
const THINKS = { no: 'no', reasoned: `at B = ${R}`, always: 'always' }
const FAM_ORDER = ['Llama', 'Qwen2.5', 'Qwen3', 'Phi-4', 'Large']

function Flow() {
  const fams = ['Llama', 'Qwen2.5', 'Qwen3', 'Phi-4', 'Large']
  const labels = facts.verdictLabels
  const steps = [
    { n: '01', t: 'A question and its reference answer', d: 'math, science and general reasoning', big: facts.nClosed, unit: 'datasets', g: <span className="mf-g mono">Q · A</span> },
    { n: '02', t: 'A student answers it', d: 'judgments per configuration', big: count(facts.nItems), unit: '', g: <span className="mf-g mono">“…”</span> },
    { n: '03', t: 'The judge gives a verdict', d: `at B = ${QUICK} or B = ${R}`, big: facts.nSmall + facts.nLarge, unit: 'judges', g: <svg className="mf-g" viewBox="-60 -12 120 24" aria-hidden="true">{fams.map((f, i) => <g key={f} transform={`translate(${-44 + i * 22},0)`}><Mark fam={f} r={6} /></g>)}</svg> },
    { n: '04', t: 'Parsed and checked against the oracle', d: `${labels.slice(0, -1).join(', ')} or ${labels[labels.length - 1]}`, big: labels.length, unit: 'outcomes', g: <span className="mf-g mono">{labels.map((l) => l[0]).join(' · ')}</span> },
    { n: '05', t: 'Judges compared in pairs', d: `points, the equivalence margin, over ${facts.nJudgePairs} pairs`, big: `±${claims.margin}`, unit: '', gold: true, g: <span className="mf-g mono">A ≈ B</span> },
  ]
  return (
    <ol className="mflow" aria-label="The judging pipeline">
      {steps.map((s) => (
        <li key={s.n} className="mf">
          <div className="mf-top">{s.g}</div>
          <div className="mf-n tnum">{s.n}</div>
          <div className="mf-t">{s.t}</div>
          <div className={`mf-big tnum${s.gold ? ' gold' : ''}`}>{s.big}{s.unit && <span> {s.unit}</span>}</div>
          <div className="mf-d">{s.d}</div>
        </li>
      ))}
    </ol>
  )
}

function Fold({ id, title, meta, children, open: start = false }) {
  const [open, setOpen] = useState(start)
  useEffect(() => { if (start) setOpen(true) }, [start])
  return (
    <section className={`fold method-anchor${open ? ' open' : ''}`} id={id}>
      <button type="button" className="fold-btn" aria-expanded={open} onClick={() => setOpen((o) => !o)}>
        <span className="fold-t">{title}</span>
        <span className="fold-m">{meta}</span>
        <Icon.chevronDown className="fold-c" />
      </button>
      {open && <div className="fold-body">{children}</div>}
    </section>
  )
}

export default function Method({ route }) {
  const section = route?.parts?.[1]
  useEffect(() => {
    if (!section) return
    const el = document.getElementById(`m-${section}`)
    if (el) requestAnimationFrame(() => el.scrollIntoView({ block: 'start' }))
  }, [section])
  const ds = core.datasets
  const byFam = [...judges].sort((a, b) => FAM_ORDER.indexOf(a.family) - FAM_ORDER.indexOf(b.family) || a.params - b.params || a.name.localeCompare(b.name))
  const BEYOND = [
    ['Adversarial personas', facts.nPersonas, 'system prompts that push the judge, such as Strict and Lenient'],
    ['Majority voting', facts.nJuries, `juries of three from the top ${word(facts.nJuryPool)} judges, decided by majority vote`],
    ['Multi-agent debate', facts.nAgents, 'agents give verdicts, then critique and refine them (Reflect-Critique-Refine)'],
  ]
  const VARIES = [
    ['Open-ended scoring', `${count(facts.nSummaries)} + ${facts.nConversations}`, `SummEval summaries and MT-Bench conversations, scored 1 to ${facts.scoreScale} and compared by Spearman’s ρ`],
    ['Wordings', facts.variants.length, 'ways to phrase the judging instruction'],
    ['Protocols', facts.nProtocols, 'no reference, a G-Eval rubric, and pairwise comparison in both orders'],
    ['Budgets', facts.nSweep, `points in a sweep from ${QUICK} to ${R} tokens, one factor at a time`],
  ]
  return (
    <>
      <section className="page-head">
        <div className="wrap">
          <h1 className="h-display">One pipeline for every judge.</h1>
          <p className="lead">Every judge gets the same students, prompts, parser and oracle. Then one setting changes at a time.</p>
        </div>
      </section>

      <section className="wrap">
        <Flow />
        <p className="mnote">
          At <strong>B = {QUICK}</strong> the judge answers in one word, and at <strong>B = {R}</strong> it reasons first.
          Accuracy is the share of verdicts that match a programmatic oracle, which checks each student answer against the reference.
          Every comparison is paired on the same items, corrected over all {facts.nJudgePairs} judge pairs.
        </p>
      </section>

      <section className="wrap">
        <h2 className="h3 msub">Beyond one judge</h2>
        <ul className="mlist">
          {BEYOND.map(([t, n, d]) => (
            <li key={t} className="ml">
              <span className="ml-t">{t}</span>
              <span className="ml-n tnum">{n}</span>
              <span className="ml-d">{d}</span>
            </li>
          ))}
        </ul>
        <a className="link-btn mbeyond" href="#/findings/strategies">What they do to accuracy <Icon.arrow width="14" /></a>
        <h2 className="h3 msub">What else changes, one at a time</h2>
        <ul className="mlist">
          {VARIES.map(([t, n, d]) => (
            <li key={t} className="ml">
              <span className="ml-t">{t}</span>
              <span className="ml-n tnum">{n}</span>
              <span className="ml-d">{d}</span>
            </li>
          ))}
        </ul>
      </section>

      <section className="wrap mfolds">
        <Fold id="m-datasets" title="Datasets" meta={`${ds.closed.length} closed-ended, ${facts.nOpen} open-ended`} open={section === 'datasets'}>
          <div className="table-wrap">
            <table className="dtable">
              <thead><tr><th>Dataset</th><th>Key</th><th>Domain</th><th>Answer</th><th className="r">Items per student</th></tr></thead>
              <tbody>
                {ds.closed.map((d) => (
                  <tr key={d.key}><td>{d.name}</td><td><code>{d.key}</code></td><td>{d.domain}</td><td>{d.answer}</td><td className="r">{count(d.items)}</td></tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="table-wrap" style={{ marginTop: 14 }}>
            <table className="dtable">
              <thead><tr><th>Dataset</th><th>Task</th><th className="r">Items</th><th>Reference scores</th></tr></thead>
              <tbody>
                <tr><td>SummEval</td><td>rate a summary on coherence, consistency, fluency and relevance (1–{facts.scoreScale})</td><td className="r">{count(facts.nSummaries)} summaries of {facts.nArticles} articles</td><td>human scores</td></tr>
                <tr><td>MT-Bench</td><td>rate a two-turn conversation (1–{facts.scoreScale})</td><td className="r">{facts.nConversations} questions × both students</td><td>Qwen3.5-397B-A17B, with GPT-OSS-120B as a second scorer</td></tr>
              </tbody>
            </table>
          </div>
        </Fold>
        <Fold id="m-models" title="Models" meta={`${judges.length} judges, ${core.models.students.length} students`} open={section === 'models'}>
          <div className="table-wrap">
            <table className="dtable">
              <thead><tr><th>Family</th><th>Judge</th><th className="r">Size</th><th>Weights</th><th>Thinks</th></tr></thead>
              <tbody>
                {byFam.map((j) => (
                  <tr key={j.key}>
                    <td><span className="row" style={{ gap: 7, flexWrap: 'nowrap' }}><MarkIcon fam={j.family} size={12} ring={j.kind === 'ladder'} /> {j.kind === 'ladder' ? 'Qwen3 (ladder)' : famLabel(j.family)}</span></td>
                    <td><a href={`#/judge/${j.key}`}>{j.name}</a></td>
                    <td className="r">{j.size}{j.active ? ` (${j.active}B active)` : ''}</td>
                    <td><a href={hf(j.weights)} target="_blank" rel="noreferrer"><code>{j.weights}</code></a></td>
                    <td>{THINKS[j.thinks]}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="grid-2" style={{ marginTop: 14 }}>
            <div className="table-wrap">
              <table className="dtable">
                <thead><tr><th>Student</th><th>Used for</th></tr></thead>
                <tbody>
                  {core.models.students.map((s) => (
                    <tr key={s.key}><td><a href={hf(s.weights)} target="_blank" rel="noreferrer"><code>{s.weights}</code></a></td><td>{s.use === 'main' ? 'every experiment' : 'family bias'}</td></tr>
                  ))}
                </tbody>
              </table>
            </div>
            <div className="table-wrap">
              <table className="dtable">
                <thead><tr><th>MT-Bench reference scorer</th></tr></thead>
                <tbody>
                  {core.models.references.map((s) => (
                    <tr key={s.key}><td><a href={hf(s.weights)} target="_blank" rel="noreferrer"><code>{s.weights}</code></a></td></tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </Fold>
        <a className="mdocs" href="#/docs/prompts">
          <span><strong>Prompts and code</strong><span className="muted">Every prompt, word for word, and the command line, in the docs.</span></span>
          <Icon.arrow />
        </a>
      </section>
    </>
  )
}
