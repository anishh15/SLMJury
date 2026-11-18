import { useEffect, useMemo, useState } from 'react'
import { Icon, MarkIcon } from '../components/ui'
import MathText from '../components/MathText'
import { byKey, core, FAMILIES, QUICK, REASONED, setHovered } from '../lib/data'
import { count } from '../lib/format'
import { useMedia } from '../lib/motion'

const FAM_ORDER = ['Llama', 'Qwen2.5', 'Qwen3', 'Phi-4', 'Large']
const items = core.items
const nJudges = items[0].verdicts.length

// One judge: its name, then after the reveal a ✓ or ✗ for agreement with the
// oracle. The verdict itself is plain text; colour means agreement only.
function Seat({ v, i, shown, oracle }) {
  const j = byKey[v.key]
  const ok = v.verdict === oracle
  const state = shown ? (ok ? ' agree' : ' disagree') : ''
  return (
    <a href={`#/judge/${j.key}`} className={`seat${state}`} style={{ '--i': i }}
      onPointerEnter={() => setHovered(j.key)} onPointerLeave={() => setHovered(null)}
      aria-label={shown ? `${j.name} said ${v.verdict}, ${ok ? 'as the oracle did' : 'unlike the oracle'}` : `${j.name}: verdict hidden`}>
      <span className="seat-name">{j.name}{v.budget === REASONED ? <span className="seat-dag">{"\u00a0"}†</span> : null}</span>
      <span className="seat-sub">
        <span className="seat-size">{j.size}</span>
        <span className="seat-v">said {v.verdict}</span>
      </span>
      <span className="seat-mark" aria-hidden="true">
        <span className="seat-wait" />
        {shown && (ok ? <Icon.check className="seat-ok" /> : <Icon.close className="seat-no" />)}
      </span>
    </a>
  )
}

export default function Verdicts() {
  const [idx, setIdx] = useState(0)
  const [guess, setGuess] = useState(null)
  const [shown, setShown] = useState(false)
  const [open, setOpen] = useState(false)
  const [all, setAll] = useState(false)
  const phone = useMedia('(max-width: 640px)')
  const it = items[idx]
  const math = it.dataset === 'math'
  useEffect(() => { setGuess(null); setShown(false); setOpen(false); setAll(false) }, [idx])
  const tiles = !phone || all
  const byFam = useMemo(() => FAM_ORDER.map((f) => [f, it.verdicts
    .filter((v) => byKey[v.key].family === f)
    .sort((a, b) => byKey[a.key].params - byKey[b.key].params)]), [idx])
  let seatIndex = 0
  const agree = it.verdicts.filter((v) => v.verdict === it.oracle)
  const small = it.verdicts.filter((v) => byKey[v.key].kind === 'small')
  const large = it.verdicts.filter((v) => byKey[v.key].kind === 'large')
  const ok = (vs) => vs.filter((v) => v.verdict === it.oracle).length
  const reveal = (g) => { if (g) setGuess(g); setShown(true) }
  const step = (n) => setIdx((i) => (i + n + items.length) % items.length)
  const split = !math && it.answer.length >= 3 ? it.reasoning.split(it.answer) : [it.reasoning]
  return (
    <section className="section verdicts" aria-labelledby="verdicts-title">
      <div className="wrap-wide">
        <div className="sec-head split" data-reveal="stagger">
          <h2 className="h2" id="verdicts-title">{nJudges} judges, one answer</h2>
          <p className="dek">Guess first, then see all {nJudges} verdicts.</p>
        </div>
        <div className="vs-grid" data-reveal="stagger">
          <article className="item-card" aria-label={`Item ${idx + 1} of ${items.length}`}>
            <header className="item-head">
              <span className="item-ds">{it.datasetName}</span>
              <span className="item-student">Student: {it.studentName}</span>
              <span className="item-nav">
                <button type="button" className="icon-btn" onClick={() => step(-1)} aria-label="Previous item"><Icon.arrowLeft /></button>
                <span className="tnum item-count">{idx + 1} / {items.length}</span>
                <button type="button" className="icon-btn" onClick={() => step(1)} aria-label="Next item"><Icon.arrow /></button>
              </span>
            </header>
            <div className="item-q">{math ? <MathText text={it.question} /> : it.question}</div>
            <div className="item-answers">
              <div><span className="fl">Reference answer</span><span className="fv">{math ? <MathText text={it.reference} whole /> : it.reference}</span></div>
              <div><span className="fl">Student’s answer</span><span className="fv">{it.answer}</span></div>
            </div>
            <div className={`item-reasoning${open ? ' open' : ''}`} id="item-reasoning">
              <span className="fl">Student’s reasoning</span>
              <div className="reasoning-text">
                {math ? <MathText text={it.reasoning} /> : split.map((p, i) => <span key={i}>{p}{i < split.length - 1 && <mark>{it.answer}</mark>}</span>)}
              </div>
            </div>
            <button type="button" className="link-btn" aria-expanded={open} aria-controls="item-reasoning" onClick={() => setOpen((o) => !o)}>
              {open ? 'Show less' : 'Read the full reasoning'}
            </button>
            <div className="guess">
              <span className="guess-q">Is the student right?</span>
              <div className="guess-btns">
                {['Correct', 'Incorrect'].map((g) => (
                  <button key={g} type="button" className="btn guess-btn" aria-pressed={guess === g} onClick={() => reveal(g)} disabled={shown && guess !== g && guess !== null}>
                    {g}
                  </button>
                ))}
                <button type="button" className={`link-btn${shown ? ' gone' : ''}`} onClick={() => reveal(null)} aria-hidden={shown || undefined} tabIndex={shown ? -1 : undefined}>Just show the verdicts</button>
              </div>
            </div>
          </article>
          <div className={`jury${shown ? ' shown' : ''}`} aria-live="polite">
            <div className={`oracle-bar${shown ? ' shown' : ''}`}>
              {shown ? (
                <>
                  <span className="ob-oracle">Oracle: <strong>{it.oracle}</strong></span>
                  <span className="ob-count tnum"><strong>{agree.length}</strong> of {it.verdicts.length} judges agree</span>
                  <span className="ob-split tnum">small {ok(small)}/{small.length} · large {ok(large)}/{large.length}</span>
                  {guess && <span className={`ob-you ${guess === it.oracle ? 'right' : 'wrong'}`}>{guess === it.oracle ? 'You agree with the oracle' : 'The oracle disagrees with you'}</span>}
                </>
              ) : (
                <span className="ob-wait">Verdicts hidden until you guess</span>
              )}
            </div>
            {!tiles && shown && (
              <ul className="jury-groups">
                {byFam.map(([f, vs]) => (
                  <li key={f}>
                    <span className="jg-fam"><MarkIcon fam={f} size={12} /> {FAMILIES.find((x) => x.key === f).label.replace(' (70B–122B)', '')}</span>
                    <span className="jg-n tnum"><strong>{ok(vs)}</strong> of {vs.length} agree</span>
                    <span className="jg-bar" aria-hidden="true"><i style={{ width: `${(ok(vs) / vs.length) * 100}%` }} /></span>
                  </li>
                ))}
              </ul>
            )}
            {phone && (
              <button type="button" className="btn btn-sm jury-all" aria-expanded={all} onClick={() => setAll((x) => !x)}>
                {all ? 'Show the counts by family' : `Show all ${it.verdicts.length}`}
              </button>
            )}
            {tiles && <div className="jury-cols">
              {byFam.map(([f, vs]) => (
                <div key={f} className="jury-col">
                  <div className="jury-fam"><MarkIcon fam={f} size={12} /> {FAMILIES.find((x) => x.key === f).label.replace(' (70B–122B)', '')}</div>
                  {vs.map((v) => <Seat key={v.key} v={v} i={seatIndex++} shown={shown} oracle={it.oracle} />)}
                </div>
              ))}
            </div>}
            <p className="jury-note">
              <span><Icon.check className="jn-ok" /> agrees with the oracle</span>
              <span><Icon.close className="jn-no" /> disagrees</span>
              <span>Quick verdicts at B = {QUICK}; † always reasons, so B = {count(REASONED)}</span>
            </p>
          </div>
        </div>
      </div>
    </section>
  )
}
