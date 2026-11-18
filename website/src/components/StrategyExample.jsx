import { useState } from 'react'
import { Icon, MarkIcon } from './ui'
import { byKey, claims, facts, name } from '../lib/data'
import { count, fmt, word } from '../lib/format'

// One real item from the logs, run through personas, a jury and a debate, beside
// the aggregate numbers. The item is the first where the debate's opening
// majority matched the oracle and its final verdict did not.

function Verdict({ v, oracle, label }) {
  const ok = v === oracle
  return (
    <span className={`sx-v ${ok ? 'ok' : 'no'}`} title={`${label ? `${label}: ` : ''}${v}, ${ok ? 'as the oracle' : 'unlike the oracle'}`}>
      {label && <span className="sx-vl">{label}</span>}
      <span className="sx-vv">{v}</span>
      {ok ? <Icon.check aria-hidden="true" /> : <Icon.close aria-hidden="true" />}
    </span>
  )
}

const majority = (vs) => {
  const top = [...new Set(vs)].sort((a, b) => vs.filter((x) => x === b).length - vs.filter((x) => x === a).length)[0]
  return vs.filter((x) => x === top).length * 2 > vs.length ? top : null
}

export default function StrategyExample({ ex }) {
  const [open, setOpen] = useState(false)
  const o = ex.oracle
  const d = ex.debate
  const p = ex.persuasion
  const agent = name(d.members[0].key)
  const same = d.members.every((m) => m.key === d.members[0].key)
  return (
    <section className="sx" aria-labelledby="sx-h">
      <header className="sx-head">
        <span className="sx-tag">One example from the logs</span>
        <h3 className="h3" id="sx-h">What each strategy did to one answer</h3>
      </header>
      <div className="sx-item">
        <div className="sx-meta"><span>{ex.datasetName}</span><span>Student: {ex.studentName}</span></div>
        <p className="sx-q">{ex.question}</p>
        <dl className="sx-ans">
          <div><dt>Reference answer</dt><dd className="tnum">{ex.reference}</dd></div>
          <div><dt>Student’s answer</dt><dd className="tnum">{ex.answer}</dd></div>
          <div><dt>Oracle</dt><dd>{o}</dd></div>
        </dl>
      </div>

      <ol className="sx-lanes">
        <li className="sx-lane">
          <div className="sx-lt">Adversarial personas</div>
          <div className="sx-body">
            {ex.personas.map((x) => (
              <div key={x.key} className="sx-row">
                <span className="sx-who"><MarkIcon fam={byKey[x.key].family} size={11} /> {name(x.key)} <span className="faint">B = {x.budget}</span></span>
                <span className="sx-chips">
                  <Verdict v={x.base} oracle={o} label="No persona" />
                  {facts.personas.map((q) => <Verdict key={q} v={x.verdicts[q]} oracle={o} label={q} />)}
                </span>
              </div>
            ))}
          </div>
          <p className="sx-agg">
            <strong className="tnum">{claims.personaRange['qwen3-14b']} and {claims.personaRange['phi4-14b']} points</strong>
            Qwen3-14B’s and Phi-4’s accuracy range across the {word(facts.nPersonas)} personas at B = {ex.personas[0].budget}, over all {count(p.items)} judgments.
          </p>
        </li>

        <li className="sx-lane">
          <div className="sx-lt">Majority voting</div>
          <div className="sx-body">
            <div className="sx-row">
              <span className="sx-who">Best jury <span className="faint">{fmt(ex.jury.acc)}%</span></span>
              <span className="sx-chips">
                {ex.jury.members.map((m) => <Verdict key={m.key} v={m.verdict} oracle={o} label={`${name(m.key)}, B = ${count(m.budget)}`} />)}
                <span className="sx-arrow" aria-hidden="true">→</span>
                <Verdict v={ex.jury.verdict} oracle={o} label="Majority" />
              </span>
            </div>
          </div>
          <p className="sx-agg">
            <strong className="tnum">+{claims.juryGain} points</strong>
            The best of {facts.nJuries} juries gains this much over Phi-4 alone, the best single small judge.
          </p>
        </li>

        <li className="sx-lane">
          <div className="sx-lt">Multi-agent debate</div>
          <div className="sx-body">
            {d.rounds.map((rd, i) => {
              const m = majority(rd.map((a) => a.verdict))
              return (
                <div key={i} className="sx-row">
                  <span className="sx-who">Round {i + 1} <span className="faint">{same ? `${agent} × ${rd.length}` : ''}</span></span>
                  <span className="sx-chips">
                    {rd.map((a, k) => <Verdict key={k} v={a.verdict} oracle={o} label={same ? `Agent ${k + 1}, temperature ${d.members[k].t}` : name(d.members[k].key)} />)}
                    {m && <><span className="sx-arrow" aria-hidden="true">→</span><Verdict v={m} oracle={o} label={i === d.rounds.length - 1 ? 'Final' : 'Majority'} /></>}
                  </span>
                </div>
              )
            })}
            <button type="button" className="link-btn sx-more" aria-expanded={open} aria-controls="sx-args" onClick={() => setOpen((v) => !v)}>
              {open ? 'Hide the arguments' : 'Read the agents’ arguments'}
            </button>
            {open && (
              <ol className="sx-args" id="sx-args">
                {d.rounds.map((rd, i) => (
                  <li key={i}>
                    <div className="sx-argr">Round {i + 1}</div>
                    {rd.map((a, k) => <p key={k}><span className="faint">Agent {k + 1} ({a.verdict}):</span> {a.text}</p>)}
                  </li>
                ))}
              </ol>
            )}
          </div>
          <p className="sx-agg">
            <strong className="tnum">{count(p.swayed)} of {count(p.recoverable)}</strong>
            Items where {agent}’s three agents opened split, with at least one of them right, and the debate still ended
            wrong, over all {count(p.items)} judgments. Its accuracy is {claims.debateGap} points below {agent} judging alone.
          </p>
        </li>
      </ol>
    </section>
  )
}
