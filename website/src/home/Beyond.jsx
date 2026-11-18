import { Icon } from '../components/ui'
import { claims, facts } from '../lib/data'
import { word } from '../lib/format'

// The three ways the study goes beyond one judge, each with its number.
const ROWS = [
  {
    term: 'Adversarial personas',
    big: <>{[claims.personaRange['phi4-14b'], claims.personaRange['qwen3-14b']].sort((a, b) => a - b).join('–')}</>,
    unit: 'points',
    line: `Phi-4 and Qwen3-14B stay within a point across ${word(facts.nPersonas)} adversarial personas.`,
  },
  {
    term: 'Majority voting',
    big: <>+{claims.juryGain}</>,
    unit: 'points',
    line: `The best of ${facts.nJuries} majority-vote juries of three, over the best single small judge.`,
  },
  {
    term: 'Multi-agent debate',
    big: <>−{claims.debateGap}</>,
    unit: 'points',
    line: 'Debate degrades accuracy. The best debate falls this far below Qwen3-4B judging alone.',
  },
]

export default function Beyond() {
  return (
    <section className="section beyond" aria-labelledby="beyond-title">
      <div className="wrap-wide">
        <h2 className="h2" id="beyond-title" data-reveal>Beyond one judge</h2>
        <ul className="by-row" data-reveal="stagger">
          {ROWS.map((r) => (
            <li key={r.term}>
              <a className="by-item" href="#/findings/strategies">
                <span className="by-term">{r.term}</span>
                <span className="by-big tnum">{r.big}<small>{r.unit}</small></span>
                <span className="by-line">{r.line}</span>
                <Icon.arrow className="by-go" />
              </a>
            </li>
          ))}
        </ul>
      </div>
    </section>
  )
}
