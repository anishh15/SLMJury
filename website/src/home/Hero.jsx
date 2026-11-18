import { useEffect, useRef, useState } from 'react'
import NumberFlow from '@number-flow/react'
import Atmosphere from './Atmosphere'
import JudgeStrip from './JudgeStrip'
import Wordmark from '../components/Wordmark'
import JudgePicker from '../components/JudgePicker'
import { Icon, MarkIcon } from '../components/ui'
import { byKey, claims, facts, meta, pointOf, statement } from '../lib/data'
import { count, fmt, word } from '../lib/format'

const nf2 = { minimumFractionDigits: 2, maximumFractionDigits: 2 }

export function useMedia(q) {
  const [m, setM] = useState(() => typeof window !== 'undefined' && window.matchMedia(q).matches)
  useEffect(() => {
    const mq = window.matchMedia(q)
    const on = () => setM(mq.matches)
    mq.addEventListener?.('change', on)
    return () => mq.removeEventListener?.('change', on)
  }, [q])
  return m
}

// One judge of the headline comparison; the name opens a picker.
function Side({ k, other, onPick, align }) {
  const j = byKey[k]
  const p = pointOf[k]
  return (
    <div className={`res-side ${align}`}>
      <JudgePicker value={k} onChange={onPick} disabled={[other]} label={align === 'left' ? 'First judge' : 'Second judge'} align={align}>
        <MarkIcon fam={j.family} size={13} />
        <span className="res-name">{j.name}</span>
        <span className="res-size">{j.size}</span>
        <Icon.chevronDown className="chev" />
      </JudgePicker>
      <span className="res-acc tnum"><NumberFlow value={p.acc} format={nf2} /><span className="pct">%</span></span>
    </div>
  )
}

export default function Hero() {
  const [[a, b], setPair] = useState(['phi4-14b', claims.topLarge])
  const ref = useRef(null)
  const [live, setLive] = useState(true)
  useEffect(() => {
    if (!('IntersectionObserver' in window) || !ref.current) return
    const io = new IntersectionObserver(([e]) => setLive(e.isIntersecting), { threshold: 0.02 })
    io.observe(ref.current)
    return () => io.disconnect()
  }, [])
  const pa = pointOf[a]
  const pb = pointOf[b]
  const st = statement(a, b)
  const diff = Math.abs(pb.accExact - pa.accExact)
  const ratio = Math.max(pa.costExact, pb.costExact) / Math.min(pa.costExact, pb.costExact)
  const cheap = pa.costExact <= pb.costExact ? byKey[a] : byKey[b]
  const ratioText = ratio >= 9.5 ? Math.round(ratio) : ratio.toFixed(1)
  // costs are compared only at one budget; otherwise name the judge that runs at the other
  const sameBudget = pa.budget === pb.budget
  const reasoner = pa.budget > pb.budget ? byKey[a] : byKey[b]
  const isDefault = a === 'phi4-14b' && b === claims.topLarge
  return (
    <section className={`hero${live ? '' : ' paused'}`} ref={ref} aria-labelledby="hero-title">
      <div className="hero-sky" aria-hidden="true" />
      <Atmosphere />
      <div className="hero-inner wrap-wide">
        <a className="venue-chip" href={meta.arxiv} target="_blank" rel="noreferrer" title={`Paper, ${meta.venue.label}`}>{meta.venue.chip}</a>
        <h1 id="hero-title" className="hero-mark"><Wordmark size="hero" glow /></h1>
        <p className="hero-q">Can small language models judge as well as large ones?</p>

        <div className={`result${st ? ' stated' : ''}`} aria-live="polite">
          <p className="res-metric">Accuracy on the {word(facts.nClosed)} closed-ended benchmarks</p>
          <Side k={a} other={b} align="left" onPick={(k) => setPair([k, b])} />
          <span className="res-mid" aria-hidden="true">{st && st.kinds.includes('equivalent') ? '≈' : diff < 0.005 ? '=' : pa.accExact > pb.accExact ? '>' : '<'}</span>
          <Side k={b} other={a} align="right" onPick={(k) => setPair([a, k])} />
          <p className="res-note">
            <span className="res-claim">{st ? st.short : `${fmt(diff)} points apart`}</span>
            <span className="res-dot" aria-hidden="true" />
            {sameBudget
              ? <span>{cheap.name} costs {ratioText}× less to serve <span className="faint">(estimated)</span></span>
              : <span>{reasoner.name} runs at B = {count(Math.max(pa.budget, pb.budget))} only</span>}
            {!isDefault && <button type="button" className="res-reset" onClick={() => setPair(['phi4-14b', claims.topLarge])}>Reset</button>}
          </p>
        </div>

        <JudgeStrip pair={[a, b]} />

        <div className="hero-actions">
          <a className="btn btn-primary btn-lg" href="#/findings">Read the findings <Icon.arrow className="arrow" /></a>
          <a className="hero-link" href="#/leaderboard">Leaderboard <Icon.arrow className="arrow" /></a>
        </div>
      </div>
    </section>
  )
}
