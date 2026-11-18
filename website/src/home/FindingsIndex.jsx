import { Icon } from '../components/ui'
import { FINDINGS } from '../lib/findings'
import { useInView } from '../lib/motion'
import { MINIS } from './Minis'

function Row({ f, i }) {
  const [ref, seen] = useInView({ threshold: 0.3 })
  const Mini = MINIS[f.slug]
  return (
    <li ref={ref} className={`f-row${seen ? ' in' : ''}`}>
      <a href={`#/findings/${f.slug}`} className="f-link">
        <span className="f-n tnum">{String(i + 1).padStart(2, '0')}</span>
        <span className="f-text">
          <span className="f-title">{f.title}</span>
        </span>
        <span className="f-big tnum">{f.big}</span>
        <span className="f-mini"><Mini /></span>
        <span className="f-go" aria-hidden="true"><Icon.arrow /></span>
      </a>
    </li>
  )
}

export default function FindingsIndex() {
  return (
    <section className="section findings-index" aria-labelledby="fi-title">
      <div className="wrap-wide fi-grid">
        <div className="fi-side" data-reveal="stagger">
          <h2 className="h2" id="fi-title">Six findings</h2>
          <a className="btn" href="#/findings">All findings <Icon.arrow className="arrow" /></a>
        </div>
        <ol className="f-list" data-reveal="stagger">
          {FINDINGS.map((f, i) => <Row key={f.slug} f={f} i={i} />)}
        </ol>
      </div>
    </section>
  )
}
