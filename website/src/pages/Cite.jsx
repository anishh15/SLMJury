import { Code, Icon } from '../components/ui'
import { AFFILIATIONS, AUTHORS, EMAILS, meta } from '../lib/data'
import { FINDINGS } from '../lib/findings'
import '../styles/content.css'
import '../styles/cite.css'

export default function Cite() {
  const id = meta.arxiv.split('/').pop()
  return (
    <>
      <section className="page-head cite-head">
        <div className="wrap wrap-narrow">
          <span className="venue-chip" title={meta.venue.label}>{meta.venue.chip}</span>
          <h1 className="h-display">SLMJury: Can Small Language Models Judge as Well as Large Ones?</h1>
          <p className="authors">
            {AUTHORS.map((a, i) => (
              <span key={a.name}>
                {a.href ? <a href={a.href} target="_blank" rel="noreferrer">{a.name}</a> : <strong>{a.name}</strong>}
                <sup>{a.aff}</sup>{i < AUTHORS.length - 1 ? ' · ' : ''}
              </span>
            ))}
            <br />
            <span className="faint">{AFFILIATIONS.map((a, i) => <span key={a}><sup>{i + 1}</sup> {a}{i < AFFILIATIONS.length - 1 ? ' · ' : ''}</span>)}</span>
          </p>
          <div className="row cite-actions">
            <a className="btn btn-primary" href={meta.arxiv} target="_blank" rel="noreferrer"><Icon.paper /> arXiv {id}</a>
            <a className="btn" href={meta.code} target="_blank" rel="noreferrer"><Icon.github /> Code</a>
            <a className="btn" href={meta.pypi} target="_blank" rel="noreferrer"><Icon.pypi /> PyPI</a>
          </div>
        </div>
      </section>
      <section className="wrap wrap-narrow cite-body">
        <Code code={meta.bibtex} lang="bibtex" title="BibTeX" />
        <h2 className="h3 cite-sub" data-reveal>The six findings</h2>
        <ol className="cite-findings" data-reveal="stagger">
          {FINDINGS.map((f, i) => (
            <li key={f.slug}>
              <a href={`#/findings/${f.slug}`}>
                <span className="ci-n tnum">{String(i + 1).padStart(2, '0')}</span>
                <span className="ci-t">{f.title}</span>
                <span className="ci-b tnum">{f.big}</span>
                <Icon.arrow className="ci-go" />
              </a>
            </li>
          ))}
        </ol>
        <h2 className="h3 cite-sub" data-reveal>Contact</h2>
        <p className="cite-contact" data-reveal>
          {EMAILS.map((e, i) => <span key={e}><a href={`mailto:${e}`}>{e}</a>{i < EMAILS.length - 1 ? ' · ' : ''}</span>)}
          <br />
          <span className="muted">Bug reports and new judges are welcome on <a href={`${meta.code}/issues`} target="_blank" rel="noreferrer">GitHub Issues</a>.</span>
        </p>
      </section>
    </>
  )
}
