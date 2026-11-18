import { useState } from 'react'
import { Code, Icon } from '../components/ui'
import { meta } from '../lib/data'

// The paper in one line, with a Cite button that opens the BibTeX.
export default function CiteBlock() {
  const [open, setOpen] = useState(false)
  const id = meta.arxiv.split('/').pop()
  return (
    <section className="section cite-block" aria-labelledby="cite-title">
      <div className="wrap-wide">
        <div className="cite-row">
          <div>
            <span className="venue-chip" title={meta.venue.label}>{meta.venue.chip}</span>
            <h2 className="cite-paper" id="cite-title">SLMJury: Can Small Language Models Judge as Well as Large Ones?</h2>
          </div>
          <div className="row">
            <a className="btn btn-primary" href={meta.arxiv} target="_blank" rel="noreferrer"><Icon.paper /> arXiv {id}</a>
            <button type="button" className="btn" aria-expanded={open} aria-controls="cite-bibtex" onClick={() => setOpen((o) => !o)}>
              Cite <Icon.chevronDown className={`cite-chev${open ? ' up' : ''}`} />
            </button>
            <a className="btn btn-ghost" href={meta.code} target="_blank" rel="noreferrer"><Icon.github /> Code</a>
          </div>
        </div>
        <div id="cite-bibtex" className="cite-bib" hidden={!open}>
          {open && <Code code={meta.bibtex} lang="bibtex" title="BibTeX" />}
        </div>
      </div>
    </section>
  )
}
