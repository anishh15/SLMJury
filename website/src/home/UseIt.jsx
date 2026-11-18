import { useState } from 'react'
import { Code, CopyButton } from '../components/ui'
import snippets from '../data/snippets.json'

// Tasks on the left, code and the output it really printed on the right.
export default function UseIt() {
  const [i, setI] = useState(0)
  const s = snippets[i]
  return (
    <section className="section useit" aria-labelledby="useit-title">
      <div className="wrap-wide useit-grid">
        <div className="useit-side">
          <h2 className="h2" id="useit-title">Use it on a laptop</h2>
          <p className="dek">No GPU needed. Each output was printed by its code when the site was built.</p>
          <div className="install">
            <span className="prompt" aria-hidden="true">$</span><code>pip install slmjury</code>
            <CopyButton text="pip install slmjury" iconOnly className="copy-btn copy-light" label="Copy the install command" />
          </div>
          <div className="tasks" role="tablist" aria-label="Examples">
            {snippets.map((x, k) => (
              <button key={x.slug} type="button" role="tab" aria-selected={k === i} aria-controls="useit-panel" id={`task-${x.slug}`}
                className="task" onClick={() => setI(k)}>
                <span className="task-title">{x.title}</span>
                <span className="task-blurb">{x.blurb}</span>
              </button>
            ))}
          </div>
        </div>
        <div className="useit-panel" role="tabpanel" id="useit-panel" aria-labelledby={`task-${s.slug}`}>
          <Code key={s.slug} code={s.code} lang="python" title={s.source.split('/').pop()} output={s.output} />
          <a className="link-btn useit-more" href="#/docs/getting-started">More in Getting started →</a>
        </div>
      </div>
    </section>
  )
}
