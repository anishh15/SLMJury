import { useEffect, useMemo, useRef, useState } from 'react'
import { Code, CopyButton, Icon, copyText } from '../components/ui'
import { loadDocIndex, peekDoc, useDoc } from '../lib/docs'
import { replaceHash } from '../lib/router'
import '../styles/content.css'
import '../styles/docs.css'

const GENERATED = {
  'cli-pipeline': "Generated from each script's argparse parser.",
  'cli-experiments': "Generated from each script's argparse parser.",
  'cli-analyses': "Generated from each script's argparse parser.",
  'api-core': "Generated from the package's docstrings.",
  'api-parsers': "Generated from the package's docstrings.",
  'api-strategies': "Generated from the package's docstrings.",
  'api-analysis': "Generated from the package's docstrings.",
  prompts: 'Read from the package and prompt_variants.yaml.',
  models: 'Read from slmjury/configs/models.yaml.',
  reproduce: 'From docs/REPRODUCE.md in the repository.',
}

const Html = ({ as: Tag = 'p', html, className }) => <Tag className={className} dangerouslySetInnerHTML={{ __html: html }} />

function Anchor({ id, page }) {
  return (
    <a className="anchor" href={`#/docs/${page}/${id}`} aria-label="Link to this section"
      onClick={(e) => { e.preventDefault(); replaceHash(`/docs/${page}/${id}`); copyText(window.location.href); document.getElementById(id)?.scrollIntoView({ behavior: 'smooth', block: 'start' }) }}>
      <Icon.link />
    </a>
  )
}

function PromptText({ text, fields }) {
  const parts = text.split(/(⟦[^⟧]+⟧)/)
  return (
    <pre className="prompt-text" tabIndex={0}>
      {parts.map((p, i) => (p.startsWith('⟦') ? <span key={i} className="field" title="Filled in for each item">{p.slice(1, -1)}</span> : p))}
    </pre>
  )
}
const rawPrompt = (text) => text.replace(/⟦([^⟧]+)⟧/g, '{$1}')

function Prompt({ b, page }) {
  return (
    <section className="prompt" id={b.id}>
      <div className="prompt-head">
        <h3>{b.title}<Anchor id={b.id} page={page} /></h3>
        <CopyButton text={rawPrompt(b.text)} className="copy-btn copy-light" label="Copy" />
      </div>
      <div className="prompt-meta"><span>{b.when}</span><a href={b.href} target="_blank" rel="noreferrer"><code>{b.source}</code></a></div>
      <PromptText text={b.text} fields={b.fields} />
    </section>
  )
}

function Variants({ b }) {
  const [tab, setTab] = useState(0)
  const t = b.tabs[tab]
  const LABEL = { quick: 'Quick verdict', reasoned: 'Reasoned verdict', summeval: 'SummEval scoring', mtbench: 'MT-Bench scoring' }
  return (
    <div className="variants" id={b.id}>
      <div className="variant-tabs" role="tablist" aria-label="Prompt variants">
        {b.tabs.map((x, i) => (
          <button key={x.name} type="button" role="tab" aria-selected={i === tab} onClick={() => setTab(i)}>{x.name.replace(/_/g, ' ')}</button>
        ))}
      </div>
      <div role="tabpanel" className="variant-panel">
        {t.entries.map((e) => (
          <div key={e.key} className="variant-entry">
            <div className="prompt-head"><h4>{LABEL[e.key] || e.key}</h4><CopyButton text={rawPrompt(e.text)} className="copy-btn copy-light" label="Copy" /></div>
            <PromptText text={e.text} fields={e.fields} />
          </div>
        ))}
      </div>
    </div>
  )
}

function Cli({ b, page }) {
  return (
    <section className="cli" id={b.id}>
      <div className="cli-head">
        <h3><code>{b.path}</code><Anchor id={b.id} page={page} /></h3>
        <a className="src" href={b.href} target="_blank" rel="noreferrer">Source <Icon.arrowUpRight /></a>
      </div>
      {b.lead && <Html html={b.lead} className="cli-lead" />}
      {b.about.length > 0 && (
        <details className="more">
          <summary>From the docstring</summary>
          <Blocks blocks={b.about} page={page} />
        </details>
      )}
      <Code code={b.usage} lang="bash" title="usage" />
      {b.groups.map((g) => (
        <div key={g.title} className="opts-wrap">
          {b.groups.length > 1 && <div className="opts-title">{g.title}</div>}
          <table className="opts">
            <thead><tr><th>Option</th><th>Default</th><th>What it does</th></tr></thead>
            <tbody>
              {g.actions.map((a) => (
                <tr key={a.flags.join()}>
                  <td className="opt-flag">
                    {a.flags.map((f, i) => <span key={f}><code>{f}</code>{i < a.flags.length - 1 ? ' ' : ''}</span>)}
                    {a.metavar && <span className="metavar"> {a.metavar}</span>}
                    {a.required && <span className="req">required</span>}
                  </td>
                  <td className="opt-def">{a.default != null && a.default !== '' ? <code>{a.default}</code> : <span className="faint">—</span>}</td>
                  <td className="opt-help">
                    {a.help}
                    {a.choices && <span className="choices">{a.choices.map((c) => <code key={c}>{c}</code>)}</span>}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ))}
      {b.exclusive.length > 0 && <p className="tiny muted">Use one of {b.exclusive.map((e) => e.map((f) => <code key={f}>{f}</code>).reduce((acc, x) => [...acc, ', ', x], []).slice(1)).map((x, i) => <span key={i}>{x}</span>)} at a time.</p>}
    </section>
  )
}

function Args({ args }) {
  if (!args?.length) return null
  return (
    <dl className="args">
      {args.map((a) => (
        <div key={a.name}><dt><code>{a.name}</code>{a.type && <span className="faint"> {a.type}</span>}</dt><dd dangerouslySetInnerHTML={{ __html: a.html }} /></div>
      ))}
    </dl>
  )
}

function Api({ b, page }) {
  return (
    <section className={`api api-${b.kind}`} id={b.id}>
      <div className="api-sig">
        <span className="api-kind">{b.kind === 'class' ? 'class' : 'def'}</span>
        <code className="api-name">{b.name}</code>
        <Anchor id={b.id} page={page} />
        <a className="src" href={b.href} target="_blank" rel="noreferrer">Source <Icon.arrowUpRight /></a>
      </div>
      <code className="api-params">{b.sig}</code>
      {b.summary && <Html html={b.summary} className="api-summary" />}
      <Blocks blocks={b.body} page={page} />
      <Args args={b.args} />
      {b.returns && <p className="api-returns"><span className="lbl-k">Returns</span> <span dangerouslySetInnerHTML={{ __html: b.returns }} /></p>}
      {b.raises?.length > 0 && <div className="api-raises"><span className="lbl-k">Raises</span><Args args={b.raises} /></div>}
      {b.methods?.length > 0 && (
        <div className="methods">
          {b.methods.map((mm) => (
            <div key={mm.name} className="method" id={mm.id}>
              <code className="api-name sm">{mm.kind === 'property' ? '' : '.'}{mm.name}</code>
              {mm.sig && <code className="api-params sm">{mm.sig}</code>}
              {mm.kind !== 'method' && <span className="pill tiny-pill">{mm.kind}</span>}
              {mm.summary && <Html html={mm.summary} className="api-summary" />}
              <Blocks blocks={mm.body} page={page} />
              <Args args={mm.args} />
              {mm.returns && <p className="api-returns"><span className="lbl-k">Returns</span> <span dangerouslySetInnerHTML={{ __html: mm.returns }} /></p>}
            </div>
          ))}
        </div>
      )}
    </section>
  )
}

function Blocks({ blocks, page }) {
  return blocks.map((b, i) => {
    switch (b.t) {
      case 'h2': return <h2 key={i} id={b.id} className="doc-h2">{b.text}<Anchor id={b.id} page={page} /></h2>
      case 'h3': return <h3 key={i} id={b.id} className="doc-h3">{b.text}<Anchor id={b.id} page={page} /></h3>
      case 'h4': case 'h5': return <h4 key={i} className="doc-h4">{b.text}</h4>
      case 'p': return <Html key={i} html={b.html} />
      case 'note': return <Html key={i} html={b.html} className="doc-note" />
      case 'ul': return <ul key={i}>{b.items.map((x, k) => <li key={k} dangerouslySetInnerHTML={{ __html: x }} />)}</ul>
      case 'ol': return <ol key={i}>{b.items.map((x, k) => <li key={k} dangerouslySetInnerHTML={{ __html: x }} />)}</ol>
      case 'code': return <Code key={i} code={b.code} lang={b.lang === 'text' ? 'text' : b.lang} title={b.lang} />
      case 'pre': return <pre key={i} className="doc-pre" tabIndex={0}>{b.code}</pre>
      case 'snippet': return (
        <div key={i} className="doc-snippet">
          <div className="snip-title">{b.title}<span>{b.blurb}</span></div>
          <Code code={b.code} lang="python" title={b.source.split('/').pop()} output={b.output} />
        </div>
      )
      case 'table': return (
        <div key={i} className="table-wrap doc-table">
          <table className="dtable">
            <thead><tr>{b.head.map((h, k) => <th key={k} className={b.align[k] === 'r' ? 'r' : ''} dangerouslySetInnerHTML={{ __html: h }} />)}</tr></thead>
            <tbody>{b.rows.map((r, k) => <tr key={k}>{r.map((c, m) => <td key={m} className={b.align[m] === 'r' ? 'r' : ''} dangerouslySetInnerHTML={{ __html: c }} />)}</tr>)}</tbody>
          </table>
        </div>
      )
      case 'cli': return <Cli key={i} b={b} page={page} />
      case 'api': return <Api key={i} b={b} page={page} />
      case 'module': return (
        <section key={i} className="module" id={b.id}>
          <h2 className="doc-h2 mono-h"><code>{b.name}</code><Anchor id={b.id} page={page} /><a className="src" href={b.href} target="_blank" rel="noreferrer">Source <Icon.arrowUpRight /></a></h2>
          {b.lead && <Html html={b.lead} className="cli-lead" />}
          {b.about.length > 0 && <details className="more"><summary>Module notes</summary><Blocks blocks={b.about} page={page} /></details>}
        </section>
      )
      case 'reexports': return (
        <div key={i} className="reexports">
          {b.items.map((x) => {
            const pg = x.home.startsWith('slmjury.parsers') ? 'api-parsers' : x.home.startsWith('slmjury.strategies') ? 'api-strategies' : x.home.startsWith('slmjury.analysis') ? 'api-analysis' : 'api-core'
            const id = `${x.home}.${x.name}`.toLowerCase().replace(/[`*_]/g, '').replace(/[^a-z0-9]+/g, '-')
            return <a key={x.name} href={`#/docs/${pg}/${id}`}><code>{x.name}</code><span>{x.home}</span></a>
          })}
        </div>
      )
      case 'prompt': return <Prompt key={i} b={b} page={page} />
      case 'variants': return <Variants key={i} b={b} />
      default: return null
    }
  })
}

function useSpy(ids, deps) {
  const [active, setActive] = useState(null)
  useEffect(() => {
    if (!ids.length || !('IntersectionObserver' in window)) return
    const els = ids.map((id) => document.getElementById(id)).filter(Boolean)
    const seen = new Map()
    const io = new IntersectionObserver((entries) => {
      entries.forEach((e) => seen.set(e.target.id, e.isIntersecting ? e.boundingClientRect.top : null))
      const vis = [...seen].filter(([, t]) => t != null).sort((a, b) => a[1] - b[1])
      if (vis.length) setActive(vis[0][0])
    }, { rootMargin: '-80px 0px -65% 0px' })
    els.forEach((el) => io.observe(el))
    return () => io.disconnect()
  }, deps)
  return active
}

export default function Docs({ route }) {
  const slug = route.parts[1] || 'getting-started'
  const target = route.parts[2]
  const doc = useDoc(slug)
  const [index, setIndex] = useState(() => peekDoc('index'))
  const [menu, setMenu] = useState(false)
  useEffect(() => { loadDocIndex().then(setIndex) }, [])
  useEffect(() => setMenu(false), [slug])
  useEffect(() => {
    if (!doc || !target) return
    const el = document.getElementById(target)
    if (el) requestAnimationFrame(() => el.scrollIntoView({ block: 'start' }))
  }, [doc, target])
  const tocIds = useMemo(() => (doc ? doc.toc.filter((t) => t.level === 2 || doc.toc.length < 40).map((t) => t.id) : []), [doc])
  const active = useSpy(tocIds, [doc])
  const pages = index?.pages || []
  const i = pages.findIndex((p) => p.slug === slug)
  const prev = i > 0 ? pages[i - 1] : null
  const next = i >= 0 && i < pages.length - 1 ? pages[i + 1] : null
  const groups = useMemo(() => {
    const g = []
    pages.forEach((p) => { const last = g[g.length - 1]; if (last && last[0] === p.group) last[1].push(p); else g.push([p.group, [p]]) })
    return g
  }, [pages])
  const nav = (
    <nav className="docs-nav" aria-label="Docs">
      <button type="button" className="docs-search" onClick={() => window.dispatchEvent(new Event('slmjury:search'))}>
        <Icon.search /> <span>Search the docs</span> <kbd>/</kbd>
      </button>
      {groups.map(([g, ps]) => (
        <div key={g} className="dn-group">
          <div className="dn-title">{g}</div>
          {ps.map((p) => <a key={p.slug} href={`#/docs/${p.slug}`} aria-current={p.slug === slug ? 'page' : undefined}>{p.title}</a>)}
        </div>
      ))}
    </nav>
  )
  if (doc === null && pages.length && i < 0) {
    return (
      <section className="wrap notfound">
        <h1 className="h2">No docs page called <code>{slug}</code></h1>
        <p><a className="btn btn-primary" href="#/docs/getting-started">Getting started <Icon.arrow /></a></p>
      </section>
    )
  }
  const shortToc = doc ? doc.toc.filter((t) => tocIds.includes(t.id)) : []
  return (
    <div className="wrap-wide docs">
      <aside className="docs-side">{nav}</aside>
      <div className="docs-mobile-bar">
        <button type="button" className="btn btn-sm" aria-expanded={menu} onClick={() => setMenu((m) => !m)}>
          <Icon.book /> {doc?.group}: {doc?.title || 'Docs'} <Icon.chevronDown />
        </button>
        {menu && <div className="docs-mobile-menu">{nav}</div>}
      </div>
      <article className="docs-main">
        {!doc ? <div className="skeleton" style={{ height: 480 }} /> : (
          <>
            <header className="doc-head">
              <h1 className="doc-title">{doc.title}</h1>
              <p className="lead">{doc.blurb}</p>
            </header>
            <div className="doc-body">
              <Blocks blocks={doc.blocks} page={slug} />
            </div>
            {GENERATED[slug] && <p className="gen-note">{GENERATED[slug]}</p>}
            <nav className="doc-pager" aria-label="Previous and next page">
              {prev ? <a href={`#/docs/${prev.slug}`} className="pager prev"><span>Previous</span>{prev.title}</a> : <span />}
              {next ? <a href={`#/docs/${next.slug}`} className="pager next"><span>Next</span>{next.title}</a> : <span />}
            </nav>
          </>
        )}
      </article>
      <aside className="docs-toc" aria-label="On this page">
        {shortToc.length > 0 && (
          <>
            <div className="dn-title">On this page</div>
            <div className="toc-scroll">
              {shortToc.map((t) => (
                <a key={t.id} href={`#/docs/${slug}/${t.id}`} className={`lvl-${t.level}${active === t.id ? ' active' : ''}`}
                  onClick={(e) => { e.preventDefault(); replaceHash(`/docs/${slug}/${t.id}`); document.getElementById(t.id)?.scrollIntoView({ behavior: window.matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth', block: 'start' }) }}>
                  {t.text}
                </a>
              ))}
            </div>
          </>
        )}
      </aside>
    </div>
  )
}
