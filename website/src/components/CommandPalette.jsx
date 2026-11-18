import { useEffect, useMemo, useRef, useState } from 'react'
import { copyText, Icon, MarkIcon } from './ui'
import { core, judges, meta, pointOf } from '../lib/data'
import { FINDINGS } from '../lib/findings'
import { loadDocIndex } from '../lib/docs'
import { setTheme, getTheme } from '../lib/theme'
import { fmt } from '../lib/format'

const KIND_ICON = { page: Icon.book, section: Icon.hash, script: Icon.terminal, flag: Icon.terminal, module: Icon.cube,
  class: Icon.cube, function: Icon.cube, method: Icon.cube, prompt: Icon.paper, field: Icon.table, model: Icon.cube, dataset: Icon.table, action: Icon.chevron }
const GROUP_ORDER = ['Pages', 'Findings', 'Judges', 'Datasets', 'Docs', 'Command line', 'Python API', 'Prompts and config', 'Actions']

function baseItems() {
  const items = [
    { group: 'Pages', title: 'Overview', sub: 'The question and the main result', href: '#/', kind: 'page' },
    { group: 'Pages', title: 'Findings', sub: 'Six findings, each a short scroll story', href: '#/findings', kind: 'page' },
    { group: 'Pages', title: 'Leaderboard', sub: 'Every judge, sortable, with a compare view', href: '#/leaderboard', kind: 'page' },
    { group: 'Pages', title: 'Pick a judge', sub: 'Which judge to run for your task and budget', href: '#/pick', kind: 'page' },
    { group: 'Pages', title: 'Method', sub: 'Pipeline, metrics, datasets and models', href: '#/method', kind: 'page' },
    { group: 'Pages', title: 'Paper and citation', sub: 'BibTeX, authors and contact', href: '#/cite', kind: 'page' },
  ]
  FINDINGS.forEach((f, i) => items.push({ group: 'Findings', title: f.title, sub: `Finding ${i + 1}`, href: `#/findings/${f.slug}`, kind: 'section' }))
  judges.forEach((j) => items.push({
    group: 'Judges', title: j.name, sub: `${j.size} · ${j.kind === 'ladder' ? 'ladder judge' : `${j.kind} judge`} · ${j.weights}`,
    meta: pointOf[j.key] ? `${fmt(pointOf[j.key].acc)}%` : '', href: `#/judge/${j.key}`, glyph: j.family, keys: j.key,
  }))
  core.datasets.closed.forEach((d) => items.push({ group: 'Datasets', title: d.name, sub: `${d.domain} · ${d.answer} · ${d.items.toLocaleString('en-US')} items per student`, href: '#/method/datasets', kind: 'dataset', keys: d.key }))
  ;['SummEval', 'MT-Bench'].forEach((n) => items.push({ group: 'Datasets', title: n, sub: 'open-ended scoring', href: '#/method/datasets', kind: 'dataset' }))
  items.push(
    { group: 'Actions', title: 'Switch theme', sub: 'Light or dark', kind: 'action', run: () => setTheme(getTheme() === 'dark' ? 'light' : 'dark') },
    { group: 'Actions', title: 'Copy the BibTeX', sub: 'Cite the paper', kind: 'action', run: () => copyText(meta.bibtex) },
    { group: 'Actions', title: 'Copy pip install slmjury', sub: 'Install the package', kind: 'action', run: () => copyText('pip install slmjury') },
    { group: 'Actions', title: 'Open the code on GitHub', sub: meta.code, kind: 'action', run: () => window.open(meta.code, '_blank', 'noopener') },
  )
  return items
}

function docItems(index) {
  const pageTitle = Object.fromEntries(index.pages.map((p) => [p.slug, p.title]))
  return index.search.map((s) => {
    const group = s.page.startsWith('cli-') ? 'Command line' : s.page.startsWith('api-') ? 'Python API'
      : ['prompts', 'models'].includes(s.page) ? 'Prompts and config' : 'Docs'
    return {
      group, kind: s.kind, title: s.title,
      sub: s.kind === 'page' ? s.text : `${pageTitle[s.page]}${s.text ? ` · ${s.text}` : ''}`,
      href: s.id ? `#/docs/${s.page}/${s.id}` : `#/docs/${s.page}`,
    }
  })
}

// Every query word must appear; titles, prefixes and word starts score higher.
function score(item, words) {
  const t = item.title.toLowerCase()
  const hay = `${t} ${(item.sub || '').toLowerCase()} ${(item.keys || '').toLowerCase()}`
  let s = 0
  for (const w of words) {
    const i = hay.indexOf(w)
    if (i < 0) return -1
    const ti = t.indexOf(w)
    if (ti === 0) s += 12
    else if (ti > 0) s += /[\s\-_.(/]/.test(t[ti - 1]) ? 8 : 5
    else s += 1
  }
  if (item.kind === 'page') s += 3
  if (item.group === 'Judges' || item.group === 'Findings') s += 2
  return s - t.length / 200
}

function highlightTitle(title, words) {
  if (!words.length) return title
  const re = new RegExp(`(${words.map((w) => w.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')).join('|')})`, 'ig')
  return title.split(re).map((p, i) => (i % 2 ? <mark key={i}>{p}</mark> : p))
}

export default function CommandPalette({ onClose }) {
  const [q, setQ] = useState('')
  const [docs, setDocs] = useState([])
  const [sel, setSel] = useState(0)
  const input = useRef(null)
  const list = useRef(null)
  const restoreFocus = useRef(typeof document !== 'undefined' ? document.activeElement : null)
  const base = useMemo(baseItems, [])
  useEffect(() => { loadDocIndex().then((idx) => idx && setDocs(docItems(idx))) }, [])
  useEffect(() => {
    input.current?.focus()
    document.documentElement.classList.add('lock')
    return () => { document.documentElement.classList.remove('lock'); restoreFocus.current?.focus?.() }
  }, [])

  const words = q.trim().toLowerCase().split(/\s+/).filter(Boolean)
  const results = useMemo(() => {
    if (!words.length) {
      return base.filter((x) => x.group === 'Pages' || x.group === 'Findings' || x.group === 'Actions' ||
        (x.group === 'Judges' && ['phi4-14b', 'qwen3.5-122b', 'qwen3-14b', 'qwen2.5-72b'].includes(x.keys)))
        .concat(docs.filter((d) => d.kind === 'page').slice(0, 6))
    }
    const all = base.concat(docs)
    return all.map((x) => [x, score(x, words)]).filter(([, s]) => s >= 0).sort((a, b) => b[1] - a[1]).slice(0, 60).map(([x]) => x)
  }, [q, docs])
  const grouped = useMemo(() => {
    const g = {}
    results.forEach((r) => { (g[r.group] ||= []).push(r) })
    const order = words.length ? [...new Set(results.map((r) => r.group))] : GROUP_ORDER
    return order.filter((k) => g[k]).map((k) => [k, g[k]])
  }, [results])
  const flat = grouped.flatMap(([, items]) => items)
  useEffect(() => setSel(0), [q])
  useEffect(() => {
    list.current?.querySelector('[aria-selected="true"]')?.scrollIntoView({ block: 'nearest' })
  }, [sel])

  const choose = (item) => {
    if (!item) return
    onClose()
    if (item.run) item.run()
    else if (item.href) window.location.hash = item.href.slice(1)
  }
  const onKey = (e) => {
    if (e.key === 'ArrowDown') { e.preventDefault(); setSel((s) => Math.min(flat.length - 1, s + 1)) }
    else if (e.key === 'ArrowUp') { e.preventDefault(); setSel((s) => Math.max(0, s - 1)) }
    else if (e.key === 'Enter') { e.preventDefault(); choose(flat[sel]) }
    else if (e.key === 'Escape') { e.preventDefault(); onClose() }
    else if (e.key === 'Tab') { e.preventDefault() }
  }
  let n = -1
  return (
    <>
      <div className="palette-scrim" onClick={onClose} />
      <div className="palette" role="dialog" aria-modal="true" aria-label="Search the site" onKeyDown={onKey}>
        <div className="palette-input">
          <Icon.search />
          <input ref={input} value={q} onChange={(e) => setQ(e.target.value)} placeholder="Search judges, findings, docs, flags, prompts…"
            role="combobox" aria-expanded="true" aria-controls="palette-list" aria-activedescendant={flat[sel] ? `pi-${sel}` : undefined}
            aria-autocomplete="list" spellCheck={false} autoComplete="off" />
          <kbd>esc</kbd>
        </div>
        <div className="palette-list" id="palette-list" role="listbox" ref={list} aria-label="Results">
          {!flat.length && <div className="palette-empty">Nothing matches “{q}”. Try a judge name, a flag such as <code>--max-tokens</code>, or a function.</div>}
          {grouped.map(([g, items]) => (
            <div key={g} role="presentation">
              <div className="palette-group" role="presentation">{g}</div>
              {items.map((it) => {
                n += 1
                const idx = n
                const KI = KIND_ICON[it.kind] || Icon.chevron
                return (
                  <div key={`${g}-${it.title}-${it.href || ''}-${idx}`} id={`pi-${idx}`} role="option" aria-selected={idx === sel}
                    className="palette-item" onMouseMove={() => sel !== idx && setSel(idx)} onClick={() => choose(it)}>
                    <span className="pi-icon">{it.glyph ? <MarkIcon fam={it.glyph} size={13} /> : <KI />}</span>
                    <span style={{ minWidth: 0 }}>
                      <span className="pi-title">{highlightTitle(it.title, words)}</span>
                      {it.sub && <span className="pi-sub">{it.sub}</span>}
                    </span>
                    <span className="pi-meta">{it.meta}</span>
                  </div>
                )
              })}
            </div>
          ))}
        </div>
        <div className="palette-foot">
          <span><kbd>↑</kbd><kbd>↓</kbd> move</span>
          <span><kbd>↵</kbd> open</span>
          <span><kbd>esc</kbd> close</span>
          <span style={{ marginLeft: 'auto' }}>{base.length + docs.length} entries</span>
        </div>
      </div>
    </>
  )
}
