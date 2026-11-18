import { useEffect, useRef, useState } from 'react'
import { copyText, Icon } from '../components/ui'
import { chartLink, download, exportPNG, slugify, toCSV } from '../lib/export'
import { useNear } from '../lib/motion'

// One frame for every chart: title, a small toolbar (table view, CSV, PNG,
// link), optional controls, the chart, and a footer.
// table: () => ({ columns: [{ key, label, align }], rows: [{...}] })

const SOURCE = 'SLMJury · anishh15.github.io/SLMJury'
const decimals = (v) => (typeof v === 'number' && Number.isFinite(v) ? (String(v).split('.')[1] || '').length : 0)
const fmtCell = (v, nd) => (typeof v === 'number' ? v.toLocaleString('en-US', { minimumFractionDigits: nd, maximumFractionDigits: nd }) : v ?? '—')

export function DataTable({ columns, rows, caption }) {
  // each column shows as many decimals as its most precise value (at most three)
  const nd = Object.fromEntries(columns.map((c) => [c.key, Math.min(3, Math.max(0, ...rows.map((r) => decimals(r[c.key]))))]))
  return (
    <div className="cf-table-wrap" tabIndex={0} role="region" aria-label={`${caption} as a table`}>
      <table className="dtable cf-table">
        <thead>
          <tr>{columns.map((c) => <th key={c.key} className={c.align === 'left' ? '' : 'r'}>{c.label}</th>)}</tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={i}>{columns.map((c) => <td key={c.key} className={c.align === 'left' ? '' : 'r'}>{fmtCell(r[c.key], nd[c.key])}</td>)}</tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

function Tool({ label, onClick, pressed, children, done }) {
  return (
    <button type="button" className={`cf-tool${done ? ' done' : ''}`} onClick={onClick} aria-pressed={pressed} aria-label={label} data-tip={done ? 'Done' : label}>
      {done ? <Icon.check /> : children}
    </button>
  )
}

// The toolbar on its own, for charts that draw their own frame.
export function Toolbar({ id, title, sub, table, view, setView, body }) {
  const [flash, setFlash] = useState(null)
  const timer = useRef(0)
  useEffect(() => () => clearTimeout(timer.current), [])
  const ok = (k) => { setFlash(k); clearTimeout(timer.current); timer.current = setTimeout(() => setFlash(null), 1400) }
  const name = slugify(id || title)
  const data = () => (typeof table === 'function' ? table() : table)
  const onCSV = () => {
    const t = data()
    if (!t) return
    download(`slmjury-${name}.csv`, toCSV(t.columns, t.rows), 'text/csv')
    ok('csv')
  }
  const onPNG = async () => {
    let svg = body.current?.querySelector('svg[role="img"]')
    if (!svg && view === 'table') { setView('chart'); return }
    if (!svg) return
    await exportPNG(svg, { title, sub: typeof sub === 'string' ? sub : '', source: SOURCE, filename: `slmjury-${name}.png` })
    ok('png')
  }
  const onLink = async () => {
    await copyText(chartLink(id))
    ok('link')
  }
  return (
    <div className="cf-tools" role="toolbar" aria-label="Chart tools">
      {table && (
        <Tool label={view === 'table' ? 'Show the chart' : 'Show as a table'} pressed={view === 'table'} onClick={() => setView(view === 'table' ? 'chart' : 'table')}>
          {view === 'table' ? <Icon.chart /> : <Icon.table />}
        </Tool>
      )}
      {table && <Tool label="Download CSV" onClick={onCSV} done={flash === 'csv'}><Icon.download /></Tool>}
      <Tool label="Save as PNG" onClick={onPNG} done={flash === 'png'}><Icon.image /></Tool>
      {id && <Tool label="Copy link" onClick={onLink} done={flash === 'link'}><Icon.link /></Tool>}
    </div>
  )
}

// Mounts a chart only when it comes near the viewport; the placeholder has
// the chart's height, so nothing moves when it arrives.
export function Lazy({ height, children }) {
  const [ref, near] = useNear('40% 0px 40% 0px')
  return <div ref={ref}>{near ? children : <div className="stage-ph" style={{ height }} />}</div>
}

export default function ChartFrame({ id, title, sub, controls, legend, caption, table, children, className = '', head }) {
  const [view, setView] = useState('chart')
  const body = useRef(null)
  const t = view === 'table' && table ? (typeof table === 'function' ? table() : table) : null
  return (
    <figure className={`cf ${className}`} id={id ? `chart-${id}` : undefined}>
      <div className="cf-top">
        <div className="cf-titles">
          {head}
          <div className="cf-title">{title}</div>
          {sub && <div className="cf-sub">{sub}</div>}
        </div>
        <Toolbar id={id} title={title} sub={sub} table={table} view={view} setView={setView} body={body} />
      </div>
      {controls && <div className="cf-controls">{controls}</div>}
      <div className="cf-body" ref={body}>
        {t ? <DataTable columns={t.columns} rows={t.rows} caption={title} /> : children}
      </div>
      {(legend || caption) && (
        <figcaption className="cf-foot">
          {legend}
          {caption && <div className="cf-cap">{caption}</div>}
        </figcaption>
      )}
    </figure>
  )
}
