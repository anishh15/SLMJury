import { useEffect, useLayoutEffect, useRef, useState } from 'react'
import { famColor } from '../lib/data'

// ── Family marks: the paper's shapes ────────────────────────────────────────
// Llama triangle, Qwen2.5 square, Qwen3 circle, Phi-4 diamond, large hexagon.
export function shapePath(fam, r) {
  switch (fam) {
    case 'Llama': {
      const h = r * 1.18
      return `M0,${-h}L${h * 0.98},${h * 0.66}L${-h * 0.98},${h * 0.66}Z`
    }
    case 'Qwen2.5': {
      const s = r * 0.86
      return `M${-s},${-s}H${s}V${s}H${-s}Z`
    }
    case 'Phi-4': {
      const s = r * 1.2
      return `M0,${-s}L${s},0L0,${s}L${-s},0Z`
    }
    case 'Large': {
      const s = r * 1.08
      const pts = Array.from({ length: 6 }, (_, i) => {
        const a = (Math.PI / 3) * i + Math.PI / 6
        return `${(Math.cos(a) * s).toFixed(2)},${(Math.sin(a) * s).toFixed(2)}`
      })
      return `M${pts.join('L')}Z`
    }
    default:
      return `M${r},0A${r},${r} 0 1,1 ${-r},0A${r},${r} 0 1,1 ${r},0Z`
  }
}

export function Mark({ fam, r = 6, open = false, ring = false, color, strokeW = 1.6, style, className = '' }) {
  const c = color || famColor(fam)
  return (
    <g className={className} style={style}>
      {ring && <circle r={r + 4} fill="none" style={{ stroke: c }} strokeWidth="1.2" strokeDasharray="2 2" />}
      <path
        d={shapePath(fam, r)}
        style={{ fill: open ? 'var(--surface)' : c, stroke: open ? c : 'var(--surface)' }}
        strokeWidth={open ? strokeW : 1.2}
        strokeLinejoin="round"
      />
    </g>
  )
}

export function MarkIcon({ fam, size = 14, open = false, ring = false, title, color }) {
  return (
    <svg viewBox="-9 -9 18 18" width={size} height={size} aria-hidden={title ? undefined : true} role={title ? 'img' : undefined} style={{ display: 'inline-block', verticalAlign: '-0.12em', flexShrink: 0 }}>
      {title && <title>{title}</title>}
      <Mark fam={fam} r={5} open={open} ring={ring} color={color} />
    </svg>
  )
}

export function Verdict({ v, pending = false }) {
  const cls = String(v).toLowerCase()
  return <span className={`verdict ${cls}${pending ? ' pending' : ''}`}>{v}</span>
}

// ── Icons (1.6 px strokes, 24 grid) ─────────────────────────────────────────
const I = (d) => (props) => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true" {...props}>
    {d}
  </svg>
)
export const Icon = {
  arrow: I(<path d="M5 12h14M13 6l6 6-6 6" />),
  arrowLeft: I(<path d="M19 12H5M11 6l-6 6 6 6" />),
  arrowUpRight: I(<path d="M7 17 17 7M8 7h9v9" />),
  up: I(<path d="M12 19V5M6 11l6-6 6 6" />),
  down: I(<path d="M12 5v14M6 13l6 6 6-6" />),
  chevron: I(<path d="m9 6 6 6-6 6" />),
  chevronDown: I(<path d="m6 9 6 6 6-6" />),
  copy: I(<><rect x="9" y="9" width="11" height="11" rx="2.5" /><path d="M5 15V6.5A2.5 2.5 0 0 1 7.5 4H15" /></>),
  check: I(<path d="M5 12.5l4.5 4.5L19 7.5" />),
  sun: I(<><circle cx="12" cy="12" r="4" /><path d="M12 2.5v2M12 19.5v2M4.6 4.6 6 6M18 18l1.4 1.4M2.5 12h2M19.5 12h2M4.6 19.4 6 18M18 6l1.4-1.4" /></>),
  moon: I(<path d="M20 14.5A8 8 0 0 1 9.5 4a8 8 0 1 0 10.5 10.5Z" />),
  menu: I(<path d="M4 8h16M4 16h16" />),
  close: I(<path d="M6 6l12 12M18 6 6 18" />),
  search: I(<><circle cx="11" cy="11" r="6.5" /><path d="m20 20-4.2-4.2" /></>),
  sort: I(<path d="M8 9l4-4 4 4M8 15l4 4 4-4" />),
  sortDown: I(<path d="M7 10l5 5 5-5" />),
  sortUp: I(<path d="M7 14l5-5 5 5" />),
  paper: I(<><path d="M7 3h7l5 5v13H7z" /><path d="M14 3v5h5M10 13h6M10 17h6" /></>),
  info: I(<><circle cx="12" cy="12" r="8.5" /><path d="M12 11v5M12 8h.01" /></>),
  external: I(<path d="M14 5h5v5M19 5l-8 8M18 14v4a1 1 0 0 1-1 1H6a1 1 0 0 1-1-1V7a1 1 0 0 1 1-1h4" />),
  terminal: I(<><rect x="3" y="4.5" width="18" height="15" rx="2.5" /><path d="M7 9.5l3 2.5-3 2.5M12.5 15H17" /></>),
  download: I(<path d="M12 4v11M7 10l5 5 5-5M5 20h14" />),
  link: I(<><path d="M10 14a4 4 0 0 0 5.7 0l3-3a4 4 0 0 0-5.7-5.7l-1 1" /><path d="M14 10a4 4 0 0 0-5.7 0l-3 3a4 4 0 0 0 5.7 5.7l1-1" /></>),
  swap: I(<path d="M7 7h12l-3-3M17 17H5l3 3" />),
  book: I(<><path d="M5 4.5h9a3 3 0 0 1 3 3V20H8a3 3 0 0 1-3-3z" /><path d="M5 17a3 3 0 0 1 3-3h9" /></>),
  table: I(<><rect x="3.5" y="4.5" width="17" height="15" rx="2" /><path d="M3.5 9.5h17M3.5 14.5h17M9.5 9.5v10" /></>),
  grid: I(<><rect x="4" y="4" width="7" height="7" rx="1.5" /><rect x="13" y="4" width="7" height="7" rx="1.5" /><rect x="4" y="13" width="7" height="7" rx="1.5" /><rect x="13" y="13" width="7" height="7" rx="1.5" /></>),
  wrap: I(<path d="M4 6h16M4 12h13a3 3 0 0 1 0 6h-4m0 0 2-2m-2 2 2 2M4 18h5" />),
  hash: I(<path d="M9 4 7 20M17 4l-2 16M4 9h16M3 15h16" />),
  image: I(<><rect x="3.5" y="4.5" width="17" height="15" rx="2.5" /><circle cx="9" cy="10" r="1.6" /><path d="m20.5 16-4.5-4.5L6.5 19.5" /></>),
  chart: I(<><path d="M4 19.5h16" /><path d="M7 16v-5M11.5 16V7M16 16v-3.5" /></>),
  plus: I(<path d="M12 5v14M5 12h14" />),
  minus: I(<path d="M5 12h14" />),
  cube: I(<><path d="M12 3 4 7.5v9L12 21l8-4.5v-9z" /><path d="M4 7.5l8 4.5 8-4.5M12 12v9" /></>),
  github: (props) => (
    <svg viewBox="0 0 24 24" fill="currentColor" aria-hidden="true" {...props}>
      <path d="M12 2a10 10 0 0 0-3.16 19.49c.5.09.68-.22.68-.48v-1.7c-2.78.6-3.37-1.34-3.37-1.34-.46-1.16-1.11-1.47-1.11-1.47-.9-.62.07-.6.07-.6 1 .07 1.53 1.03 1.53 1.03.9 1.52 2.34 1.08 2.91.83.09-.65.35-1.08.63-1.33-2.22-.25-4.55-1.11-4.55-4.94 0-1.09.39-1.98 1.03-2.68-.1-.25-.45-1.27.1-2.64 0 0 .84-.27 2.75 1.02a9.6 9.6 0 0 1 5 0c1.91-1.29 2.75-1.02 2.75-1.02.55 1.37.2 2.39.1 2.64.64.7 1.03 1.59 1.03 2.68 0 3.84-2.34 4.69-4.57 4.94.36.31.68.92.68 1.85v2.74c0 .27.18.58.69.48A10 10 0 0 0 12 2Z" />
    </svg>
  ),
  pypi: I(<><path d="M12 3 4 7.5v9L12 21l8-4.5v-9z" /><path d="M4 7.5l8 4.5 8-4.5M12 12v9" /></>),
}

// ── Brand mark (the logo: a small judge and a large one on a level beam) ────
export function BrandMark({ className }) {
  return (
    <svg viewBox="0 0 120 120" className={className} aria-hidden="true">
      <circle cx="60" cy="60" r="57" style={{ fill: 'var(--brand-disc, #1f3a5f)' }} />
      <circle cx="40" cy="64" r="12" style={{ fill: '#e1e8f3' }} />
      <polygon points="94.5,66.5 78.0,76.0 61.5,66.5 61.5,47.5 78.0,38.0 94.5,47.5" style={{ fill: '#bfcfea' }} />
      <rect x="22" y="80" width="78" height="4" rx="2" style={{ fill: '#e1e8f3' }} />
      <rect x="30" y="88" width="62" height="3" rx="1.5" style={{ fill: '#bfcfea', opacity: 0.8 }} />
    </svg>
  )
}

// ── Segmented control with a sliding thumb ──────────────────────────────────
export function Segmented({ options, value, onChange, label, size }) {
  const ref = useRef(null)
  const [thumb, setThumb] = useState(null)
  const measure = () => {
    const el = ref.current?.querySelector('[aria-pressed="true"]')
    if (el) setThumb({ x: el.offsetLeft, w: el.offsetWidth })
  }
  useLayoutEffect(measure, [value, options.length])
  useEffect(() => {
    document.fonts?.ready.then(measure)
    let ro
    if ('ResizeObserver' in window && ref.current) { ro = new ResizeObserver(measure); ro.observe(ref.current) }
    return () => ro?.disconnect()
  }, [])
  return (
    <div className={`seg${size === 'sm' ? ' seg-sm' : ''}`} role="group" aria-label={label} ref={ref}>
      {thumb && <span className="seg-thumb" style={{ width: thumb.w, transform: `translateX(${thumb.x}px)` }} />}
      {options.map((o) => (
        <button key={String(o.value)} type="button" aria-pressed={o.value === value} onClick={() => onChange(o.value)} title={o.title}>
          {o.label}
        </button>
      ))}
    </div>
  )
}

// ── Copy ────────────────────────────────────────────────────────────────────
export async function copyText(text) {
  try {
    await navigator.clipboard.writeText(text)
  } catch (e) {
    const ta = document.createElement('textarea')
    ta.value = text
    ta.style.position = 'fixed'
    ta.style.opacity = '0'
    document.body.appendChild(ta)
    ta.select()
    try { document.execCommand('copy') } catch (err) { /* nothing more to try */ }
    ta.remove()
  }
}

export function CopyButton({ text, label = 'Copy', className = 'copy-btn', iconOnly = false }) {
  const [done, setDone] = useState(false)
  const t = useRef(0)
  useEffect(() => () => clearTimeout(t.current), [])
  const copy = async () => {
    await copyText(typeof text === 'function' ? text() : text)
    setDone(true)
    clearTimeout(t.current)
    t.current = setTimeout(() => setDone(false), 1600)
  }
  return (
    <button type="button" className={`${className}${done ? ' done' : ''}`} onClick={copy} aria-label={done ? 'Copied' : `${label} to clipboard`} title={label}>
      {done ? <Icon.check /> : <Icon.copy />}
      {!iconOnly && <span aria-live="polite">{done ? 'Copied' : label}</span>}
    </button>
  )
}

// Minimal highlighting for shell, Python, YAML and BibTeX.
const esc = (s) => s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
export function highlight(code, lang) {
  return code.split('\n').map((line) => {
    const hash = lang === 'bibtex' || lang === 'text' ? -1 : line.search(/(^|\s)#(?!#)/)
    let body = hash >= 0 ? line.slice(0, hash) : line
    const comment = hash >= 0 ? line.slice(hash) : ''
    body = esc(body)
    if (lang === 'python') {
      body = body
        .replace(/(r?(&quot;|"|')(?:\\.|(?!\2).)*\2)/g, '\u0001$1\u0002')
        .replace(/\b(from|import|as|def|return|for|in|if|else|None|True|False|print|with|class)\b(?![^\u0001]*\u0002)/g, '<span class="k">$1</span>')
        .replace(/(?<![\w.])(\d+(?:\.\d+)?)(?![^\u0001]*\u0002)/g, '<span class="n">$1</span>')
        .replace(/\u0001/g, '<span class="s">').replace(/\u0002/g, '</span>')
    } else if (lang === 'bash') {
      body = body
        .replace(/("[^"]*")/g, '<span class="s">$1</span>')
        .replace(/^(\s*)(pip|python|git|cd|npm|export)\b/, '$1<span class="k">$2</span>')
        .replace(/(\s)(--?[a-z][\w-]*)/g, '$1<span class="p">$2</span>')
    } else if (lang === 'yaml') {
      body = body.replace(/^(\s*)([\w.-]+)(:)/, '$1<span class="p">$2</span>$3').replace(/(&quot;|")([^"]*)\1/g, '<span class="s">$&</span>')
    } else if (lang === 'bibtex') {
      body = body.replace(/^(@\w+)/, '<span class="k">$1</span>').replace(/^(\s*)(\w+)(=)/, '$1<span class="p">$2</span>$3')
    }
    return body + (comment ? `<span class="c">${esc(comment)}</span>` : '')
  }).join('\n')
}

export function Code({ code, lang = 'bash', title, output, wrapToggle = true, className = '' }) {
  const [wrap, setWrap] = useState(false)
  return (
    <div className={`code${wrap ? ' wrap' : ''} ${className}`}>
      <div className="code-head">
        <span>{title || lang}</span>
        <span className="code-actions">
          {wrapToggle && (
            <button type="button" className={`copy-btn${wrap ? ' on' : ''}`} aria-pressed={wrap} onClick={() => setWrap((w) => !w)} title="Wrap long lines">
              <Icon.wrap /><span className="sr-only">Wrap long lines</span>
            </button>
          )}
          <CopyButton text={code} />
        </span>
      </div>
      <pre tabIndex={0}><code dangerouslySetInnerHTML={{ __html: highlight(code, lang) }} /></pre>
      {output != null && (
        <div className="code-out">
          <div className="out-label">Output</div>
          <pre tabIndex={0}><code>{output}</code></pre>
        </div>
      )}
    </div>
  )
}

// A number as it is (no count-up: values never start from zero).
export function Counter({ value, nd = 0, format }) {
  const shown = format ? format(value) : nd === 0 ? Number(value).toLocaleString('en-US') : Number(value).toFixed(nd)
  return <span className="num">{shown}</span>
}

export function Kicker({ children }) {
  return <div className="kicker">{children}</div>
}
