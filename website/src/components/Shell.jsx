import { useEffect, useLayoutEffect, useRef, useState } from 'react'
import { Icon } from './ui'
import Mark from './Wordmark'
import { meta } from '../lib/data'
import { setTheme, useTheme } from '../lib/theme'
import { springAnimate } from '../lib/motion'

export const NAV = [
  { page: 'findings', label: 'Findings', href: '#/findings' },
  { page: 'leaderboard', label: 'Leaderboard', href: '#/leaderboard' },
  { page: 'pick', label: 'Pick a judge', href: '#/pick' },
  { page: 'method', label: 'Method', href: '#/method' },
  { page: 'docs', label: 'Docs', href: '#/docs/getting-started' },
]
const isMac = typeof navigator !== 'undefined' && /Mac|iPhone|iPad/.test(navigator.platform || navigator.userAgent)

export function ThemeToggle({ className = 'icon-btn' }) {
  const theme = useTheme()
  const next = theme === 'dark' ? 'light' : 'dark'
  return (
    <button type="button" className={className} onClick={() => setTheme(next)} aria-label={`Switch to ${next} theme`} title={`Switch to ${next} theme`}>
      {theme === 'dark' ? <Icon.sun /> : <Icon.moon />}
    </button>
  )
}

export function Wordmark() {
  return (
    <a className="brand" href="#/" aria-label="SLMJury home">
      <Mark size="md" />
    </a>
  )
}

// The active link's underline slides between items on a spring.
function NavLinks({ current }) {
  const ref = useRef(null)
  const bar = useRef(null)
  const last = useRef(null)
  const measure = (animateIt) => {
    const el = ref.current?.querySelector('[aria-current="page"]')
    const b = bar.current
    if (!b) return
    if (!el) { b.style.opacity = '0'; last.current = null; return }
    const next = { x: el.offsetLeft + 12, w: el.offsetWidth - 24 }
    if (animateIt && last.current) {
      springAnimate(b, { transform: [`translateX(${last.current.x}px)`, `translateX(${next.x}px)`], width: [`${last.current.w}px`, `${next.w}px`], opacity: 1 })
    }
    b.style.transform = `translateX(${next.x}px)`
    b.style.width = `${next.w}px`
    b.style.opacity = '1'
    last.current = next
  }
  useLayoutEffect(() => measure(true), [current])
  useEffect(() => {
    const on = () => measure(false)
    document.fonts?.ready.then(on)
    window.addEventListener('resize', on)
    return () => window.removeEventListener('resize', on)
  }, [])
  return (
    <nav className="nav" aria-label="Main" ref={ref}>
      {NAV.map((n) => (
        <a key={n.page} href={n.href} aria-current={current === n.page ? 'page' : undefined}>{n.label}</a>
      ))}
      <span className="nav-bar" aria-hidden="true" ref={bar} />
    </nav>
  )
}

export function Header({ route, onSearch }) {
  const [scrolled, setScrolled] = useState(false)
  const [open, setOpen] = useState(false)
  const progress = useRef(null)
  const menuBtn = useRef(null)
  const menu = useRef(null)
  useEffect(() => {
    let raf = 0
    const on = () => {
      cancelAnimationFrame(raf)
      raf = requestAnimationFrame(() => {
        setScrolled(window.scrollY > 6)
        const h = document.documentElement.scrollHeight - window.innerHeight
        if (progress.current) progress.current.style.transform = `scaleX(${h > 0 ? Math.min(1, window.scrollY / h) : 0})`
      })
    }
    on()
    window.addEventListener('scroll', on, { passive: true })
    window.addEventListener('resize', on)
    return () => { window.removeEventListener('scroll', on); window.removeEventListener('resize', on); cancelAnimationFrame(raf) }
  }, [route.page])
  useEffect(() => setOpen(false), [route.raw])
  // the phone menu keeps focus inside itself (and its toggle) and gives it back to the toggle on close
  useEffect(() => {
    if (!open) return
    const btn = menuBtn.current
    const items = () => [btn, ...(menu.current?.querySelectorAll('a[href], button:not([disabled])') || [])].filter(Boolean)
    const onKey = (e) => {
      if (e.key === 'Escape') { setOpen(false); return }
      if (e.key !== 'Tab') return
      const list = items()
      const i = list.indexOf(document.activeElement)
      const next = e.shiftKey ? (i <= 0 ? list.length - 1 : i - 1) : (i === -1 || i === list.length - 1 ? 0 : i + 1)
      e.preventDefault()
      list[next]?.focus()
    }
    window.addEventListener('keydown', onKey)
    document.documentElement.classList.add('lock')
    requestAnimationFrame(() => menu.current?.querySelector('a[href]')?.focus())
    return () => {
      window.removeEventListener('keydown', onKey)
      document.documentElement.classList.remove('lock')
      const lost = !document.activeElement || document.activeElement === document.body
      if (btn?.isConnected && btn.offsetParent && lost) btn.focus()
    }
  }, [open])
  const current = route.page === 'judge' ? 'leaderboard' : route.page
  const long = ['findings', 'docs', 'method', 'judge'].includes(route.page)
  return (
    <>
      <header className={`site-header${scrolled || open ? ' solid' : ''}${route.page === 'home' ? ' on-home' : ''}`}>
        <div className="wrap-wide header-row">
          <Wordmark />
          <NavLinks current={current} />
          <div className="header-actions">
            <button type="button" className="search-btn" onClick={onSearch} aria-label="Search the site" aria-keyshortcuts={isMac ? 'Meta+K' : 'Control+K'}>
              <Icon.search />
              <span className="search-label">Search</span>
              <span className="search-keys"><kbd>{isMac ? '⌘' : 'Ctrl'}</kbd><kbd>K</kbd></span>
            </button>
            <a className="btn btn-sm header-paper" href={meta.arxiv} target="_blank" rel="noreferrer" aria-label={`Paper, ${meta.venue.label}`}>
              <Icon.paper /><span>Paper<span className="header-venue"> · {meta.venue.short}</span></span>
            </a>
            <a className="icon-btn hide-sm" href={meta.code} target="_blank" rel="noreferrer" aria-label="Code on GitHub" title="Code on GitHub">
              <Icon.github />
            </a>
            <ThemeToggle className="icon-btn hide-sm" />
            <button type="button" className="icon-btn menu-btn" ref={menuBtn} aria-expanded={open} aria-controls="mobile-nav" aria-label={open ? 'Close menu' : 'Open menu'} onClick={() => setOpen((o) => !o)}>
              {open ? <Icon.close /> : <Icon.menu />}
            </button>
          </div>
        </div>
        {long && <div className="progress" ref={progress} aria-hidden="true" />}
      </header>
      {open && (
          <div className="mobile-nav" id="mobile-nav" ref={menu}>
            <nav aria-label="Main">
              <a href="#/" aria-current={current === 'home' ? 'page' : undefined}>Overview <Icon.arrow /></a>
              {NAV.map((n) => (
                <a key={n.page} href={n.href} aria-current={current === n.page ? 'page' : undefined}>{n.label} <Icon.arrow /></a>
              ))}
              <a href="#/cite" aria-current={current === 'cite' ? 'page' : undefined}>Paper and citation <Icon.arrow /></a>
            </nav>
            <div className="mobile-tools">
              <button type="button" className="btn" onClick={() => { setOpen(false); onSearch() }}><Icon.search /> Search</button>
              <a className="btn" href={meta.code} target="_blank" rel="noreferrer"><Icon.github /> GitHub</a>
              <ThemeToggle className="btn btn-icon-only" />
            </div>
          </div>
      )}
    </>
  )
}

export function Footer() {
  return (
    <footer className="site-footer">
      <div className="wrap-wide">
        <div className="footer-top">
          <div className="footer-brand">
            <Wordmark />
          </div>
          <nav className="footer-cols" aria-label="Footer">
            <div>
              <h2>Study</h2>
              <a href="#/findings">Findings</a>
              <a href="#/leaderboard">Leaderboard</a>
              <a href="#/pick">Pick a judge</a>
              <a href="#/method">Method</a>
            </div>
            <div>
              <h2>Docs</h2>
              <a href="#/docs/getting-started">Getting started</a>
              <a href="#/docs/cli-pipeline">Command line</a>
              <a href="#/docs/api-core">Python API</a>
              <a href="#/docs/prompts">Prompts</a>
              <a href="#/docs/reproduce">Reproduce the paper</a>
            </div>
            <div>
              <h2>Project</h2>
              <a href={meta.code} target="_blank" rel="noreferrer">GitHub</a>
              <a href={meta.pypi} target="_blank" rel="noreferrer">PyPI</a>
              <a href={`${meta.code}/issues`} target="_blank" rel="noreferrer">Issues</a>
              <a href={meta.arxiv} target="_blank" rel="noreferrer">arXiv</a>
              <a href="#/cite">Cite</a>
            </div>
          </nav>
        </div>
        <div className="footer-base">
          <span>Version {meta.version} · Apache 2.0</span>
        </div>
      </div>
    </footer>
  )
}

export function ToTop() {
  const [show, setShow] = useState(false)
  useEffect(() => {
    const on = () => setShow(window.scrollY > 1600)
    window.addEventListener('scroll', on, { passive: true })
    return () => window.removeEventListener('scroll', on)
  }, [])
  return (
    <button type="button" className={`to-top${show ? ' show' : ''}`} aria-label="Back to top" tabIndex={show ? 0 : -1}
      onClick={() => window.scrollTo({ top: 0, behavior: window.matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth' })}>
      <Icon.up />
    </button>
  )
}
