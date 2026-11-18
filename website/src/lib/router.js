import { useEffect, useState } from 'react'
import { flushSync } from 'react-dom'
import { reducedMotion } from './motion'

// A small hash router: #/path/parts?query. Hash routes keep deep links working
// on GitHub Pages, where any path but index.html would 404.
const REDIRECTS = { '/reproduce': '/docs/reproduce', '/docs': '/docs/getting-started' }

// A malformed escape such as #/%ZZ makes decodeURIComponent throw, so it decodes to null.
const decode = (part) => { try { return decodeURIComponent(part) } catch (e) { return null } }

export function parse(hash = window.location.hash) {
  let raw = hash.replace(/^#/, '') || '/'
  const [p0, q0 = ''] = raw.split('?')
  const target = REDIRECTS[p0.replace(/\/$/, '')]
  if (target) raw = target + (q0 ? `?${q0}` : '')
  const [path, query = ''] = raw.split('?')
  const parts = path.split('/').filter(Boolean).map(decode)
  if (parts.includes(null)) return { page: 'notfound', parts: [], query: new URLSearchParams(query), raw }
  return { page: parts[0] || 'home', parts, query: new URLSearchParams(query), raw }
}

// Page modules load on demand. A route change waits for its module, so the
// view transition never snapshots a loading state.
const loaders = {
  findings: () => import('../pages/Findings'),
  leaderboard: () => import('../pages/Leaderboard'),
  judge: () => import('../pages/Judge'),
  pick: () => import('../pages/Pick'),
  method: () => import('../pages/Method'),
  docs: () => import('../pages/Docs'),
  cite: () => import('../pages/Cite'),
  notfound: () => import('../pages/NotFound'),
}
const loaded = {}
const extras = {}
export const registerExtra = (page, fn) => { extras[page] = fn }
export const pageKey = (page) => (page === 'home' || loaders[page] ? page : 'notfound')
export const pageModule = (page) => loaded[pageKey(page)]
export const setHomeModule = (C) => { loaded.home = C }
export function preload(page, route) {
  const k = pageKey(page)
  if (k === 'home') return Promise.resolve()
  const mod = loaded[k] ? Promise.resolve() : loaders[k]().then((m) => { loaded[k] = m.default })
  return Promise.all([mod, extras[k]?.(route)]).catch(() => {})
}

// ── Scroll restoration ──────────────────────────────────────────────────────
// Each history entry gets a key: a new entry starts at the top, and Back or
// Forward returns to where the reader was.
const positions = new Map()
try {
  JSON.parse(sessionStorage.getItem('slmjury-scroll') || '[]').forEach(([k, y]) => positions.set(k, y))
} catch (e) { /* storage unavailable */ }
if (typeof history !== 'undefined' && 'scrollRestoration' in history) history.scrollRestoration = 'manual'

const newKey = () => Math.random().toString(36).slice(2, 10)
function ensureKey() {
  if (!history.state || !history.state.k) history.replaceState({ ...(history.state || {}), k: newKey() }, '')
  return history.state.k
}
let currentKey = ensureKey()
const persist = () => {
  try { sessionStorage.setItem('slmjury-scroll', JSON.stringify([...positions].slice(-40))) } catch (e) { /* ignore */ }
}
let saveTimer = 0
window.addEventListener('scroll', () => {
  clearTimeout(saveTimer)
  saveTimer = setTimeout(() => { positions.set(currentKey, window.scrollY) }, 80)
}, { passive: true })
window.addEventListener('pagehide', () => { positions.set(currentKey, window.scrollY); persist() })

function restore(y) {
  let tries = 0
  const step = () => {
    const max = document.documentElement.scrollHeight - window.innerHeight
    window.scrollTo({ top: Math.min(y, Math.max(0, max)), behavior: 'instant' })
    if (max < y && tries++ < 60) requestAnimationFrame(step)
  }
  step()
}

// ── Navigation ──────────────────────────────────────────────────────────────
let current = parse()
if (window.location.hash && `#${current.raw}` !== window.location.hash) {
  history.replaceState(history.state, '', `#${current.raw}`)
}
const listeners = new Set()
let token = 0

async function onHash() {
  const next = parse()
  if (`#${next.raw}` !== window.location.hash) history.replaceState(history.state, '', `#${next.raw}`)
  positions.set(currentKey, window.scrollY)
  const st = history.state
  const traversal = !!(st && st.k && st.k !== currentKey && positions.has(st.k))
  const key = ensureKey()
  const prev = current
  const pageChanged = next.page !== prev.page || (['judge', 'docs'].includes(next.page) && next.parts[1] !== prev.parts[1])
  const my = ++token
  if (pageChanged) await preload(next.page, next)
  if (my !== token) return
  currentKey = key
  current = next
  const apply = () => listeners.forEach((fn) => fn(next))
  const scroll = () => {
    if (traversal) restore(positions.get(key) || 0)
    else if (pageChanged && !(next.page === 'findings' && next.parts[1]) && !(next.page === 'docs' && next.parts[2])) {
      window.scrollTo({ top: 0, behavior: 'instant' })
    }
  }
  if (pageChanged && document.startViewTransition && !reducedMotion() && document.visibilityState === 'visible') {
    const vt = document.startViewTransition(() => { flushSync(apply); scroll() })
    vt.finished.catch(() => {})
  } else {
    apply()
    if (pageChanged) scroll()
  }
}
window.addEventListener('hashchange', onHash)

export const initialRoute = () => current

export function useRoute() {
  const [route, setRoute] = useState(current)
  useEffect(() => {
    listeners.add(setRoute)
    return () => listeners.delete(setRoute)
  }, [])
  return route
}

export const href = (path, query) => {
  const q = query && [...new URLSearchParams(query)].length ? `?${new URLSearchParams(query).toString().replace(/%2C/g, ',')}` : ''
  return `#${path}${q}`
}

// Rewrite the query of the current hash without a history entry or a
// hashchange (filters, sort orders, the findings scroll spy).
export function replaceHash(path, query) {
  const h = href(path, query)
  if (h !== window.location.hash) history.replaceState(history.state, '', h)
  current = parse(h)
}

export const navigate = (path) => { window.location.hash = path }
