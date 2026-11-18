import { lazy, Suspense, useEffect, useState } from 'react'
import { pageModule, parse, preload, registerExtra, setHomeModule, useRoute } from './lib/router'
import { Header, Footer, ToTop } from './components/Shell'
import Home from './pages/Home'
import { byKey, loadResults } from './lib/data'
import { loadDoc, loadDocIndex } from './lib/docs'

setHomeModule(Home)
// Pages that need the per-figure tables or a docs page get them before the
// route changes, so they render complete inside the view transition.
const withResults = () => loadResults()
registerExtra('findings', withResults)
registerExtra('judge', withResults)
registerExtra('pick', withResults)
registerExtra('leaderboard', withResults)
registerExtra('docs', (route) => Promise.all([loadDoc(route?.parts?.[1] || 'getting-started'), loadDocIndex()]))

const Palette = lazy(() => import('./components/CommandPalette'))

const TITLES = {
  home: 'SLMJury: Can Small Language Models Judge as Well as Large Ones?',
  findings: 'Findings · SLMJury',
  leaderboard: 'Leaderboard · SLMJury',
  pick: 'Pick a judge · SLMJury',
  method: 'Method · SLMJury',
  docs: 'Docs · SLMJury',
  cite: 'Paper and citation · SLMJury',
}

// A page loads when a link to it is hovered, focused or touched; the main pages follow one at a time once the first screen settles.
function usePrefetch() {
  useEffect(() => {
    const seen = new Set()
    const warm = (hash) => {
      if (seen.has(hash)) return Promise.resolve()
      seen.add(hash)
      const r = parse(hash)
      return preload(r.page, r)
    }
    const onIntent = (e) => {
      const a = e.target instanceof Element ? e.target.closest('a[href^="#/"]') : null
      if (a) warm(a.getAttribute('href'))
    }
    document.addEventListener('pointerover', onIntent, { passive: true })
    document.addEventListener('focusin', onIntent)
    document.addEventListener('touchstart', onIntent, { passive: true })

    const queue = [
      () => warm('#/findings'),
      () => warm('#/leaderboard'),
      () => warm('#/docs/getting-started'),
      () => import('./components/CommandPalette'),
    ]
    const hasIdle = 'requestIdleCallback' in window
    let timer = 0
    let idle = 0
    let stopped = false
    const whenIdle = (fn) => (hasIdle ? requestIdleCallback(fn, { timeout: 4000 }) : setTimeout(fn, 1200))
    const next = () => {
      if (stopped || !queue.length) return
      Promise.resolve(queue.shift()()).catch(() => {}).finally(() => { if (!stopped) idle = whenIdle(next) })
    }
    const settle = () => { timer = setTimeout(() => { idle = whenIdle(next) }, 2000) }
    const saveData = navigator.connection && navigator.connection.saveData
    if (!saveData) {
      if (document.readyState === 'complete') settle()
      else window.addEventListener('load', settle, { once: true })
    }
    return () => {
      stopped = true
      document.removeEventListener('pointerover', onIntent)
      document.removeEventListener('focusin', onIntent)
      document.removeEventListener('touchstart', onIntent)
      window.removeEventListener('load', settle)
      clearTimeout(timer)
      if (hasIdle) cancelIdleCallback(idle)
      else clearTimeout(idle)
    }
  }, [])
}

export default function App() {
  const route = useRoute()
  const [palette, setPalette] = useState(false)
  const [, force] = useState(0)
  const Page = pageModule(route.page)
  usePrefetch()
  useEffect(() => { if (!Page) preload(route.page, route).then(() => force((n) => n + 1)) }, [Page, route.page])
  useEffect(() => {
    if (route.page === 'judge') document.title = `${byKey[route.parts[1]]?.name ?? 'Judge'} · SLMJury`
    else document.title = TITLES[route.page] || 'Not found · SLMJury'
  }, [route.page, route.parts[1]])
  useEffect(() => {
    const onKey = (e) => {
      const typing = /^(INPUT|TEXTAREA|SELECT)$/.test(e.target.tagName) || e.target.isContentEditable
      if ((e.key === 'k' || e.key === 'K') && (e.metaKey || e.ctrlKey)) { e.preventDefault(); setPalette((o) => !o) }
      else if (e.key === '/' && !typing && !palette) { e.preventDefault(); setPalette(true) }
    }
    const onOpen = () => setPalette(true)
    window.addEventListener('keydown', onKey)
    window.addEventListener('slmjury:search', onOpen)
    return () => { window.removeEventListener('keydown', onKey); window.removeEventListener('slmjury:search', onOpen) }
  }, [palette])
  // a link to one chart (?chart=<id>): scroll to it once it has rendered
  const chartId = route.query.get('chart')
  useEffect(() => {
    if (!chartId || route.page === 'findings') return
    let tries = 0
    let raf = 0
    const go = () => {
      const el = document.getElementById(`chart-${chartId}`)
      if (el) el.scrollIntoView({ block: 'center' })
      else if (tries++ < 90) raf = requestAnimationFrame(go)
    }
    raf = requestAnimationFrame(go)
    return () => cancelAnimationFrame(raf)
  }, [chartId, route.page, route.parts[1]])
  const pageKey = route.page === 'judge' ? `judge-${route.parts[1]}` : route.page === 'docs' ? `docs-${route.parts[1]}` : route.page
  return (
    <>
        <a className="skip-link" href="#main" onClick={(e) => { e.preventDefault(); document.getElementById('main')?.focus() }}>Skip to content</a>
        <Header route={route} onSearch={() => setPalette(true)} />
        <main id="main" tabIndex={-1}>
          <div className="page" key={pageKey} data-page={route.page}>
            {Page ? <Page route={route} /> : <div className="wrap page-loading"><div className="skeleton" style={{ height: 420 }} /></div>}
          </div>
        </main>
        <Footer />
        <ToTop />
        {palette && (
          <Suspense fallback={null}>
            <Palette onClose={() => setPalette(false)} />
          </Suspense>
        )}
    </>
  )
}
