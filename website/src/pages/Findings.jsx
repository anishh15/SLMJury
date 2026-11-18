import { lazy, Suspense, useEffect, useMemo, useRef, useState } from 'react'
import ChartFrame from '../charts/Frame'
import Stage from '../charts/Stage'
import { Icon, MarkIcon } from '../components/ui'
import { facts, FAMILIES, useResults } from '../lib/data'
import { FINDINGS } from '../lib/findings'
import { count } from '../lib/format'
import { useMedia, useNear, useViewportH } from '../lib/motion'
import { replaceHash } from '../lib/router'
import { buildStories } from '../lib/stories'
import StrategyExample from '../components/StrategyExample'
import '../styles/content.css'
import '../styles/story.css'

const Details = lazy(() => import('./FindingDetails'))
const R = count(facts.budgets[facts.budgets.length - 1])

export function FamilyKey({ reasoned = true, short = false }) {
  return (
    <span className="fam-key">
      {FAMILIES.map((f) => <span key={f.key}><MarkIcon fam={f.key} size={11} />{short ? f.key : f.label}</span>)}
      {reasoned && <span><MarkIcon fam="Qwen3" size={11} open />B = {R}</span>}
    </span>
  )
}

function SwatchKey({ items }) {
  return <span className="fam-key">{items.map((x) => <span key={x.label}><i className="swatch" style={{ background: x.color }} />{x.label}</span>)}</span>
}

// a one-line family key under each chart (short labels); open marks are B = 8,192 unless a chart says otherwise
const keyFor = (spec) => (spec.legend ? <SwatchKey items={spec.legend} />
  : spec.marks.some((m) => m.fam) ? <FamilyKey short reasoned={!spec.openNote && spec.marks.some((m) => m.open && m.fam)} /> : null)

function StepCard({ s }) {
  return (
    <div className="step-card">
      <div className="step-big tnum">{s.big.v}{s.big.u && <small>{s.big.u}</small>}</div>
      <p className="step-note">{s.big.note}</p>
    </div>
  )
}

function InlineStep({ s, slug }) {
  const [ref, near] = useNear('60% 0px 60% 0px')
  const h = s.spec.heightNarrow ?? 330
  return (
    <li className="step inline" ref={ref} data-reveal>
      <StepCard s={s} />
      <ChartFrame id={`${slug}-step-${s.key}`} title={s.spec.title} sub={s.spec.sub} table={s.spec.table} legend={keyFor(s.spec)}>
        {near ? <Stage spec={s.spec} height={h} /> : <div className="stage-ph" style={{ height: h }} />}
      </ChartFrame>
    </li>
  )
}

function Story({ f, i, story, wide, vh, openDetails, intro }) {
  const [step, setStep] = useState(0)
  const [ref, near] = useNear()
  const steps = useRef([])
  const [open, setOpen] = useState(false)
  useEffect(() => { if (openDetails) setOpen(true) }, [openDetails])

  useEffect(() => {
    if (!wide || !('IntersectionObserver' in window)) return
    const io = new IntersectionObserver((es) => {
      es.forEach((e) => { if (e.isIntersecting) setStep(Number(e.target.dataset.i)) })
    }, { rootMargin: '-48% 0px -48% 0px' })
    steps.current.forEach((el) => el && io.observe(el))
    return () => io.disconnect()
  }, [wide])

  const cur = story.steps[step]
  const h = Math.max(320, Math.min(cur.spec.height ?? 440, vh - 300))
  const goto = (k) => steps.current[k]?.scrollIntoView({ behavior: window.matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth', block: 'center' })
  const dots = (
    <div className="step-dots" role="group" aria-label="Steps">
      {story.steps.map((s, k) => (
        <button key={s.key} type="button" aria-pressed={k === step} aria-label={`Step ${k + 1}`} onClick={() => goto(k)}><span /></button>
      ))}
    </div>
  )
  return (
    <section id={`f-${f.slug}`} className="story" aria-labelledby={`f-${f.slug}-h`} ref={ref}>
      <div className="wrap">
        <header className="story-head" data-reveal="stagger">
          <span className="story-n tnum">{String(i + 1).padStart(2, '0')}</span>
          <h2 className="h2" id={`f-${f.slug}-h`}>{f.title}</h2>
        </header>
        {intro}
        {wide ? (
          <div className="story-grid">
            <ol className="story-steps">
              {story.steps.map((s, k) => (
                <li key={s.key} id={`step-${f.slug}-step-${s.key}`} className={`step${k === step ? ' on' : ''}`} data-i={k} data-reveal ref={(el) => { steps.current[k] = el }}>
                  <StepCard s={s} />
                </li>
              ))}
            </ol>
            <div className="story-sticky" data-reveal>
              <ChartFrame id={`${f.slug}-step-${cur.key}`} title={cur.spec.title} sub={cur.spec.sub} table={cur.spec.table} legend={keyFor(cur.spec)} head={dots}>
                {near ? <Stage spec={cur.spec} height={h} /> : <div className="stage-ph" style={{ height: h }} />}
              </ChartFrame>
            </div>
          </div>
        ) : (
          <ol className="story-steps">
            {story.steps.map((s) => <InlineStep key={s.key} s={s} slug={f.slug} />)}
          </ol>
        )}
        <div className={`details${open ? ' open' : ''}`} data-reveal>
          <button type="button" className="details-btn" aria-expanded={open} aria-controls={`d-${f.slug}`} onClick={() => setOpen((o) => !o)}>
            <span className="details-ic">{open ? <Icon.minus /> : <Icon.plus />}</span>
            <span className="details-tx">
              <span className="details-t">{open ? 'Hide the details' : 'Details'}</span>
            </span>
          </button>
          <div className="details-body" id={`d-${f.slug}`}>
            <div className="details-inner">
              {open && (
                <Suspense fallback={<div className="skeleton" style={{ height: 420 }} />}>
                  <Details slug={f.slug} />
                </Suspense>
              )}
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}

export default function Findings({ route }) {
  const res = useResults()
  const slug = route.parts[1]
  const chart = route.query.get('chart')
  const wide = useMedia('(min-width: 900px)')
  const vh = useViewportH()
  const stories = useMemo(() => (res ? buildStories(res) : null), [res])
  const [active, setActive] = useState(slug || FINDINGS[0].slug)
  const lock = useRef(0)
  const nav = useRef(null)
  // a chart link names one story step (the same id on wide screens and phones) or a chart in Details
  const stepIds = useMemo(() => new Set(stories ? FINDINGS.flatMap((f, i) => stories[i].steps.map((s) => `${f.slug}-step-${s.key}`)) : []), [stories])
  const isStep = (c) => stepIds.has(c) || /-story$/.test(c)
  const detailFor = chart && stories && !isStep(chart) ? FINDINGS.find((f) => chart.startsWith(`${f.slug}-`))?.slug : null

  // deep link: a section, or one chart
  useEffect(() => {
    if (!stories) return
    const target = slug || (chart && FINDINGS.find((f) => chart.startsWith(`${f.slug}-`))?.slug)
    if (!target) return
    lock.current = performance.now()
    let tries = 0
    const go = () => {
      // wide screens show one sticky chart per story, so a step link scrolls to its step
      const stepEl = chart && wide && document.getElementById(`step-${chart}`)
      const chartEl = chart && !/-story$/.test(chart) && document.getElementById(`chart-${chart}`)
      if (stepEl) { stepEl.scrollIntoView({ block: 'center' }); return }
      if (chartEl) { chartEl.scrollIntoView({ block: 'start' }); return }
      if (chart && !/-story$/.test(chart) && tries++ < 60) { requestAnimationFrame(go); return }
      document.getElementById(`f-${target}`)?.scrollIntoView({ block: 'start' })
    }
    requestAnimationFrame(go)
    setActive(target)
  }, [stories, slug, chart, wide])

  // scroll spy: the section in view sets the nav and the hash
  useEffect(() => {
    if (!stories || !('IntersectionObserver' in window)) return
    const io = new IntersectionObserver((entries) => {
      const vis = entries.filter((e) => e.isIntersecting)
      if (!vis.length || performance.now() - lock.current < 900) return
      const s = vis[0].target.id.slice(2)
      setActive(s)
      replaceHash(`/findings/${s}`)
    }, { rootMargin: '-40% 0px -55% 0px' })
    FINDINGS.forEach((f) => { const el = document.getElementById(`f-${f.slug}`); if (el) io.observe(el) })
    return () => io.disconnect()
  }, [stories])

  // keep the active chip in view in the section nav
  useEffect(() => {
    const box = nav.current?.querySelector('.sec-nav-in')
    const el = box?.querySelector('.active')
    if (el && box.scrollWidth > box.clientWidth) box.scrollTo({ left: el.offsetLeft - 16, behavior: 'smooth' })
  }, [active])

  return (
    <>
      <section className="page-head findings-head">
        <div className="wrap">
          <h1 className="h-display">Six findings</h1>
        </div>
      </section>
      <nav className="sec-nav" aria-label="Findings" ref={nav}>
        <div className="wrap sec-nav-row">
          <div className="sec-nav-in">
            {FINDINGS.map((f, i) => (
              <a key={f.slug} href={`#/findings/${f.slug}`} className={active === f.slug ? 'active' : ''} aria-current={active === f.slug ? 'true' : undefined}>
                <span className="tnum">{String(i + 1).padStart(2, '0')}</span>{f.short}
              </a>
            ))}
          </div>
        </div>
      </nav>
      {!stories && <div className="wrap"><div className="skeleton" style={{ height: 560, marginTop: 40 }} aria-label="Loading the results" /></div>}
      {stories && FINDINGS.map((f, i) => (
        <Story key={f.slug} f={f} i={i} story={stories[i]} wide={wide} vh={vh} openDetails={detailFor === f.slug}
          intro={f.slug === 'strategies' ? <StrategyExample ex={res.strategyExample} /> : null} />
      ))}
    </>
  )
}
