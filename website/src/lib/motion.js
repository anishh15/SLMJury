import { spring } from 'motion'
import { animate as waapi } from 'motion/mini'
import { useEffect, useLayoutEffect, useRef, useState } from 'react'

// Motion tokens, mirrored in styles/tokens.css.
export const DUR = { press: 0.12, hover: 0.18, ui: 0.24, layout: 0.36, reveal: 0.45, morph: 0.7 }
export const EASE_OUT = [0.16, 1, 0.3, 1]
export const EASE_UI = [0.2, 0, 0, 1]
export const SPRING_UI = { type: 'spring', stiffness: 380, damping: 32 }
export const SPRING_BEAM = { stiffness: 120, damping: 14, mass: 1 }

export const reducedMotion = () =>
  typeof window !== 'undefined' && window.matchMedia?.('(prefers-reduced-motion: reduce)').matches

const inViewport = (el, margin = 0) => {
  const r = el.getBoundingClientRect()
  return r.top < window.innerHeight - margin && r.bottom > 0
}

// True once the element is 15% visible, and true from the start if it is
// already on screen at mount, so nothing above the fold waits.
export function useInView({ threshold = 0.15, rootMargin = '0px 0px -8% 0px' } = {}) {
  const ref = useRef(null)
  const [seen, setSeen] = useState(false)
  const [atOnce, setAtOnce] = useState(false)
  useLayoutEffect(() => {
    const el = ref.current
    if (el && inViewport(el)) { setSeen(true); setAtOnce(true) }
  }, [])
  useEffect(() => {
    const el = ref.current
    if (!el || seen) return
    if (!('IntersectionObserver' in window)) { setSeen(true); return }
    const io = new IntersectionObserver((entries) => {
      if (entries.some((e) => e.isIntersecting)) { setSeen(true); io.disconnect() }
    }, { threshold, rootMargin })
    io.observe(el)
    return () => io.disconnect()
  }, [seen])
  return [ref, seen, atOnce]
}

const easeOut = (t) => 1 - Math.pow(1 - t, 3)
const easeIO = (t) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2)

// Tween a flat array of numbers toward `target`; charts animate through this,
// so axes, marks and lines move together.
export function useTween(target, { duration = 650, from = null, run = true, ease = 'io' } = {}) {
  const key = target.join(',')
  const [vals, setVals] = useState(() => (from && run === false ? from : target))
  const cur = useRef(vals)
  const raf = useRef(0)
  const first = useRef(true)
  useEffect(() => {
    if (!run) return
    const start = first.current && from && !reducedMotion() ? from : cur.current
    first.current = false
    if (reducedMotion() || start.length !== target.length || start.join(',') === key) {
      cur.current = target
      setVals(target)
      return
    }
    const fn = ease === 'out' ? easeOut : easeIO
    const t0 = performance.now()
    cancelAnimationFrame(raf.current)
    const step = (now) => {
      const t = Math.min(1, (now - t0) / duration)
      const k = fn(t)
      const next = target.map((v, i) => {
        const a = start[i]
        if (a === null || v === null || Number.isNaN(a) || Number.isNaN(v)) return v
        return a + (v - a) * k
      })
      cur.current = next
      setVals(next)
      if (t < 1) raf.current = requestAnimationFrame(step)
    }
    raf.current = requestAnimationFrame(step)
    return () => cancelAnimationFrame(raf.current)
  }, [key, run])
  return vals
}

// True while the document is visible (ambient motion pauses in hidden tabs).
export function usePageVisible() {
  const [v, setV] = useState(() => typeof document === 'undefined' || document.visibilityState === 'visible')
  useEffect(() => {
    const on = () => setV(document.visibilityState === 'visible')
    document.addEventListener('visibilitychange', on)
    return () => document.removeEventListener('visibilitychange', on)
  }, [])
  return v
}

export function useReducedMotionPref() {
  const [r, setR] = useState(reducedMotion)
  useEffect(() => {
    const mq = window.matchMedia('(prefers-reduced-motion: reduce)')
    const on = () => setR(mq.matches)
    mq.addEventListener?.('change', on)
    return () => mq.removeEventListener?.('change', on)
  }, [])
  return r
}

// ── Springs ─────────────────────────────────────────────────────────────────
// A value that follows its target on a spring (motion's spring generator),
// keeping its velocity when the target changes mid-flight.

export function createSpring(initial, { stiffness, damping, mass = 1 }, onFrame) {
  let value = initial
  let velocity = 0
  let gen = null
  let t0 = 0
  let raf = 0
  const step = (now) => {
    const t = now - t0
    const s = gen.next(t)
    const prev = gen.next(Math.max(0, t - 8)).value
    velocity = ((s.value - prev) / 8) * 1000
    value = s.value
    onFrame(value)
    if (s.done) { velocity = 0; raf = 0; return }
    raf = requestAnimationFrame(step)
  }
  return {
    get: () => value,
    animating: () => raf !== 0,
    jump(v) { cancelAnimationFrame(raf); raf = 0; value = v; velocity = 0; onFrame(v) },
    set(target) {
      if (reducedMotion()) return this.jump(target)
      cancelAnimationFrame(raf)
      gen = spring({ keyframes: [value, target], velocity, stiffness, damping, mass, restDelta: 0.002, restSpeed: 0.01 })
      t0 = performance.now()
      raf = requestAnimationFrame(step)
    },
    stop() { cancelAnimationFrame(raf); raf = 0 },
  }
}

// WAAPI animation of an element on a spring, skipped under reduced motion.
export function springAnimate(el, keyframes, opts = SPRING_UI) {
  if (!el) return null
  if (reducedMotion()) {
    const last = Object.fromEntries(Object.entries(keyframes).map(([k, v]) => [k, Array.isArray(v) ? v[v.length - 1] : v]))
    Object.assign(el.style, last)
    return null
  }
  return waapi(el, keyframes, { type: spring, stiffness: opts.stiffness, damping: opts.damping })
}

// True while a media query matches.
export function useMedia(q) {
  const [on, setOn] = useState(() => window.matchMedia(q).matches)
  useEffect(() => {
    const mq = window.matchMedia(q)
    const fn = () => setOn(mq.matches)
    mq.addEventListener?.('change', fn)
    return () => mq.removeEventListener?.('change', fn)
  }, [q])
  return on
}

// True once the element comes within about a screen of the viewport, so
// charts below the fold mount only when the reader gets near them.
export function useNear(margin = '90% 0px 90% 0px') {
  const ref = useRef(null)
  const [near, setNear] = useState(false)
  useEffect(() => {
    const el = ref.current
    if (!el || near) return
    if (!('IntersectionObserver' in window)) { setNear(true); return }
    const io = new IntersectionObserver((es) => {
      if (es.some((e) => e.isIntersecting)) { setNear(true); io.disconnect() }
    }, { rootMargin: margin })
    io.observe(el)
    return () => io.disconnect()
  }, [near])
  return [ref, near]
}

// The viewport height, updated on resize.
export function useViewportH() {
  const [h, setH] = useState(() => window.innerHeight)
  useEffect(() => {
    let raf = 0
    const on = () => { cancelAnimationFrame(raf); raf = requestAnimationFrame(() => setH(window.innerHeight)) }
    window.addEventListener('resize', on)
    return () => { window.removeEventListener('resize', on); cancelAnimationFrame(raf) }
  }, [])
  return h
}
