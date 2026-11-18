import { reducedMotion } from './motion'

// Scroll reveals for the whole site. An element marked data-reveal rises in the
// first time it scrolls into view; data-reveal="stagger" staggers its children.
// Only elements low on the screen or below it at mount are hidden, so nothing up top waits.
const STEP_MS = 60
const MAX_STEPS = 8
const DONE_MS = 2600

let io = null

function prepare(el) {
  const r = el.getBoundingClientRect()
  const hidden = r.width === 0 && r.height === 0
  if (hidden || r.top < window.innerHeight * 0.8) { el.dataset.rv = 'done'; return }
  if (el.dataset.reveal === 'stagger') {
    [...el.children].forEach((c, i) => c.style.setProperty('--rv-i', Math.min(i, MAX_STEPS)))
  }
  el.dataset.rv = 'wait'
  io.observe(el)
}

function show(el) {
  io.unobserve(el)
  el.dataset.rv = 'in'
  const extra = el.dataset.reveal === 'stagger' ? Math.min(el.children.length, MAX_STEPS) * STEP_MS : 0
  setTimeout(() => { el.dataset.rv = 'done' }, DONE_MS + extra)
}

export function startReveals(root = document.body) {
  if (io || reducedMotion() || !('IntersectionObserver' in window) || !('MutationObserver' in window)) return
  io = new IntersectionObserver((entries) => {
    entries.forEach((e) => { if (e.isIntersecting) show(e.target) })
  }, { rootMargin: '0px 0px -10% 0px', threshold: 0 })
  const scan = () => root.querySelectorAll('[data-reveal]:not([data-rv])').forEach(prepare)
  new MutationObserver(scan).observe(root, { childList: true, subtree: true })
  // printing shows everything at once
  window.addEventListener('beforeprint', () => root.querySelectorAll('[data-rv="wait"]').forEach(show))
  scan()
}
