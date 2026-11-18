import { useLayoutEffect, useRef } from 'react'
import { reducedMotion } from './motion'

// FLIP: rows marked data-flip="<key>" glide from their old position to the
// new one when the order changes, instead of re-rendering in place.
export function useFlip(ref, deps, { duration = 360, easing = 'cubic-bezier(0.16, 1, 0.3, 1)' } = {}) {
  const prev = useRef(null)
  useLayoutEffect(() => {
    const el = ref.current
    if (!el) return
    const rows = [...el.querySelectorAll('[data-flip]')]
    const next = new Map(rows.map((r) => [r.dataset.flip, r.offsetTop]))
    if (prev.current && !reducedMotion()) {
      rows.forEach((r) => {
        const before = prev.current.get(r.dataset.flip)
        const after = next.get(r.dataset.flip)
        if (before == null) r.animate([{ opacity: 0 }, { opacity: 1 }], { duration: 220, easing: 'ease-out' })
        else if (Math.abs(before - after) > 0.5) {
          r.animate([{ transform: `translateY(${before - after}px)` }, { transform: 'translateY(0)' }], { duration, easing })
        }
      })
    }
    prev.current = next
  }, deps)
}
