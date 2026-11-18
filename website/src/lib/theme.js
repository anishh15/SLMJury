import { useEffect, useState } from 'react'

const KEY = 'slmjury-theme'
const listeners = new Set()

export function getTheme() {
  return document.documentElement.dataset.theme === 'dark' ? 'dark' : 'light'
}

export function setTheme(t) {
  const root = document.documentElement
  const apply = () => {
    root.dataset.theme = t
    root.style.colorScheme = t
    listeners.forEach((fn) => fn(t))
  }
  if (document.startViewTransition && !window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
    root.classList.add('theme-vt')
    const vt = document.startViewTransition(apply)
    vt.finished.finally(() => root.classList.remove('theme-vt'))
  } else apply()
  try { localStorage.setItem(KEY, t) } catch (e) { /* storage unavailable */ }
}

// With no saved choice, follow the system setting when it changes
if (typeof window !== 'undefined' && window.matchMedia) {
  window.matchMedia('(prefers-color-scheme: light)').addEventListener('change', (e) => {
    let saved = null
    try { saved = localStorage.getItem(KEY) } catch (err) { /* storage unavailable */ }
    if (saved !== 'light' && saved !== 'dark') {
      const t = e.matches ? 'light' : 'dark'
      document.documentElement.dataset.theme = t
      document.documentElement.style.colorScheme = t
      listeners.forEach((fn) => fn(t))
    }
  })
}

export function useTheme() {
  const [theme, set] = useState(getTheme)
  useEffect(() => {
    listeners.add(set)
    return () => listeners.delete(set)
  }, [])
  return theme
}
