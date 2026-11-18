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

export function useTheme() {
  const [theme, set] = useState(getTheme)
  useEffect(() => {
    listeners.add(set)
    return () => listeners.delete(set)
  }, [])
  return theme
}
