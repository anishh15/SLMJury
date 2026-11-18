// Formatting only. Values arrive from src/data/*.json already rounded once,
// half up, at the precision shown here, so toFixed never rounds again.
const MINUS = '−'

export function fmt(v, nd = 2) {
  if (v === null || v === undefined || Number.isNaN(v)) return '—'
  const s = Number(v).toFixed(nd)
  if (Number(s) === 0) return s.replace('-', '')
  return s.startsWith('-') ? MINUS + s.slice(1) : s
}

export function signed(v, nd = 2) {
  if (v === null || v === undefined) return '—'
  const s = fmt(Math.abs(v), nd)
  if (Number(v.toFixed(nd)) === 0) return s
  return (v > 0 ? '+' : MINUS) + s
}

export const count = (n) => Number(n).toLocaleString('en-US')
export const budgetLabel = (b) => count(b)
export const minusText = (s) => String(s).replace(/^-/, MINUS)

const WORDS = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten']
export const word = (n) => WORDS[n] ?? String(n)
export const cap = (s) => s.charAt(0).toUpperCase() + s.slice(1)
