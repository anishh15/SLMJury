import { useEffect, useState } from 'react'
import core from '../data/core.json'

// core.json ships with the first page; results.json (every per-figure table)
// and the docs load on demand, once.
export { core }
export const { facts, claims, meta } = core
export const judges = core.judges
export const byKey = Object.fromEntries(judges.map((j) => [j.key, j]))
export const name = (k) => byKey[k]?.name ?? k
export const FAMILIES = core.families

export const FAMILY_SLUG = { Llama: 'llama', 'Qwen2.5': 'qwen25', Qwen3: 'qwen3', 'Phi-4': 'phi4', Large: 'large' }
export const famColor = (fam) => `var(--fam-${FAMILY_SLUG[fam]})`
export const famLabel = (fam) => FAMILIES.find((f) => f.key === fam)?.label ?? fam

// Leaderboard order (rank), ladder judge last.
export const ranked = [...judges].sort((a, b) => (a.rank ?? 99) - (b.rank ?? 99))
export const QUICK = facts.budgets[0]
export const REASONED = facts.budgets[facts.budgets.length - 1]

// One point per judge, as in the paper's cost table (quick verdict, or the
// reasoned budget for a judge that always reasons).
export const points = core.points
export const pointOf = Object.fromEntries(points.map((p) => [p.key, p]))
export const onFront = new Set(core.front)

// The comparisons the paper states in words, keyed by the sorted pair.
const statementOf = Object.fromEntries(core.statements.map((s) => [s.pair.join('|'), s]))
export const statement = (a, b) => statementOf[[a, b].sort().join('|')] || null

let resultsPromise = null
let resultsCache = null
export function loadResults() {
  if (!resultsPromise) resultsPromise = import('../data/results.json').then((m) => (resultsCache = m.default))
  return resultsPromise
}
export const peekResults = () => resultsCache

export function useResults() {
  const [res, setRes] = useState(peekResults)
  useEffect(() => { if (!res) loadResults().then(setRes) }, [])
  return res
}

export const AUTHORS = [
  { name: 'Anish Laddha', aff: 1, href: 'https://github.com/anishh15' },
  { name: 'Nitesh Pradhan', aff: 1 },
  { name: 'Gaurav Srivastava', aff: 2, href: 'https://github.com/ctrl-gaurav' },
]
export const AFFILIATIONS = ['LNMIIT, India', 'Virginia Tech, USA']
export const EMAILS = ['anshladdha15@gmail.com', 'nitesh.pradhan@lnmiit.ac.in', 'gks@vt.edu']

// Linked highlighting: hovering a judge anywhere lights it up everywhere.
const hoverSubs = new Set()
let hovered = null
export function setHovered(k) {
  if (k === hovered) return
  hovered = k
  hoverSubs.forEach((fn) => fn(k))
}
export function useHovered() {
  const [k, setK] = useState(hovered)
  useEffect(() => {
    hoverSubs.add(setK)
    return () => hoverSubs.delete(setK)
  }, [])
  return k
}
