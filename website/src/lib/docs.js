import { useEffect, useState } from 'react'

// Each Docs page is its own JSON chunk, fetched when the page or the search opens.
const files = import.meta.glob('../data/docs/*.json', { import: 'default' })
const promises = {}
const loaded = {}

export function loadDoc(slug) {
  const k = `../data/docs/${slug}.json`
  if (!files[k]) return Promise.resolve(null)
  if (!promises[slug]) promises[slug] = files[k]().then((d) => (loaded[slug] = d))
  return promises[slug]
}
export const peekDoc = (slug) => loaded[slug]
export const loadDocIndex = () => loadDoc('index')

export function useDoc(slug) {
  const [doc, setDoc] = useState(() => peekDoc(slug))
  useEffect(() => {
    setDoc(peekDoc(slug))
    let live = true
    loadDoc(slug).then((d) => { if (live) setDoc(d) })
    return () => { live = false }
  }, [slug])
  return doc
}
