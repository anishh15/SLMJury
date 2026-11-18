import { useEffect, useState } from 'react'

// LaTeX in item text ($…$, $$…$$, \(…\), \[…\], or a bare answer such as \frac{11}{2}),
// rendered with KaTeX. KaTeX loads on first use, so only pages that show a
// math item pay for it; until then the raw text shows.
let katexLoad = null
const loadKatex = () => {
  katexLoad ??= Promise.all([import('katex'), import('katex/dist/katex.min.css')]).then(([m]) => m.default)
  return katexLoad
}

const MATH = /\$\$([\s\S]+?)\$\$|\\\[([\s\S]+?)\\\]|\\\(([\s\S]+?)\\\)|\$([^$\n]+?)\$/g

function render(katex, tex, display = false) {
  return katex.renderToString(tex, { throwOnError: false, strict: 'ignore', output: 'htmlAndMathml', displayMode: display })
}

export default function MathText({ text, whole = false }) {
  const [katex, setKatex] = useState(null)
  useEffect(() => {
    let live = true
    loadKatex().then((k) => live && setKatex(() => k))
    return () => { live = false }
  }, [])
  if (!katex || !text) return text
  if (whole) return /\\/.test(text) ? <span className="math" dangerouslySetInnerHTML={{ __html: render(katex, text) }} /> : text
  const out = []
  let last = 0
  for (const m of text.matchAll(MATH)) {
    if (m.index > last) out.push(text.slice(last, m.index))
    const display = m[2] != null
    out.push(<span key={m.index} className={display ? 'math math-block' : 'math'} dangerouslySetInnerHTML={{ __html: render(katex, m[1] ?? m[2] ?? m[3] ?? m[4], display) }} />)
    last = m.index + m[0].length
  }
  if (last < text.length) out.push(text.slice(last))
  return out
}
