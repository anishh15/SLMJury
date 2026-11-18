import '../styles/wordmark.css'

// The SLMJury wordmark as the README banner draws it: "SLM" in Newsreader
// Bold, "Jury" in Bold Italic, each with its gradient. With split, each letter
// of "SLM" and the word "Jury" sit in their own boxes, so the hero can raise them in turn.
export default function Wordmark({ as: Tag = 'span', size = 'md', split = false, className = '' }) {
  return (
    <Tag className={`wordmark wm-${size}${split ? ' wm-split' : ''} ${className}`}>
      {split && <span className="sr-only">SLMJury</span>}
      <span className="wm-text" aria-hidden={split || undefined}>
        {split ? (
          <>
            {['S', 'L', 'M'].map((c, i) => <span key={c} className="wm-l" style={{ '--i': i }}><span className="wm-slm">{c}</span></span>)}
            <span className="wm-l wm-lj"><span className="wm-jury">Jury</span></span>
          </>
        ) : (
          <><span className="wm-slm">SLM</span><span className="wm-jury">Jury</span></>
        )}
      </span>
    </Tag>
  )
}
