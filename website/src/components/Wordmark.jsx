import '../styles/wordmark.css'

// The SLMJury wordmark as the README banner draws it: "SLM" in Newsreader
// Bold, "Jury" in Bold Italic, each with its gradient.
export default function Wordmark({ as: Tag = 'span', size = 'md', glow = false, className = '' }) {
  return (
    <Tag className={`wordmark wm-${size}${glow ? ' wm-glow' : ''} ${className}`}>
      <span className="wm-text">
        <span className="wm-slm">SLM</span><span className="wm-jury">Jury</span>
      </span>
    </Tag>
  )
}
