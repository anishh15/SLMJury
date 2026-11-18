import { useEffect, useId, useMemo, useRef, useState } from 'react'
import { Icon, MarkIcon } from './ui'
import { byKey, judges, pointOf, QUICK } from '../lib/data'
import { count, fmt } from '../lib/format'

const GROUPS = [['small', 'Small judges'], ['large', 'Large judges'], ['ladder', 'Ladder judge']]
const order = [...judges].filter((j) => pointOf[j.key]).sort((a, b) => pointOf[b.key].accExact - pointOf[a.key].accExact)

// A button that opens a searchable list of judges.
export default function JudgePicker({ value, onChange, disabled = [], label, className = '', children, align = 'left' }) {
  const [open, setOpen] = useState(false)
  const [q, setQ] = useState('')
  const [sel, setSel] = useState(0)
  const btn = useRef(null)
  const pop = useRef(null)
  const input = useRef(null)
  const id = useId()
  const list = useMemo(() => {
    const t = q.trim().toLowerCase()
    return order.filter((j) => !t || j.name.toLowerCase().includes(t) || j.key.includes(t) || j.size.toLowerCase().startsWith(t))
  }, [q])
  const flat = GROUPS.flatMap(([k]) => list.filter((j) => j.kind === k))
  useEffect(() => {
    if (!open) return
    setQ('')
    setSel(Math.max(0, flat.findIndex((j) => j.key === value)))
    requestAnimationFrame(() => input.current?.focus())
    const onDoc = (e) => { if (!pop.current?.contains(e.target) && !btn.current?.contains(e.target)) setOpen(false) }
    document.addEventListener('pointerdown', onDoc)
    return () => document.removeEventListener('pointerdown', onDoc)
  }, [open])
  useEffect(() => { pop.current?.querySelector('[aria-selected="true"]')?.scrollIntoView({ block: 'nearest' }) }, [sel, open])
  const pick = (k) => {
    if (disabled.includes(k)) return
    onChange(k)
    setOpen(false)
    btn.current?.focus()
  }
  const onKey = (e) => {
    if (e.key === 'ArrowDown') { e.preventDefault(); setSel((s) => Math.min(flat.length - 1, s + 1)) }
    else if (e.key === 'ArrowUp') { e.preventDefault(); setSel((s) => Math.max(0, s - 1)) }
    else if (e.key === 'Enter') { e.preventDefault(); flat[sel] && pick(flat[sel].key) }
    else if (e.key === 'Escape') { e.preventDefault(); setOpen(false); btn.current?.focus() }
  }
  const j = byKey[value]
  let n = -1
  return (
    <div className={`picker-wrap ${className}`}>
      <button ref={btn} type="button" className="picker-btn" aria-haspopup="listbox" aria-expanded={open} aria-controls={`${id}-list`}
        aria-label={`${label}: ${j.name}. Change judge`} onClick={() => setOpen((o) => !o)}>
        {children || (<><MarkIcon fam={j.family} size={14} /> <span>{j.name}</span> <Icon.chevronDown className="chev" /></>)}
      </button>
      {open && (
          <div ref={pop} className={`picker-pop ${align}`} onKeyDown={onKey}>
            <div className="picker-search">
              <Icon.search />
              <input ref={input} value={q} onChange={(e) => { setQ(e.target.value); setSel(0) }} placeholder="Find a judge"
                role="combobox" aria-expanded="true" aria-controls={`${id}-list`} aria-activedescendant={flat[sel] ? `${id}-${sel}` : undefined} />
            </div>
            <div className="picker-list" role="listbox" id={`${id}-list`} aria-label={label}>
              {GROUPS.map(([k, title]) => {
                const items = list.filter((x) => x.kind === k)
                if (!items.length) return null
                return (
                  <div key={k} role="group" aria-label={title}>
                    <div className="picker-group">{title}</div>
                    {items.map((x) => {
                      n += 1
                      const idx = n
                      const p = pointOf[x.key]
                      const off = disabled.includes(x.key)
                      return (
                        <div key={x.key} id={`${id}-${idx}`} role="option" aria-selected={idx === sel} aria-disabled={off || undefined}
                          className={`picker-item${x.key === value ? ' current' : ''}${off ? ' off' : ''}`}
                          onMouseMove={() => idx !== sel && setSel(idx)} onClick={() => pick(x.key)}>
                          <MarkIcon fam={x.family} size={12} open={p.budget !== QUICK} ring={x.kind === 'ladder'} />
                          <span className="pi-name">{x.name}<span className="pi-size">{x.size}</span></span>
                          <span className="pi-acc">{fmt(p.acc)}%</span>
                          <span className="pi-cost">${fmt(p.cost)}</span>
                        </div>
                      )
                    })}
                  </div>
                )
              })}
              {!flat.length && <div className="picker-empty">No judge matches.</div>}
            </div>
            <div className="picker-foot">Accuracy at B = {QUICK} (open marks: B = {count(8192)}, judges that always reason) · est. $ per 1,000 verdicts</div>
          </div>
      )}
    </div>
  )
}
