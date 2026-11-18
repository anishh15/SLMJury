// Chart exports: CSV of a chart's table, PNG of its SVG (with a small
// SLMJury watermark), and a link to the chart.

export function toCSV(columns, rows) {
  const cell = (v) => {
    if (v == null) return ''
    const s = String(v)
    return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s
  }
  const head = columns.map((c) => cell(c.label)).join(',')
  const body = rows.map((r) => columns.map((c) => cell(r[c.key])).join(','))
  return [head, ...body].join('\n') + '\n'
}

export function download(name, data, type) {
  const blob = data instanceof Blob ? data : new Blob([data], { type })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = name
  document.body.appendChild(a)
  a.click()
  a.remove()
  setTimeout(() => URL.revokeObjectURL(url), 1500)
}

export const slugify = (s) => String(s).toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '')

export function chartLink(id) {
  const path = (window.location.hash.replace(/^#/, '') || '/').split('?')[0]
  const q = new URLSearchParams(window.location.hash.split('?')[1] || '')
  q.set('chart', id)
  return `${window.location.origin}${window.location.pathname}#${path}?${q.toString()}`
}

// ── PNG ─────────────────────────────────────────────────────────────────────
const PROPS = [
  'fill', 'fill-opacity', 'stroke', 'stroke-width', 'stroke-opacity', 'stroke-dasharray', 'stroke-linecap',
  'stroke-linejoin', 'opacity', 'font-family', 'font-size', 'font-weight', 'font-style', 'letter-spacing',
  'text-anchor', 'dominant-baseline', 'paint-order', 'visibility', 'display', 'transform', 'transform-origin',
]

let fontCss = null
async function embeddedFonts() {
  if (fontCss) return fontCss
  const rules = []
  for (const sheet of [...document.styleSheets]) {
    let list
    try { list = sheet.cssRules } catch (e) { continue }
    for (const r of [...(list || [])]) {
      if (r.type !== 5) continue
      const fam = r.style.getPropertyValue('font-family').replace(/['"]/g, '').trim()
      if (!/^Plex Sans$|^Newsreader$/.test(fam)) continue
      const m = r.style.getPropertyValue('src').match(/url\(["']?([^"')]+)["']?\)/)
      if (!m) continue
      rules.push({ fam, weight: r.style.getPropertyValue('font-weight') || '400', url: new URL(m[1], sheet.href || window.location.href).href })
    }
  }
  const out = await Promise.all(rules.map(async (f) => {
    try {
      const buf = await (await fetch(f.url)).arrayBuffer()
      let bin = ''
      const bytes = new Uint8Array(buf)
      for (let i = 0; i < bytes.length; i += 0x8000) bin += String.fromCharCode.apply(null, bytes.subarray(i, i + 0x8000))
      return `@font-face{font-family:'${f.fam}';font-weight:${f.weight};src:url(data:font/woff2;base64,${btoa(bin)}) format('woff2')}`
    } catch (e) { return '' }
  }))
  fontCss = out.join('\n')
  return fontCss
}

function inlineStyles(src, dst) {
  const cs = getComputedStyle(src)
  const style = PROPS.map((p) => {
    const v = cs.getPropertyValue(p)
    return v && v !== 'none' && v !== 'normal' && v !== 'auto' ? `${p}:${v}` : (p === 'transform' || p === 'display' ? '' : '')
  }).filter(Boolean).join(';')
  dst.setAttribute('style', style)
  dst.removeAttribute('class')
  const a = src.children
  const b = dst.children
  for (let i = 0; i < a.length; i++) if (b[i]) inlineStyles(a[i], b[i])
}

const cssVar = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim()

function loadImage(url) {
  return new Promise((resolve, reject) => {
    const img = new Image()
    img.onload = () => resolve(img)
    img.onerror = reject
    img.src = url
  })
}

function wrapText(ctx, text, maxW) {
  const words = String(text).split(/\s+/)
  const lines = []
  let line = ''
  for (const w of words) {
    const t = line ? `${line} ${w}` : w
    if (ctx.measureText(t).width > maxW && line) { lines.push(line); line = w } else line = t
  }
  if (line) lines.push(line)
  return lines
}

export async function exportPNG(svg, { title, sub, source, filename }) {
  const box = svg.getBoundingClientRect()
  const w = Math.round(box.width)
  const h = Math.round(box.height)
  const clone = svg.cloneNode(true)
  inlineStyles(svg, clone)
  clone.setAttribute('xmlns', 'http://www.w3.org/2000/svg')
  clone.setAttribute('width', w)
  clone.setAttribute('height', h)
  const style = document.createElementNS('http://www.w3.org/2000/svg', 'style')
  style.textContent = await embeddedFonts()
  clone.insertBefore(style, clone.firstChild)
  clone.querySelectorAll('.hit, [data-export="skip"]').forEach((n) => n.remove())
  const xml = new XMLSerializer().serializeToString(clone)
  const img = await loadImage(`data:image/svg+xml;charset=utf-8,${encodeURIComponent(xml)}`)

  const scale = 2
  const pad = 32
  const W = w + pad * 2
  const probe = document.createElement('canvas').getContext('2d')
  probe.font = '400 14px "Plex Sans", sans-serif'
  const subLines = sub ? wrapText(probe, sub, W - pad * 2) : []
  const top = pad + 26 + subLines.length * 20 + 18
  const H = top + h + 52
  const canvas = document.createElement('canvas')
  canvas.width = W * scale
  canvas.height = H * scale
  const ctx = canvas.getContext('2d')
  ctx.scale(scale, scale)
  ctx.fillStyle = cssVar('--surface') || '#0a1530'
  ctx.fillRect(0, 0, W, H)
  ctx.fillStyle = cssVar('--ink')
  ctx.font = '600 19px "Plex Sans", sans-serif'
  ctx.fillText(title || '', pad, pad + 18)
  ctx.fillStyle = cssVar('--muted')
  ctx.font = '400 14px "Plex Sans", sans-serif'
  subLines.forEach((l, i) => ctx.fillText(l, pad, pad + 44 + i * 20))
  ctx.drawImage(img, pad, top, w, h)
  // footer: source on the left, the watermark on the right
  ctx.strokeStyle = cssVar('--line')
  ctx.beginPath(); ctx.moveTo(pad, H - 38.5); ctx.lineTo(W - pad, H - 38.5); ctx.stroke()
  ctx.fillStyle = cssVar('--faint')
  ctx.font = '400 12px "Plex Sans", sans-serif'
  if (source) ctx.fillText(source, pad, H - 16)
  ctx.globalAlpha = 0.75
  ctx.textAlign = 'right'
  ctx.font = 'italic 640 17px "Newsreader", Georgia, serif'
  const jw = ctx.measureText('Jury').width
  ctx.fillStyle = cssVar('--cyan')
  ctx.fillText('Jury', W - pad, H - 14)
  ctx.font = '700 17px "Newsreader", Georgia, serif'
  ctx.fillStyle = cssVar('--ink')
  ctx.fillText('SLM', W - pad - jw - 1, H - 14)
  ctx.globalAlpha = 1
  const blob = await new Promise((r) => canvas.toBlob(r, 'image/png'))
  download(filename, blob)
}
