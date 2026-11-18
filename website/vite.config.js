import { readFileSync } from 'node:fs'
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// index.html's meta tags take their counts and venue from the generated site data ({{facts.nSmall}} and so on).
function pageMeta() {
  const core = JSON.parse(readFileSync(new URL('./src/data/core.json', import.meta.url), 'utf8'))
  return {
    name: 'slmjury-page-meta',
    transformIndexHtml(html) {
      return html.replace(/\{\{([\w.]+)\}\}/g, (m, path) => {
        const v = path.split('.').reduce((o, k) => (o == null ? o : o[k]), core)
        if (v == null) throw new Error(`index.html: no value for {{${path}}} in core.json`)
        return String(v)
      })
    },
  }
}

// The site is served from https://anishh15.github.io/SLMJury/, so every asset
// path is under /SLMJury/. Routing uses the URL hash, so deep links work on
// GitHub Pages without a server-side fallback.
export default defineConfig({
  base: '/SLMJury/',
  plugins: [react(), pageMeta()],
  build: {
    target: 'es2020',
    cssCodeSplit: true,
    assetsInlineLimit: 2048,
    chunkSizeWarningLimit: 400,
  },
})
