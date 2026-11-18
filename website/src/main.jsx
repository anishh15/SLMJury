import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import App from './App'
import { initialRoute, preload } from './lib/router'
import './styles/tokens.css'
import './styles/base.css'
import './styles/shell.css'
import './styles/home.css'
import './styles/charts.css'
import sans400 from '@fontsource/ibm-plex-sans/files/ibm-plex-sans-latin-400-normal.woff2?url'
import sans500 from '@fontsource/ibm-plex-sans/files/ibm-plex-sans-latin-500-normal.woff2?url'
import sans600 from '@fontsource/ibm-plex-sans/files/ibm-plex-sans-latin-600-normal.woff2?url'
import serif from './fonts/newsreader-display.woff2?url'

// Fetch the text fonts while the first page loads, so text does not reflow when they arrive.
for (const href of [sans400, sans500, sans600, serif]) {
  const l = document.createElement('link')
  Object.assign(l, { rel: 'preload', as: 'font', type: 'font/woff2', crossOrigin: 'anonymous', href })
  document.head.appendChild(l)
}

// A deep link waits for its page module, so the first paint is the page itself.
const first = initialRoute()
preload(first.page, first).finally(() => {
  createRoot(document.getElementById('root')).render(
    <StrictMode>
      <App />
    </StrictMode>,
  )
})
