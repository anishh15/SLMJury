import { Icon } from '../components/ui'
import '../styles/content.css'

export default function NotFound({ route }) {
  return (
    <section className="wrap notfound">
      <h1 className="h-display" style={{ fontSize: 'clamp(2rem,4vw,3rem)', marginBottom: 14 }}>No page at <code>#{route.raw}</code></h1>
      <a className="btn btn-primary" href="#/">Back to the overview <Icon.arrow /></a>
    </section>
  )
}
