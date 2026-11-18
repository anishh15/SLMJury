import { memo } from 'react'

// The hero's backdrop, after the README banner, kept quiet: a starfield, faint
// circuit traces that carry a verdict in now and then, and a low grid floor.
// Static SVG plus a few CSS animations, paused off-screen.
const W = 1440
const H = 760
const CX = W / 2
const GH = 90

function rand(seed) {
  let s = seed
  return () => { s = (s * 16807) % 2147483647; return (s - 1) / 2147483646 }
}
const stars = (() => {
  const r = rand(7)
  return Array.from({ length: 90 }, (_, i) => ({ x: r() * W, y: r() * H * 0.86, r: 0.5 + r() * 1, o: 0.2 + r() * 0.55, tw: i % 11 === 0 }))
})()

// polylines from the edges towards the centre (left side; the right mirrors it)
const LEFT = [
  [[120, 150], [270, 150], [270, 236]],
  [[200, 236], [270, 236], [270, 318], [330, 318]],
  [[90, 410], [196, 410], [196, 318], [270, 318]],
]
const VERDICT = ['var(--v-correct)', 'var(--v-incorrect)', 'var(--v-correct)']

function Atmosphere() {
  const traces = []
  ;[1, -1].forEach((side) => {
    LEFT.forEach((pts, i) => {
      const P = pts.map(([x, y]) => [side === 1 ? x : W - x, y])
      traces.push({ key: `${side}-${i}`, d: `M${P.map(([x, y]) => `${x},${y}`).join('L')}`, at: P[0], i, side })
    })
  })
  const rays = []
  for (let k = -9; k <= 9; k++) rays.push(k)
  const rows = [0.18, 0.42, 0.74]
  return (
    <>
      <svg className="atmo atmo-sky" viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="xMidYMid slice" aria-hidden="true" focusable="false">
        <g className="atmo-stars">
          {stars.map((st, i) => <circle key={i} cx={st.x.toFixed(1)} cy={st.y.toFixed(1)} r={st.r.toFixed(2)} opacity={st.o.toFixed(2)} className={st.tw ? 'tw' : undefined} style={st.tw ? { animationDelay: `${(i % 7) * -0.8}s` } : undefined} />)}
        </g>
        <g className="atmo-traces">
          {traces.map((t) => (
            <g key={t.key}>
              <path d={t.d} className="trace" />
              <path d={t.d} pathLength="100" className="pulse" style={{ stroke: VERDICT[t.i], animationDuration: `${7 + 1.3 * t.i}s`, animationDelay: `${-(2.2 * t.i + (t.side < 0 ? 3.1 : 0))}s` }} />
              <circle cx={t.at[0]} cy={t.at[1]} r="2.6" className="pad" />
            </g>
          ))}
        </g>
      </svg>
      <svg className="atmo atmo-ground" viewBox={`0 0 ${W} ${GH}`} preserveAspectRatio="xMidYMin slice" aria-hidden="true" focusable="false">
        <g className="atmo-floor">
          {rays.map((k) => <line key={k} x1={CX + k * 40} y1={0} x2={CX + k * 150} y2={GH} />)}
          {rows.map((t) => {
            const hw = 360 + (CX + 400 - 360) * t
            return <line key={t} x1={CX - hw} y1={GH * t} x2={CX + hw} y2={GH * t} className="row" />
          })}
        </g>
      </svg>
    </>
  )
}

export default memo(Atmosphere)
