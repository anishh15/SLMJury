import { claims, facts } from './data'
import { word } from './format'

// The six findings, as the paper states them; every number comes from claims and facts in core.json.
export const FINDINGS = [
  {
    slug: 'scale',
    title: 'Reliable verification does not need a large judge',
    short: 'Small judges are enough',
    big: <>±{claims.margin}<small>points</small></>,
    summary: <>Phi-4 ({claims.phiSize}) is equivalent to {claims.topLargeName} within ±{claims.margin} points and costs an estimated {claims.costLo} to {claims.costHi} times less to serve.</>,
    details: 'Accuracy by size at both budgets, and what each judge costs',
  },
  {
    slug: 'budget',
    title: 'Overthinking depends on the domain, and on the prompt',
    short: 'Overthinking',
    big: <>{claims.unchanged}<small>of {claims.nBudget}</small></>,
    summary: <>Raising only the budget leaves {claims.unchanged} of {claims.nBudget} judges unchanged. The quick-versus-reasoned difference comes from the prompt.</>,
    details: 'All four single-factor tests and the budget sweep',
  },
  {
    slug: 'domain',
    title: 'The math-to-general gap belongs to the task',
    short: 'Domain gap',
    big: <>{claims.strongGap[0]}–{claims.strongGap[1]}<small>points</small></>,
    summary: <>The gap reaches {claims.gapHi} points and stays at {claims.strongGap[0]}–{claims.strongGap[1]} for the strongest judges from {claims.saturation[1]} to {facts.largeRange[1]}.</>,
    details: 'Accuracy for every judge on every dataset',
  },
  {
    slug: 'open-ended',
    title: 'Closed- and open-ended judging rank judges differently',
    short: 'Open-ended scoring',
    big: <>{claims.seMatch}<small>of {claims.seN}</small></>,
    summary: <>On human-scored summaries the top small judges match or beat the large ones in {claims.seMatch} of {claims.seN} paired comparisons.</>,
    details: 'Every paired comparison and agreement for every judge',
  },
  {
    slug: 'strategies',
    title: 'Multi-agent debate degrades accuracy; adversarial personas stay within a point',
    short: 'Personas, juries, debate',
    big: <>−{claims.debateGap}<small>points</small></>,
    summary: <>Multi-agent debate degrades accuracy in every configuration tested. The two strongest judges stay within a point across {word(facts.nPersonas)} adversarial personas.</>,
    details: 'Personas, rewordings, protocols and answer-order swaps',
  },
  {
    slug: 'students',
    title: 'Judges rank students by quality, not by lineage',
    short: 'Students and lineage',
    big: <>τ = {claims.studentsTauSolver}</>,
    summary: <>Judges rank {word(claims.nStudentsWide)} students by the students’ own solver accuracy, with no lineage preference at the quick verdict.</>,
    details: 'The students and the lineage effect at both budgets',
  },
]
