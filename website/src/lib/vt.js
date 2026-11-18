// Shared-element transitions between a leaderboard row and its judge page:
// the row's mark and name get the view-transition names the judge page's
// header uses, just before the route changes.

let tagged = []
let last = null

export function clearTags() {
  tagged.forEach((el) => { if (el) el.style.viewTransitionName = '' })
  tagged = []
}

export function tagJudge(root) {
  clearTags()
  if (!root) return
  const mark = root.querySelector('[data-vt="mark"]')
  const name = root.querySelector('[data-vt="name"]')
  if (mark) mark.style.viewTransitionName = 'judge-mark'
  if (name) name.style.viewTransitionName = 'judge-name'
  tagged = [mark, name]
}

export const setLastJudge = (k) => { last = k }
export const lastJudge = () => last
