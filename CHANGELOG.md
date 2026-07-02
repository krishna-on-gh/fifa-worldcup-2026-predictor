# Changelog

A running log of updates to the FIFA World Cup 2026 prediction model and dashboard.
Add a new dated section at the top each time you ship something.

**June 28, 2026**
- **R32 Fixtures bug fix** — R32 fixtures weren't proper before, so I redid them and manually hardcoded them in
- **Host advantage removed** — Removed host advantage for Mexico, USA, and Canada for the knockouts
- **Fixed accuracy tracking bug** — The accuracy of the model was shown, but not like before and was inconsistent across tabs. So, I ended up making accuracy consistent across tabs and I also added a new metric where it separates accuracy by group stage and knockouts. Now, viewers can see the accuracy for the total model, just the group stage, and just the knockouts.

**June 24, 2026**
- **Full group-stage simulation** — R32 and advancement odds are now true probabilities (the remaining group games are simulated, not assumed to a fixed projected field).
- **Track Record tab** — prediction record, championship-odds-over-time, and this changelog.
- **Live in-game win probability** — a Bayesian model that updates each match's odds in real time from the score and clock, with a next-game preview.

**June 23, 2026**
- **Clinch detection** — teams are flagged the moment they've mathematically secured a Round-of-32 spot.
- **Goal-difference tiebreaks** — group standings break ties on actual GD, then goals scored.
- **Real group stats** — added Games Played, Points, and GD columns plus a third-place race table.

**June 22, 2026**
- **Results-conditioned odds** — knockout results lock into the simulation, so eliminated teams drop to 0% and everyone else recomputes.
- **Interactive bracket** — hover any team to trace its route to the final and see its stage odds.
