# Presentation — Reward Encoding in the Human Brain

A [reveal.js](https://revealjs.com) slide deck (`index.html`) prototyping the preliminary
project talk. Built per `../presentation.md`.

## How to present

Just open `index.html` in a browser (Chrome/Edge/Firefox). No build step — reveal.js,
KaTeX (equations) and the Chart.js visualizer all load from CDN, so **you need an
internet connection** the first time (same as `zeta_mathematical_visualizer.html`).

Keys while presenting:

| Key | Action |
|---|---|
| `→` / `←` / `Space` | next / previous slide |
| `Esc` or `o` | slide overview (grid) |
| `F` | fullscreen |
| `S` | open **speaker-notes** window (presenter view with timer + notes) |
| `B` / `.` | black-out screen |

## The interactive slides

Two slides embed a visualizer **live** as a full-slide iframe — the sliders work inside the
presentation. Because they capture the mouse, leave those slides with the arrow keys (or `Esc`).

- **Slide 9** — `zeta_mathematical_visualizer.html` (spike-timing / ZETA).
- **Slide 17** — `hgf_mathematical_visualizer.html` (perceived probability p̂ & prediction
  error δ₁): drag ω₂ to watch the belief chase the switching reward world; Tab 2 contrasts a
  fast vs. slow learner. Faithful to `analysis/hgf/model.py` (matches pyhgf to ~3e-4).

## Export to PDF / PowerPoint

- **PDF:** open `index.html?print-pdf` in Chrome → Print → *Save as PDF* (Background
  graphics ON, margins None). The live ZETA slide exports as a static frame.
- **PowerPoint:** export the PDF, or (better, keeps interactivity) keep the slide(s) that
  need the live visualizer as links to `zeta_mathematical_visualizer.html`.

## Figures — placeholders

Every result/data figure is a dashed **placeholder box** that states what it should show
and names the strongest existing candidate file under `../results/`. Workflow agreed with
you: you explore the data and give me the plot spec (or approve a candidate), and I drop
the real figure in. Placeholders to fill:

| Slide | Figure | Candidate file (if any) |
|---|---|---|
| 4 | Task schematic | none — render in matplotlib or draw |
| 7 | ZETA intuition cartoon | still frame of the visualizer (optional) |
| 10 | PSTH, choice-only neuron | from `zeta_outcome/zeta2_choice_*.csv` ∩ ¬`zeta2_reward_*.csv` |
| 10 | PSTH, reward-only neuron | from `zeta_outcome/zeta2_reward_*.csv` ∩ ¬`zeta2_choice_*.csv` |
| 10 | Regional dissociation bars | `results/report/fig_outcome_by_region.png` |
| 14 | Model comparison ΔBIC | `results/hgf/figures/model_comparison.png` |
| 14 | Behaviour overview | `results/report/fig_choice_vs_value.png`, `results/figures/behaviour_overview_all.png` |
| 17 | HGF visualizer | live `hgf_mathematical_visualizer.html` (embedded, like slide 9) ✓ |
| 18 | PE vs firing rate | generate via `viewers/firing_rate_vs_delta1.py` |
| 18 | p̂ vs firing rate | generate via `viewers/firing_rate_vs_hgf_p.py` |

When we add real figures, copy them into `presentation/figures/` so the folder stays
portable, and swap the `.placeholder` block for an `<img>`.
