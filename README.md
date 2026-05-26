# IPL Crunch '26 🏏

**A statistical audit of 19 IPL seasons. 294,757 balls. 10 verdicts. One predicted champion — confirmed live.**

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![pandas](https://img.shields.io/badge/pandas-2.x-150458?style=flat-square)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-F7931E?style=flat-square)
![UMAP](https://img.shields.io/badge/UMAP-clustering-green?style=flat-square)
![Submission](https://img.shields.io/badge/Wooble-IPL%20Crunch%20'26-1db954?style=flat-square)

---

## What This Is

A single Python pipeline (`ipl_crunch_full.py`) that ingests ball-by-ball IPL data, runs statistical analysis and machine learning, builds a prediction model, and outputs one fully self-contained interactive HTML file. No frameworks, no server, no external dependencies — judges open one file in a browser and see everything.

**Live call:** Model predicted RCB to beat GT in Q1 (May 26, Dharamsala). RCB won.

---

## The 10 Findings

| # | Finding | Key Number |
|---|---------|-----------|
| 1 | Toss winners win at coin-flip rates — the **decision** matters, not the toss | 51.3% toss WR · 44.3% bat · 53.7% field |
| 2 | Death overs decide matches, not powerplays | +1.77 RPO gap vs +0.95 powerplay |
| 3 | Over 7 is the **cheapest over in T20** — cheaper than the first | 6.99 RPO vs 8.70 (over 6) |
| 4 | The real chase wall is **175**, not 200 | Chasing 170s: 40% · 200+: 23% |
| 5 | Powerplay wickets lost = match lost | 0 wkts in PP: ~70% WR · 4+ wkts: 27% · **Δ43pp** |
| 6 | Pre-2022 IPL is statistically a **different sport** | Chow test p<0.05 · 8.31→9.81 RPO |
| 7 | Only 4 batter archetypes exist in T20 | UMAP + KMeans on 12-feature matrix |
| 8 | Momentum is **statistically real** | 12+ over → next avg 9.5 · quiet → 7.8 |
| 9 | Ahmedabad structurally favours batting first | 60%+ bat-first WR · 2026 Final venue |
| 10 | CSK wins at **exactly 60%** regardless of toss decision | Bat: 60.0% · Field: 60.0% |

---

## The Prediction Model

11-signal composite score, applied to all 4 playoff-qualified teams.

```
Signal              Weight   Detail
─────────────────────────────────────────────────────────────────
2026 Form             30%    Current-season win rate
EWMA Win Rate         10%    Exponentially-weighted 2022–25 (decay 0.5/season)
Death Batting         10%    60% 2026 + 40% historical death RPO blend
Player Archetypes      9%    UMAP cluster quality score
Venue Fit              9%    70% Ahmedabad (Final) + 30% Mullanpur (Q2/Elim)
Death Bowling          8%    Death economy, inverted
H2H vs Strong          6%    Win rate vs playoff-grade teams 2022–26
Dominance              5%    Average 2026 win margin
Home Advantage         5%    Historical home-ground win rate
NRR                    4%    Live 2026 net run rate (Wikipedia-scraped)
Toss Effect            4%    Team-specific toss-to-win correlation
```

**50,000 Monte Carlo bracket simulations → 2026 champion probabilities:**

| Team | Probability |
|------|-------------|
| **RCB** | **45.2%** ← predicted |
| GT | 38.0% |
| RR | 10.1% |
| SRH | 6.8% |

**Backtest (walk-forward, 16 seasons, zero data leakage):**
- Top-4 accuracy: **75%** (vs 40% random baseline)
- Top-2 accuracy: **50%**
- Exact winner: **19%**

---

## How to Run

### 1. Install dependencies
```bash
pip install pandas numpy matplotlib scipy scikit-learn umap-learn requests beautifulsoup4
```

### 2. Add your data
Download the Cricsheet IPL ball-by-ball CSV and update line 28 of `ipl_crunch_full.py`:
```python
CSV_PATH = r"C:\path\to\your\ipl_data.csv"   # update this
```

### 3. Run the pipeline
```bash
python ipl_crunch_full.py
```

**Output:**
- `charts/` — 8 PNG charts
- `ipl_crunch_deliverable.html` — full interactive deliverable (self-contained)
- Terminal output — all 10 findings + counterfactuals printed as formatted verdict blocks

### 4. Optional: update live 2026 data
```bash
python fetch_standings.py          # scrapes Wikipedia → standings_2026.json
python append_new_matches.py       # pulls latest matches from Cricsheet JSON zip
```

---

## File Structure

```
ipl-crunch-26/
├── ipl_crunch_full.py          # Main pipeline — run this
├── fetch_standings.py          # Live Wikipedia standings scraper
├── append_new_matches.py       # Cricsheet JSON → CSV row appender
├── ipl_crunch_deliverable.html # Self-contained interactive output
├── charts/
│   ├── 01_toss_analysis.png
│   ├── 02_phase_analysis.png
│   ├── 03_top_performers.png
│   ├── 04_phase_prediction.png
│   ├── 05_bowler_map.png
│   ├── 06_powerplay_hangover.png
│   ├── 07_scoring_evolution.png
│   └── 08_wall_and_momentum.png
└── README.md
```

---

## Tech Stack

| Category | Tools |
|----------|-------|
| Data Engineering | pandas, numpy |
| Statistics | scipy (Chow, Levene, Breusch-Pagan, Durbin-Watson, ACF tests) |
| Machine Learning | scikit-learn (KMeans), umap-learn (UMAP) |
| Visualisation | matplotlib |
| Live Data | requests, BeautifulSoup (Wikipedia scraper) |
| Output | HTML/CSS/JS generated as Python f-strings |

---

## Data Source

**Cricsheet.org** — CC-BY-SA-4.0  
294,757 deliveries · 1,239 matches · 19 seasons (2007/08–2026)

Each row = one delivery: batter, bowler, runs, extras, wicket info, venue, toss, batting team, match ID, season.

---

## Data Quality Fixes Applied

Seven bugs caught and fixed during development:

1. `LAST5` season window updated to include 2026
2. No-result and tie matches correctly awarded 1 point (not 0) in standings fallback
3. Strike rate computed on `legal_ball` column only (excludes wides/no-balls)
4. `BOWLER_TYPE` dictionary expanded from ~26 to ~80 entries (78.7% delivery coverage)
5. `BATTER_HAND` dictionary expanded from ~26 to ~100 entries (72.6% delivery coverage)
6. RCB name normalised from `Royal Challengers Bengaluru` → `Royal Challengers Bangalore` at load time
7. Per-match consistency requires ≥6 legal balls per innings to exclude not-out cameos

---

## Key Numbers

```
294,757   total deliveries
  1,239   matches
     19   seasons (2008–2026)
     75%  model top-4 backtest accuracy
     40%  random baseline
  50,000  Monte Carlo simulations per run
     11   signals in composite model
     43pp powerplay wicket swing (biggest finding)
   51.3%  toss winner win rate (the myth-killer)
   2022   year IPL structurally changed (Chow test confirmed)
```

---

## Counterfactuals

### 2019 IPL Final — MI def. CSK by 1 run

- MI scored 149 — historical win probability at that score: **28%** (underdogs)
- If MI had scored 10 more in death overs (47→57): win prob jumps **28% → 37%**
- Watson scored 80 for CSK. If dismissed for 30 (CSK score ~108): CSK win prob collapses to **6%**
- Watson personally lifted CSK from ~6% to 48% win probability. One innings. One run margin.

### SRH Toss Mismanagement (All Seasons)

- Won toss, chose bat: 38 times → won **34.2%**
- Won toss, chose field: 59 times → won **49.2%**
- Estimated matches lost by over-batting: **~6 matches**

---

## License

Data: Cricsheet.org (CC-BY-SA-4.0). All analysis code original.

---

*Built for Wooble IPL Crunch '26 · Suhaan Aneja*
