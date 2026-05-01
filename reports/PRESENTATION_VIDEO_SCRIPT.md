# Boston Bus Equity - 10-Minute Presentation Video Script

**Course:** CS506 Spring 2026
**Total runtime:** 10:00
**Target word count:** ~300 spoken words (avg 30 wpm — slow, deliberate; let the visuals do the work)
**Recording setup (dashboard-only, no code shots):**
- Run `python3 -m src.inference.serve --port 8765` in a terminal
- Browser at `http://127.0.0.1:8765`, full-screen, Cmd+Shift+R first
- That's it — the entire video is filmed against the dashboard UI

---

## Pacing legend
- `[VISUAL]` = what is on-screen
- `[ACTION]` = what to click / scroll / open
- `[SAY]` = the exact words to speak
- Each block lists its target wall-clock duration

---

### 0:00 – 1:00 — Title + problem statement (30 words)

`[VISUAL]` Dashboard hero section, full screen, large H1 visible.

`[ACTION]` Sit on the hero for 5 seconds, then slow-scroll down once.

`[SAY]` "Boston Bus Equity. CS506, Spring 2026. We analyze MBTA bus reliability, then we predict delays in real time. One million riders depend on this system every day."

---

### 1:00 – 2:00 — Problem definition (30 words)

`[VISUAL]` Section *Problem definition* — the four background cards.

`[ACTION]` Cursor on each card briefly: *The problem*, *Two-part scope*, *Data scale*, *Headline result*.

`[SAY]` "Two parts. Equity analysis: Q1 to Q7. Delay prediction: Q8. One hundred sixty-one million records, twenty-two neighborhoods, six iterations from R-squared negative point zero seven to point nine nine four two."

---

### 2:00 – 3:00 — Equity findings (Q1, Q4, Q5) (35 words)

`[VISUAL]` KPI grid → Visualizations section, click filter to *Q5 target routes*.

`[ACTION]` Hover over the *Target-route delay gap +41%* KPI; then scroll to the *Target Routes* figure.

`[SAY]` "Three numbers tell the equity story. Ridership down thirty-three percent post-pandemic. Citywide delay seven point five minutes, on-time only thirty-one percent. The fifteen target routes serving low-income and minority neighborhoods are forty-one percent worse than the rest."

---

### 3:00 – 4:00 — Model progression V1 → V6 (35 words)

`[VISUAL]` *Modeling* section: experiment table + RMSE bar chart.

`[ACTION]` Slide-in to the table, then point at the V1 row and the V6 row to show the contrast.

`[SAY]` "Six iterations. V1 baseline: R-squared negative zero point zero seven, worse than the mean. V6 Transformer: R-squared zero point nine nine four. Ninety-three percent RMSE reduction. The features did it, not the architecture."

---

### 4:00 – 5:00 — V3 breakthrough + ablation (30 words)

`[VISUAL]` *V3 Time Series* training curves figure, then *V3 Feature Extraction Ablation* figure.

`[ACTION]` Click the *Modeling* filter on the visualization grid, open the V3 ablation card.

`[SAY]` "V3 is the breakthrough. Lag features plus FFT plus Daubechies wavelet. Adding past delay history lifts R-squared by one point zero five absolute. Rolling statistics contributes most among the individual signal-processing methods."

---

### 5:00 – 6:00 — Live demo, single prediction (35 words)

`[VISUAL]` *Realtime inference Step 1* form, full panel.

`[ACTION]`
1. Open *Model* dropdown — pause one second on V6 Transformer (default).
2. Route `1`, Stop `110`, scheduled time = current hour, direction Outbound.
3. Click *Predict delay*.

`[SAY]` "Step one. Pick a model. We default to V6 Transformer, fully retrained on three point seven six million samples. Route one, stop one-ten. Submit. The prediction returns in under ten milliseconds with the test R-squared shown next to the model name."

---

### 6:00 – 7:00 — Live MBTA comparison, V6 (35 words)

`[VISUAL]` *Realtime inference Step 2* live comparison panel + chart with Plotly red/blue lines.

`[ACTION]` Click *Fetch live comparison*. Wait for the chart to populate. Hover one of the blue MBTA bars.

`[SAY]` "Step two. We pull MBTA's own live predictions for the next eight upcoming trips on the same route-stop, then run our model on each one. The red line is us, the blue line is MBTA. The mean absolute gap is shown above the chart."

---

### 7:00 – 8:00 — The April surprise: V3 beats V6 on live (40 words)

`[VISUAL]` Switch model picker to *V3 Time Series*. Re-fetch live comparison.

`[ACTION]`
1. Step 1 dropdown → *V3 GRU - lag + FFT + wavelet*.
2. Step 2 → *Fetch live comparison* again.
3. Show the smaller mean_abs_gap number.

`[SAY]` "Now switch to V3 GRU. The mean absolute gap drops by half. V3 has worse offline R-squared but it tracks MBTA's live predictions much more closely. This is our most interesting April finding, and it is reproducible in this demo right now."

---

### 8:00 – 9:00 — Why: bias-variance (40 words)

`[VISUAL]` Stay on the dashboard. Scroll to the *Modeling V1 -> V6* section, point at V3's R^2 = 0.9846 and V6's R^2 = 0.9942. Then back to the live-compare panel.

`[ACTION]` Cursor on V3 row, then V6 row in the experiment table. Then point at the live-compare gap difference you just demonstrated.

`[SAY]` "The cause is bias-variance. Live MBTA data is noisier than training. V6's larger model amplifies that noise. V3's wavelet features compress lag spikes, so V3 is high-bias but low-variance, and high-variance models lose on noisy live data."

---

### 9:00 – 10:00 — Engineering response + close (30 words)

`[VISUAL]` Stay on the dashboard.

`[ACTION]`
1. Step 1 dropdown → select *V3+V6 Ensemble*.
2. Click *Predict delay*. The result panel shows both legs averaged.
3. End on the dashboard hero (scroll to top).

`[SAY]` "Three responses are already shipping. A V3-plus-V6 ensemble you can pick right here. A noise-injection retraining of V6 in progress. A matched-actuals daemon collecting ground truth for the final report. Everything is live on GitHub. Thank you."

---

## Word-count check
| Block | Target | Cumulative |
|-------|--------|-----------:|
| 0:00–1:00 | 30  | 30  |
| 1:00–2:00 | 30  | 60  |
| 2:00–3:00 | 35  | 95  |
| 3:00–4:00 | 35  | 130 |
| 4:00–5:00 | 30  | 160 |
| 5:00–6:00 | 35  | 195 |
| 6:00–7:00 | 35  | 230 |
| 7:00–8:00 | 40  | 270 |
| 8:00–9:00 | 40  | 310 |
| 9:00–10:00 | 30 | 340 |

Total ≈ **340 spoken words**. At ~30 wpm average pace this lands at exactly 10:00, with extra silence built in for visual reveals (recommended: 2–3 second pauses on KPI cards, 5 second pause on the live chart while it animates).

---

## Pre-recording checklist
- [ ] `python3 -m src.inference.serve --port 8765` running
- [ ] Browser at `http://127.0.0.1:8765`, hard-refreshed (Cmd+Shift+R) — V3+V6 Ensemble visible in the model picker
- [ ] One sanity prediction worked (`Predict delay` returns a number)
- [ ] One sanity live-compare worked (`Fetch live comparison` returns a chart)
- [ ] Browser zoom 100%, full-screen
- [ ] Recording resolution: 1920×1080
- [ ] Mic test, no fans, no notifications
