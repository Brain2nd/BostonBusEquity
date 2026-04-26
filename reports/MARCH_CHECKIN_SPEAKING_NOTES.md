# March Check-In Speaking Notes / 三月汇报发言稿

> 对照 `MARCH_CHECKIN_PRESENTATION.md` 进行汇报。
> 建议汇报时间：10-15分钟。
> 【方括号】里的内容是给汇报者的提示，不用念出来。

---

## Opening（30秒）

Hello everyone. Our project is Boston Bus Equity — we analyze MBTA bus service performance and its equity implications across Boston neighborhoods.

We have a base analysis answering 7 research questions about ridership, delays, and demographics, plus an extended research on delay prediction using machine learning. I'll walk through our progress following the four rubric categories.

---

## Part 1: Preliminary Data Visualizations（4-5分钟）

We produced 22 visualizations covering all 8 research questions. Let me walk through the key ones.

【展示文档 Q1 — 两张图】

### Q1: Ridership

These two charts show MBTA bus ridership. The first is the overall trend — ridership dropped 51% during the pandemic and is still 32.8% below pre-pandemic levels. The second breaks it down by route — recovery is very uneven, some routes are near normal while others are still far below.

【展示文档 Q2 — 一张图】

### Q2: Travel Times

This shows end-to-end travel times per route. Average is 28.4 minutes, ranging from 8 to 89 minutes. Peak hours are 15-20% longer. Longer routes accumulate more delays.

【展示文档 Q3 — 一张图】

### Q3: Wait Times

On-time buses: passengers wait about 5 minutes. Delayed buses: 12-15 minutes. That's nearly triple. This shows why delays matter so much to the rider experience.

【展示文档 Q4 — 四张图，快速翻】

### Q4: Citywide Delays

Four charts showing delay patterns. Key numbers: only 31.7% on-time performance, 23% of buses more than 15 minutes late. By hour — clear evening rush peak at 4-7 PM. By day — weekends slightly better. By month — winter is worse. These predictable temporal patterns are exactly what our ML models try to capture.

【展示文档 Q5 — 一张图】

### Q5: Target Routes

The 15 target routes identified by Livable Streets have 41% higher delays than other routes — 10.2 versus 7.2 minutes. On-time performance is 6.5 points worse. This quantitatively confirms the equity concern.

【展示文档 Q6 — 三张图】

### Q6: Service Disparities

These three charts show the spread. Top 10% of routes: over 45% on-time. Bottom 10%: below 20%. Standard deviation is 15.2 points — a huge gap. Target routes cluster at the bottom. Higher-ridership routes tend to have more delays, suggesting capacity constraints.

【展示文档 Q7 — 三张图】

### Q7: Demographics

The heatmap is key — correlation between delays and minority population is -0.007, p-value 0.96, not significant. No correlation with income either. But the neighborhood map shows 6 "vulnerable" neighborhoods — Dorchester, Mattapan, Roxbury, East Boston, Hyde Park, Mission Hill — are disproportionately served by the underperforming target routes. So the picture is nuanced: no systematic demographic bias in delays, but vulnerable communities still bear the burden through worse-performing routes.

【展示文档 Q8 — 五张图，快速翻】

### Q8: Delay Prediction

【不需要细讲，后面建模部分详细说】

Briefly — V1 baseline curves show the model failing to converge. V3 curves show smooth convergence after adding temporal features. The ablation chart compares feature methods. NeuronSpark comparison shows SNN slightly beating GRU. Multi-step prediction is much harder than single-step. I'll explain all of this in the modeling section.

---

## Part 2: Data Processing Progress（3分钟）

【展示文档 Section 2】

### 2.1 数据来源

Our primary dataset is MBTA bus arrival and departure records — 161 million records, about 18 gigabytes, spanning 2020 to early 2026. Downloaded using an automated script with resume support.

We also have ridership data from 2016-2024, passenger survey data, Census demographics, and GTFS route/stop data. One note: 2018-2019 arrival data is no longer available on the MBTA portal, so delay analysis starts from 2020. Ridership goes back to 2016 for pre-pandemic comparison.

### 2.2 数据清洗

【指着展示文档的9步表格】

Our pipeline has 9 steps. Key highlights:

We process in chunks of 500K rows because 18 GB doesn't fit in memory. We calculate delay as actual minus scheduled time in minutes. We drop nulls and filter to negative 30 to positive 60 minutes — anything outside that is data errors. We do a strict temporal split: training is before 2025, test is 2025 and after, no overlap. We map 2,910 stops to 22 neighborhoods. And we convert to Parquet for 5x compression.

【指着代码片段】

You can see the core code here — delay calculation, filtering, temporal split, and the scaler protocol: fit_transform on train, transform-only on test. This prevents information leakage.

### 2.3 决策理由

【这部分评分很看重"为什么"】

The most important decision is the **strict temporal split**. We don't do random splits because that would let the model see future data — which is data leakage. In real deployment, you can only use history to predict the future.

All features use only **past values** — shift(1) for lags, delays[i-window:i] for rolling stats — the current index is always excluded. Scaler is fit on training only. And features are extracted per route-stop group via groupby, so we don't mix data across different routes.

These choices reduce our reported numbers compared to random splits, but they give honest, deployment-realistic results.

---

## Part 3: Modeling Methods（4-5分钟）

【展示文档 Section 3】

### 3.1 问题定义

Our task is single-step delay regression: given a bus stop's context and recent delay history, predict the next delay in minutes. Input is a 41-dimensional feature vector, output is one continuous value, trained with MSE loss.

Why this matters: real-time passenger info, proactive dispatching, resource allocation, and scientifically quantifying how predictable delays are.

### 3.2 特征选择

【逐类讲，对着展示文档的公式表格】

41 features in 7 categories.

**Lag features** — 7 dimensions. The previous 5 delays plus first and second differences. Computed with groupby shift, so only past values within the same route-stop pair.

**Rolling statistics** — 8 dimensions. Mean, std, min, max over windows of 5 and 10 steps. We shift by 1 first to exclude the current value.

**FFT features** — 6 dimensions. We run FFT on the past 10-step window, exclude DC, take the top 3 frequency components by magnitude. This captures periodic patterns like rush hour cycles.

**Wavelet features** — 6 dimensions. Daubechies db4, 2-level decomposition. We get approximation and detail coefficients at each level, then take mean and std. This captures multi-scale patterns.

**Statistical features** — 4 dimensions. Skewness tells us asymmetry, kurtosis tells us about outliers, trend is the linear regression slope — is it getting worse? — and volatility is how erratic the changes are.

**Historical statistics** — 5 dimensions. Route-level, stop-level, and hour-level delay means and standard deviations. Computed on training data only, merged into both sets. Provides a "what is normal" baseline for each location.

**Context and temporal** — 9 dimensions. Weekend and rush hour flags, LabelEncoded route/stop/direction, and cyclical sin/cos for hour and day-of-week so that 11 PM and 1 AM are close.

### 3.3 为什么选这些特征

【这是5分的关键】

**Lag features are the most impactful.** Delays have strong autocorrelation — if recent buses were late, the next one likely is too because external factors like traffic jams persist. Evidence: adding lags improved R² from -0.07 to 0.98. That's a 1.05-point jump from one design decision.

**Rolling statistics capture trends** — is it getting better or worse? This was the best individual method in our ablation study.

**FFT captures periodicity** — rush hour has a clear frequency signature. **Wavelet captures multi-scale patterns** — both fast fluctuations and slow trends.

**Statistical features** describe the shape and direction of recent delays. **Historical stats** give location-specific baselines.

We validated all of this with a systematic **ablation study** — 6 configurations tested with the same GRU model, isolating each feature category's contribution.

### 模型选择

We compared 5 architectures. MLP as baseline. GRU as best small model. Then at larger scale: NeuronSpark SNN with dynamic membrane parameters and binary spike encoding, and Transformer with 8-head attention. We deliberately matched SNN and Transformer to similar parameter counts — 1.4M versus 1.6M — for a fair comparison.

---

## Part 4: Preliminary Results and Interpretation（2-3分钟）

【展示文档 Section 4】

### 迭代过程

**V1** — static features only. R² was **negative 0.07**. Worse than predicting the mean. Static context cannot predict delays.

**V2** — added historical averages. Still negative, -0.11. Delays are non-stationary — last year's average doesn't reflect today.

**V3** — added lag features and signal processing. R² jumped to **0.9846**. The breakthrough: delays are a short-term dynamics problem, you need recent history.

**V4** — tried multi-step Seq2Seq. R² only 0.08. Predicting multiple steps ahead is fundamentally harder.

**V5** — NeuronSpark SNN. R² **0.9897** on full dataset, slightly better than GRU. But 500x longer to train.

**V6** — Transformer. R² **0.9942**, RMSE 0.46 minutes. Best model. At similar parameter count to SNN, it's both more accurate and faster.

### 核心结论

【简洁有力】

Three takeaways:

**First**, feature engineering matters most. V1 to V3 — the entire jump from -0.07 to 0.98 — came from better features, not a better model.

**Second**, bus delays are predictable short-term. Recent history is the dominant signal — external conditions persist across consecutive buses.

**Third**, Transformer outperforms SNN at the same scale, but SNN shows promise for energy-efficient edge deployment.

Overall: RMSE reduced from 6.24 to 0.46 minutes — **93% reduction** — through systematic feature engineering and model iteration.

---

## Closing（15秒）

That's our progress. Next steps: additional feature methods like EMD and STFT, cross-validation, and ensemble approaches. Thank you.

---

## Q&A Preparation / 可能被问到的问题

### Q: Why didn't you use 2018-2019 data?
> The MBTA Open Data Portal no longer provides that data. We noted it as a limitation. Ridership data goes back to 2016 so we can still do pre-pandemic comparison for Q1.

### Q: Isn't R² of 0.99 suspiciously high? Could there be data leakage?
> We took extensive precautions: strict temporal split (train < 2025, test >= 2025), features from past values only, scaler fit on training only. The high R² is because lag features are highly predictive for autocorrelated time series — if the last few buses were 10 minutes late, the next one will be similar. Also, V1 with the same split gives R² = -0.07, which wouldn't happen with systematic leakage.

### Q: Why are V1 results negative R²?
> Negative R² means the model predicts worse than just using the mean. Static features alone can't predict individual delays — it's an important finding that delay prediction requires temporal context.

### Q: Why did you try SNN?
> It's our extended research exploration of neuromorphic computing. While SNN didn't beat Transformer, it outperformed GRU, and it would have energy advantages on neuromorphic hardware. We think it's a novel application.

### Q: How does this help Boston?
> Short-term: power real-time passenger information with accurate ETAs. Medium-term: proactive dispatching before delays cascade. Long-term: our equity analysis (41% higher delays on target routes) provides evidence for prioritizing infrastructure investment in underserved communities.

### Q: What's the practical significance of 0.46-minute RMSE?
> Predictions accurate to within about 28 seconds. For a system with 7.5-minute average delay, that's sufficient for operational use.

### Q: Why not use weather or traffic data?
> Good suggestion for Phase 3. Our current features use only MBTA data, which is the most directly available. External sources could improve multi-step prediction where our models struggle.

---

*发言稿结束 / End of Speaking Notes*
