# April Check-In Review Checklist

This checklist is for a teammate, reviewer, or presenter who wants to quickly verify the current April check-in deliverables.

## 1. Core deliverables

Confirm that the repository contains:

1. a runnable realtime dashboard entrypoint
2. an April technical report
3. model comparison figures
4. a live MBTA comparison report
5. a lightweight local validation command

## 2. Files to review first

Review these files first:

1. [README.md](E:\Learn\CLASS\2\506\final\f1\BostonBusEquity\README.md)
2. [APRIL_CHECKIN_TECHNICAL_REPORT.md](E:\Learn\CLASS\2\506\final\f1\BostonBusEquity\docs\APRIL_CHECKIN_TECHNICAL_REPORT.md)
3. [LOCAL_QUICKSTART.md](E:\Learn\CLASS\2\506\final\f1\BostonBusEquity\docs\LOCAL_QUICKSTART.md)
4. [MBTA_REALTIME_OFFICIAL_VS_MODEL.md](E:\Learn\CLASS\2\506\final\f1\BostonBusEquity\reports\MBTA_REALTIME_OFFICIAL_VS_MODEL.md)
5. [MODEL_SCORING_GUIDE.md](E:\Learn\CLASS\2\506\final\f1\BostonBusEquity\reports\MODEL_SCORING_GUIDE.md)

## 3. Questions the repository should answer

By the end of a quick review, the repository should answer:

1. what data sources are used
2. what model is currently deployed in the dashboard
3. how the model is evaluated against true labels
4. how MBTA official predictions are used in comparison
5. how to start the dashboard locally

## 4. Dashboard review path

If running locally, check:

1. the server starts with `.\tools\start_dashboard.ps1`
2. the homepage loads
3. model and figure sections render
4. route-stop selection works
5. live comparison content appears when MBTA data is available

## 5. Presentation review path

For a five-minute presentation review, focus on:

1. `reports/figures/delay_distribution.png`
2. `reports/figures/delays_by_route.png`
3. `reports/figures/v4_model_sweep.png`
4. `reports/figures/v4_model_deployability_scores.png`
5. `reports/figures/mbta_realtime_model_gap_story.png`

## 6. Interpretation checks

Make sure these claims stay accurate:

1. true model accuracy is evaluated against actual delay labels, not MBTA official predictions
2. official-vs-local live plots show disagreement unless matched actual arrivals are available
3. the deployed dashboard model is chosen for deployability, not only MAE
4. the project keeps a distinction between offline research models and online-safe deployment models

## 7. Reviewer conclusion

If the dashboard runs, the figures render, and the documentation answers the questions above, the April check-in deliverables are in a reviewable state.
