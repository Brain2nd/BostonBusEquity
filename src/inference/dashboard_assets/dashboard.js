const state = {
  summary: null,
  visualizations: [],
  modelMetrics: null,
  notes: null,
  options: null,
  predictionHistory: [],
};

const palette = {
  blue: "#1f5f8b",
  red: "#bc4749",
  green: "#2f6f4e",
  teal: "#2a9d8f",
  amber: "#c47f2c",
  paper: "#fffaf2",
};

function setDefaultTime() {
  const input = document.querySelector('input[name="scheduled_time"]');
  const now = new Date();
  now.setMinutes(now.getMinutes() + 15);
  now.setSeconds(0, 0);
  input.value = new Date(now.getTime() - now.getTimezoneOffset() * 60000)
    .toISOString()
    .slice(0, 16);
}

function toTimeMs(value) {
  if (value === null || value === undefined || value === "") {
    return null;
  }
  const ms = new Date(value).getTime();
  return Number.isFinite(ms) ? ms : null;
}

function plotlyLocalDateTime(value) {
  const ms = toTimeMs(value);
  if (ms === null) {
    return value;
  }
  const date = new Date(ms);
  const pad = (number) => String(number).padStart(2, "0");
  return [
    date.getFullYear(),
    pad(date.getMonth() + 1),
    pad(date.getDate()),
  ].join("-") + "T" + [
    pad(date.getHours()),
    pad(date.getMinutes()),
    pad(date.getSeconds()),
  ].join(":");
}

async function fetchJson(url, options = {}) {
  const response = await fetch(url, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(detail || `${response.status} ${response.statusText}`);
  }
  return response.json();
}

function renderSummary(summary) {
  state.summary = summary;
  document.getElementById("bundle-status").textContent =
    `${summary.model.health.model} is loaded and ready.`;
  const details = document.getElementById("bundle-details");
  details.innerHTML = `
    <dt>Experiment</dt><dd>${summary.model.health.experiment}</dd>
    <dt>Feature version</dt><dd>${summary.model.health.feature_version}</dd>
    <dt>Runtime model</dt><dd>${summary.model.health.model_kind || "n/a"} / ${summary.model.health.feature_profile || "n/a"}</dd>
    <dt>Best model</dt><dd>${summary.model.best_model || "n/a"}</dd>
    <dt>Best MAE</dt><dd>${formatMinutes(summary.model.best_final_mae)}</dd>
  `;

  const grid = document.getElementById("kpi-grid");
  grid.innerHTML = summary.kpis
    .map(
      (item) => `
        <article class="kpi-card">
          <span>${item.label}</span>
          <strong>${item.value}</strong>
          <span>${item.detail}</span>
        </article>
      `,
    )
    .join("");
}

function renderVisualizations(payload) {
  state.visualizations = payload.items || [];
  const categories = ["All", ...new Set(state.visualizations.map((item) => item.category))];
  const filter = document.getElementById("visual-filter");
  filter.innerHTML = categories
    .map(
      (category, index) => `
        <button class="filter-button ${index === 0 ? "active" : ""}" data-category="${category}">
          ${category}
        </button>
      `,
    )
    .join("");
  filter.querySelectorAll("button").forEach((button) => {
    button.addEventListener("click", () => {
      filter.querySelectorAll("button").forEach((node) => node.classList.remove("active"));
      button.classList.add("active");
      renderVisualizationCards(button.dataset.category);
    });
  });
  renderVisualizationCards("All");
}

function refreshStaticFigures() {
  const liveStory = document.getElementById("latest-live-story");
  if (liveStory) {
    liveStory.src = `/figures/mbta_realtime_model_gap_story.png?v=${Date.now()}`;
  }
}

function renderVisualizationCards(category) {
  const grid = document.getElementById("visual-grid");
  const items = state.visualizations.filter(
    (item) => category === "All" || item.category === category,
  );
  grid.innerHTML = items
    .map(
      (item) => `
        <article class="visual-card">
          <img src="${item.url}" alt="${item.title}" loading="lazy" />
          <div>
            <span class="tag">${item.category}</span>
            <h3>${item.title}</h3>
            <p><strong>Claim:</strong> ${item.claim}</p>
            <p>${item.caption}</p>
          </div>
        </article>
      `,
    )
    .join("");
}

function renderModelMetrics(payload) {
  state.modelMetrics = payload;
  const summary = payload.summary || {};
  document.getElementById("model-summary").innerHTML = `
    <h3>${summary.best_model || "Best model pending"}</h3>
    <p><strong>Feature profile:</strong> ${summary.best_feature_profile || "n/a"}</p>
    <p><strong>Best final MAE:</strong> ${formatMinutes(summary.best_final_mae)}</p>
    <p><strong>Comparable prior baseline MAE:</strong> ${formatMinutes(summary.v2_sample_mae)}</p>
    <p><strong>Scoring rubric:</strong> ${payload.scoring?.available ? "Composite deployability score available." : "MAE-only ranking available."}</p>
    <p class="muted">Measured against true delay labels. MBTA official predictions are used only for live comparison and later V5 residual correction.</p>
  `;

  const scoreRows = payload.score_rows || [];
  const rows = (scoreRows.length ? scoreRows : payload.sweep_rows || []).slice(0, 12).reverse();
  const labels = rows.map((row) => `${pretty(row.model_kind || row.model_kind_requested)}<br>${row.feature_profile}`);
  const values = scoreRows.length
    ? rows.map((row) => row.composite_score ?? null)
    : rows.map((row) => row.final_2024_2025_to_2026_MAE ?? row.test_MAE ?? null);
  const hover = scoreRows.length
    ? rows.map((row) => `Score ${Number(row.composite_score).toFixed(1)}<br>MAE ${Number(row.primary_mae).toFixed(2)} min`)
    : rows.map((row) => `MAE ${Number(row.final_2024_2025_to_2026_MAE ?? row.test_MAE).toFixed(2)} min`);
  Plotly.newPlot(
    "model-chart",
    [
      {
        type: "bar",
        orientation: "h",
        y: labels,
        x: values,
        marker: { color: values.map((_, index) => (index === values.length - 1 ? palette.green : palette.blue)) },
        text: hover,
        hovertemplate: "%{y}<br>%{text}<extra></extra>",
      },
    ],
    chartLayout(
      scoreRows.length
        ? "Composite model deployability score, higher is better"
        : "2026 true-label MAE, lower is better",
      scoreRows.length ? "Score" : "MAE (minutes)",
      { horizontal: true },
    ),
    { responsive: true, displayModeBar: false },
  );
}

function renderNotes(payload) {
  const grid = document.getElementById("notes-grid");
  const cards = [
    ["Data processing", payload.data_processing],
    ["Modeling", payload.modeling],
    ["Interpretation", payload.interpretation],
  ];
  grid.innerHTML = cards
    .map(
      ([title, items]) => `
        <article class="note-card">
          <h3>${title}</h3>
          <ul>${items.map((item) => `<li>${item}</li>`).join("")}</ul>
        </article>
      `,
    )
    .join("");
}

function renderSelectionOptions(payload) {
  state.options = payload;
  document.querySelectorAll("form").forEach((form) => {
    const routeSelect = form.querySelector("[data-route-select]");
    if (!routeSelect) {
      return;
    }

    fillSelect(routeSelect, payload.routes || [], payload.defaults?.route_id);
    updateStopOptions(form, routeSelect.value);
    routeSelect.addEventListener("change", () => syncRouteSelection(routeSelect.value));
    const stopSelect = form.querySelector("[data-stop-select]");
    if (stopSelect) {
      stopSelect.addEventListener("change", () => syncStopSelection(stopSelect.value));
    }
  });
  if (payload.defaults?.route_id) {
    syncRouteSelection(payload.defaults.route_id, payload.defaults.stop_id);
  }
}

function syncRouteSelection(routeId, preferredStopId = null) {
  document.querySelectorAll("form").forEach((form) => {
    const routeSelect = form.querySelector("[data-route-select]");
    if (routeSelect && routeSelect.value !== routeId) {
      routeSelect.value = routeId;
    }
    updateStopOptions(form, routeId, preferredStopId);
  });
}

function syncStopSelection(stopId) {
  document.querySelectorAll("form").forEach((form) => {
    const stopSelect = form.querySelector("[data-stop-select]");
    if (!stopSelect) {
      return;
    }
    const hasStop = Array.from(stopSelect.options).some((option) => option.value === stopId);
    if (hasStop) {
      stopSelect.value = stopId;
    }
  });
}

function updateStopOptions(form, routeId, preferredStopId = null) {
  const stopSelect = form.querySelector("[data-stop-select]");
  if (!stopSelect || !state.options) {
    return;
  }
  const routeStops = state.options.route_stop_map?.[routeId] || [];
  const preferredStop = routeStops.some((item) => item.value === stopSelect.value)
    ? stopSelect.value
    : preferredStopId || state.options.defaults?.stop_id;
  fillSelect(stopSelect, routeStops, preferredStop);
}

function fillSelect(select, items, preferredValue) {
  const safeItems = items || [];
  select.innerHTML = "";
  safeItems.forEach((item) => {
    const option = document.createElement("option");
    option.value = item.value;
    option.textContent = item.label || item.value;
    select.appendChild(option);
  });

  if (safeItems.some((item) => item.value === preferredValue)) {
    select.value = preferredValue;
  } else if (safeItems.length) {
    select.value = safeItems[0].value;
  }
  select.disabled = safeItems.length === 0;
}

async function handlePredict(event) {
  event.preventDefault();
  const form = event.currentTarget;
  const { inferencePayload, horizonPayload } = buildPredictionPayload(form);

  const value = document.getElementById("prediction-value");
  const detail = document.getElementById("prediction-detail");
  value.textContent = "Running...";
  detail.textContent = "Calling local realtime inference and forecast horizon.";
  try {
    const result = await fetchJson("/api/predict", {
      method: "POST",
      body: JSON.stringify(inferencePayload),
    });
    const minutes = result.predicted_delay_minutes;
    value.textContent = `${minutes.toFixed(2)} min`;
    detail.textContent =
      `Model: ${result.model}. Defaults used: ${result.used_defaults.length ? result.used_defaults.join(", ") : "none"}.`;

    const horizon = await fetchJson("/api/predict-horizon", {
      method: "POST",
      body: JSON.stringify(horizonPayload),
    });
    const liveForecast = await tryLiveEnrichedForecast(inferencePayload);
    const nearest = liveForecast?.rows?.length
      ? nearestLiveRow(liveForecast.rows, inferencePayload.scheduled_time)
      : null;
    const liveIsRelevant = nearest && isWithinMinutes(
      nearest.scheduled_time,
      inferencePayload.scheduled_time,
      30,
    );
    if (liveForecast?.rows?.length) {
      if (liveForecast.fallback_used) {
        syncRouteSelection(liveForecast.route_id, liveForecast.stop_id);
      }
      if (liveIsRelevant) {
        const liveValue = firstFinite(
          nearest?.live_calibrated_delay_minutes,
          nearest?.official_delay_minutes,
          nearest?.independent_v4_delay_minutes,
        );
        if (Number.isFinite(liveValue)) {
          value.textContent = `${liveValue.toFixed(2)} min`;
        }
        detail.textContent =
          `${fallbackNotice(liveForecast)}Latest live-enhanced forecast. Nearest live trip: ${formatTime(nearest?.scheduled_time)}. ` +
          "The chart is trimmed to the active live window to avoid an uninformative flat tail.";
      } else {
        const windowText = liveWindowText(liveForecast.rows);
        detail.textContent =
          `${fallbackNotice(liveForecast)}MBTA live rows only cover ${windowText}; the red local forecast is clipped to the useful input-time window.`;
      }
      renderLiveEnrichedForecast(liveForecast, inferencePayload.scheduled_time, horizon);
      return;
    }

    detail.textContent =
      `No current MBTA live rows were available for this route-stop. Showing the red local forecast for the input-time window. ` +
      `Model: ${result.model}. Defaults used: ${result.used_defaults.length ? result.used_defaults.join(", ") : "none"}.`;
    renderPredictionHorizon(horizon);
  } catch (error) {
    value.textContent = "Prediction failed";
    detail.textContent = readableError(error);
    renderPredictionHorizon({ rows: [] });
  }
}

function fallbackNotice(payload) {
  if (!payload?.fallback_used) {
    return "";
  }
  return `No live rows for route ${payload.requested_route_id}/stop ${payload.requested_stop_id}; switched to route ${payload.route_id}/stop ${payload.stop_id}. `;
}

function nearestLiveRow(rows, scheduledTime) {
  if (!rows?.length || !scheduledTime) {
    return rows?.[0] || null;
  }
  const target = new Date(scheduledTime).getTime();
  if (!Number.isFinite(target)) {
    return rows[0];
  }
  return rows.reduce((best, row) => {
    const currentDiff = Math.abs(new Date(row.scheduled_time).getTime() - target);
    const bestDiff = Math.abs(new Date(best.scheduled_time).getTime() - target);
    return currentDiff < bestDiff ? row : best;
  }, rows[0]);
}

function isWithinMinutes(value, reference, minutes) {
  const valueMs = new Date(value).getTime();
  const referenceMs = new Date(reference).getTime();
  if (!Number.isFinite(valueMs) || !Number.isFinite(referenceMs)) {
    return false;
  }
  return Math.abs(valueMs - referenceMs) <= minutes * 60 * 1000;
}

function liveWindowText(rows) {
  const times = (rows || [])
    .map((row) => new Date(row.scheduled_time))
    .filter((date) => Number.isFinite(date.getTime()))
    .sort((a, b) => a - b);
  if (!times.length) {
    return "the current live prediction window";
  }
  return `${formatTime(times[0])}-${formatTime(times[times.length - 1])}`;
}

function firstFinite(...values) {
  for (const value of values) {
    const number = Number(value);
    if (Number.isFinite(number)) {
      return number;
    }
  }
  return null;
}

function modelUncertaintyMinutes() {
  const summary = state.modelMetrics?.summary || {};
  const rows = [
    ...(state.modelMetrics?.score_rows || []),
    ...(state.modelMetrics?.sweep_rows || []),
  ];
  const bestModel = String(summary.best_model || "").replaceAll("_", "-");
  const bestProfile = String(summary.best_feature_profile || "");
  const bestRow = rows.find((row) => {
    const label = String(row.model_label || row.model_kind || row.model_kind_requested || "")
      .replaceAll("_", "-");
    return label.includes(bestModel) && String(row.feature_profile || "") === bestProfile;
  }) || rows[0] || {};
  return firstFinite(
    bestRow.final_2024_2025_to_2026_RMSE,
    bestRow.test_RMSE,
    summary.best_final_rmse,
    summary.best_test_rmse,
    6.2,
  );
}

function uncertaintyBandTraces(xValues, yValues, options = {}) {
  const uncertainty = firstFinite(options.uncertainty, modelUncertaintyMinutes());
  const finitePairs = xValues
    .map((x, index) => ({ x, y: Number(yValues[index]) }))
    .filter((point) => point.x && Number.isFinite(point.y));
  if (!finitePairs.length || !Number.isFinite(uncertainty) || uncertainty <= 0) {
    return [];
  }
  const axis = {
    xaxis: options.xaxis || "x",
    yaxis: options.yaxis || "y",
  };
  const color = options.fillcolor || "rgba(188,71,73,0.14)";
  const name = options.name || `Approx. test uncertainty (+/-${uncertainty.toFixed(1)} min RMSE)`;
  return [
    {
      type: "scatter",
      mode: "lines",
      name: `${name} upper`,
      x: finitePairs.map((point) => point.x),
      y: finitePairs.map((point) => point.y + uncertainty),
      line: { width: 0 },
      hoverinfo: "skip",
      showlegend: false,
      ...axis,
    },
    {
      type: "scatter",
      mode: "lines",
      name,
      x: finitePairs.map((point) => point.x),
      y: finitePairs.map((point) => point.y - uncertainty),
      line: { width: 0 },
      fill: "tonexty",
      fillcolor: color,
      hoverinfo: "skip",
      ...axis,
    },
  ];
}

async function tryLiveEnrichedForecast(inferencePayload) {
  try {
    return await fetchJson("/api/live-enriched-forecast", {
      method: "POST",
      body: JSON.stringify({
        route_id: inferencePayload.route_id,
        stop_id: inferencePayload.stop_id,
        direction_id: inferencePayload.direction_id || null,
        prediction_limit: 12,
        vehicle_limit: 100,
      }),
    });
  } catch {
    return null;
  }
}

function buildPredictionPayload(form) {
  const payload = formPayload(form);
  ["scheduled_headway"].forEach((key) => {
    payload[key] = payload[key] === "" ? null : Number(payload[key]);
  });
  payload.direction_id = payload.direction_id || null;
  // Vehicle telemetry fields (current_stop_sequence, vehicle_speed) are not
  // collected from the user — they auto-populate via the live-enriched
  // forecast endpoint when MBTA real-time data is available.

  const horizonHours = Math.min(Number(payload.horizon_hours || 3), 3);
  const intervalMinutes = Number(payload.interval_minutes || 15);
  delete payload.horizon_hours;
  delete payload.interval_minutes;

  return {
    inferencePayload: payload,
    horizonPayload: {
      ...payload,
      horizon_hours: Number.isFinite(horizonHours) ? horizonHours : 3,
      interval_minutes: Number.isFinite(intervalMinutes) ? intervalMinutes : 15,
    },
  };
}

function renderPredictionHorizon(payload) {
  const rows = payload.rows || [];
  const modelLabel = currentModelLabel();
  const xValues = rows.map((row) => plotlyLocalDateTime(row.scheduled_time));
  const localValues = rows.map((row) => Number(row.predicted_delay_minutes));
  const baselineValues = rows.map((row) => Number(row.historical_baseline_delay_minutes));
  const uncertainty = modelUncertaintyMinutes();
  const uncertaintyValues = localValues.flatMap((value) => (
    Number.isFinite(value) ? [value - uncertainty, value + uncertainty] : []
  ));
  const localRange = paddedRange([...localValues, ...uncertaintyValues], 1.2);
  const contextRange = paddedRange([...localValues, ...baselineValues], 3.0);

  Plotly.newPlot(
    "prediction-chart",
    [
      ...uncertaintyBandTraces(xValues, localValues, {
        xaxis: "x",
        yaxis: "y",
        fillcolor: "rgba(47,111,78,0.15)",
      }),
      {
        type: "scatter",
        mode: "lines+markers",
        name: modelLabel,
        x: xValues,
        y: localValues,
        xaxis: "x",
        yaxis: "y",
        line: { color: palette.green, width: 3 },
        marker: { size: 9, color: palette.green, line: { color: "#f7efe4", width: 1.5 } },
        hovertemplate: "Local V4<br>Scheduled %{x}<br>%{y:.2f} min<extra></extra>",
      },
      {
        type: "scatter",
        mode: "lines",
        name: "Historical baseline",
        x: xValues,
        y: baselineValues,
        xaxis: "x2",
        yaxis: "y2",
        line: { color: palette.amber, width: 2.5, dash: "dash" },
        hovertemplate: "Historical baseline<br>Scheduled %{x}<br>%{y:.2f} min<extra></extra>",
      },
      {
        type: "scatter",
        mode: "lines",
        name: `${modelLabel} in context`,
        x: xValues,
        y: localValues,
        xaxis: "x2",
        yaxis: "y2",
        line: { color: palette.green, width: 2, dash: "dot" },
        opacity: 0.65,
        hovertemplate: "Local V4 context<br>Scheduled %{x}<br>%{y:.2f} min<extra></extra>",
        visible: "legendonly",
      },
    ],
    horizonChartLayout(
      rows.length
        ? `Local forecast over ${payload.horizon_hours} hours`
        : "Forecast appears after prediction",
      localRange,
      contextRange,
    ),
    { responsive: true, displayModeBar: false },
  );
}

function renderLiveEnrichedForecast(payload, selectedScheduledTime = null, selectedHorizon = null) {
  const rows = (payload.rows || [])
    .filter((row) => row.scheduled_time)
    .sort((a, b) => new Date(a.scheduled_time) - new Date(b.scheduled_time));
  const xValues = rows.map((row) => plotlyLocalDateTime(row.scheduled_time));
  const rawHorizonRows = (selectedHorizon?.rows || [])
    .filter((row) => row.scheduled_time)
    .sort((a, b) => new Date(a.scheduled_time) - new Date(b.scheduled_time));
  const liveWindow = liveTimeWindow(rows);
  const selectedMs = selectedScheduledTime ? toTimeMs(selectedScheduledTime) : null;
  const selectedNearLive = liveWindow && Number.isFinite(selectedMs)
    ? selectedMs >= liveWindow.start - 30 * 60 * 1000
      && selectedMs <= liveWindow.end + 30 * 60 * 1000
    : false;
  const horizonRows = selectedNearLive
    ? []
    : trimHorizonRows(rawHorizonRows, selectedScheduledTime, 3);
  const combinedForecast = combinedRealtimeForecastRows(rows, horizonRows);
  const combinedBaseline = combinedBaselineRows(rows, horizonRows);
  const chartRange = selectedNearLive && liveWindow
    ? visibleTimeRange(combinedForecast.map((row) => row.scheduled_time), 8, 45)
    : selectedTimeRange(selectedScheduledTime, 3);
  const traces = [
    {
      type: "scatter",
      mode: "lines+markers",
      name: "MBTA official live",
      x: xValues,
      y: rows.map((row) => row.official_delay_minutes),
      line: { color: palette.blue, width: 3 },
      marker: { size: 8 },
      hovertemplate: "MBTA official<br>Scheduled %{x}<br>%{y:.2f} min<extra></extra>",
    },
    ...uncertaintyBandTraces(
      combinedForecast.map((row) => plotlyLocalDateTime(row.scheduled_time)),
      combinedForecast.map((row) => row.predicted_delay_minutes),
      {
        fillcolor: "rgba(188,71,73,0.13)",
      },
    ),
    {
      type: "scatter",
      mode: "lines+markers",
      name: "Local forecast",
      x: combinedForecast.map((row) => plotlyLocalDateTime(row.scheduled_time)),
      y: combinedForecast.map((row) => row.predicted_delay_minutes),
      line: { color: palette.red, width: 3 },
      marker: { size: 8, symbol: "triangle-up" },
      hovertemplate: "Local forecast<br>Scheduled %{x}<br>%{y:.2f} min<extra></extra>",
    },
    {
      type: "scatter",
      mode: "lines",
      name: "Historical baseline",
      x: combinedBaseline.map((row) => plotlyLocalDateTime(row.scheduled_time)),
      y: combinedBaseline.map((row) => row.predicted_delay_minutes),
      line: { color: palette.amber, width: 2.4, dash: "dash" },
      hovertemplate: "Historical baseline<br>Scheduled %{x}<br>%{y:.2f} min<extra></extra>",
    },
  ];

  Plotly.newPlot(
    "prediction-chart",
    traces,
    liveForecastLayout(
      `${payload.fallback_used ? "Auto-selected live route-stop: " : ""}Latest live-enhanced forecast from MBTA V3 trips`,
      "Delay estimate (minutes)",
      payload.message,
      chartRange,
    ),
    { responsive: true, displayModeBar: false },
  );
}

function liveTimeWindow(rows) {
  const times = (rows || [])
    .map((row) => toTimeMs(row.scheduled_time))
    .filter((value) => Number.isFinite(value));
  if (!times.length) {
    return null;
  }
  return { start: Math.min(...times), end: Math.max(...times) };
}

function trimHorizonRows(rows, selectedScheduledTime, maxHours) {
  const selectedMs = toTimeMs(selectedScheduledTime);
  if (!Number.isFinite(selectedMs)) {
    return rows || [];
  }
  const endMs = selectedMs + maxHours * 60 * 60 * 1000;
  return (rows || []).filter((row) => {
    const rowMs = toTimeMs(row.scheduled_time);
    return Number.isFinite(rowMs) && rowMs >= selectedMs && rowMs <= endMs;
  });
}

function paddedTimeRange(startMs, endMs, paddingMinutes) {
  return [
    plotlyLocalDateTime(startMs - paddingMinutes * 60 * 1000),
    plotlyLocalDateTime(endMs + paddingMinutes * 60 * 1000),
  ];
}

function selectedTimeRange(selectedScheduledTime, horizonHours) {
  const selectedMs = toTimeMs(selectedScheduledTime);
  if (!Number.isFinite(selectedMs)) {
    return undefined;
  }
  return [
    plotlyLocalDateTime(selectedMs - 15 * 60 * 1000),
    plotlyLocalDateTime(selectedMs + horizonHours * 60 * 60 * 1000),
  ];
}

function visibleTimeRange(values, paddingMinutes = 8, minimumSpanMinutes = 45) {
  const times = (values || [])
    .map((value) => toTimeMs(value))
    .filter((value) => Number.isFinite(value));
  if (!times.length) {
    return undefined;
  }
  let start = Math.min(...times);
  let end = Math.max(...times);
  const minSpanMs = minimumSpanMinutes * 60 * 1000;
  if (end - start < minSpanMs) {
    const center = (start + end) / 2;
    start = center - minSpanMs / 2;
    end = center + minSpanMs / 2;
  }
  return paddedTimeRange(start, end, paddingMinutes);
}

function combinedRealtimeForecastRows(liveRows, horizonRows) {
  const byTime = new Map();
  (liveRows || []).forEach((row) => {
    const value = firstFinite(row.live_calibrated_delay_minutes, row.official_delay_minutes);
    if (row.scheduled_time && Number.isFinite(value)) {
      byTime.set(row.scheduled_time, {
        scheduled_time: row.scheduled_time,
        predicted_delay_minutes: value,
      });
    }
  });
  (horizonRows || []).forEach((row) => {
    const value = Number(row.predicted_delay_minutes);
    if (row.scheduled_time && Number.isFinite(value)) {
      byTime.set(row.scheduled_time, {
        scheduled_time: row.scheduled_time,
        predicted_delay_minutes: value,
      });
    }
  });
  return Array.from(byTime.values()).sort(
    (a, b) => new Date(a.scheduled_time) - new Date(b.scheduled_time),
  );
}

function combinedBaselineRows(liveRows, horizonRows) {
  const byTime = new Map();
  (liveRows || []).forEach((row) => {
    const value = Number(row.historical_baseline_delay_minutes);
    if (row.scheduled_time && Number.isFinite(value)) {
      byTime.set(row.scheduled_time, {
        scheduled_time: row.scheduled_time,
        predicted_delay_minutes: value,
      });
    }
  });
  (horizonRows || []).forEach((row) => {
    const value = Number(row.historical_baseline_delay_minutes);
    if (row.scheduled_time && Number.isFinite(value)) {
      byTime.set(row.scheduled_time, {
        scheduled_time: row.scheduled_time,
        predicted_delay_minutes: value,
      });
    }
  });
  return Array.from(byTime.values()).sort(
    (a, b) => new Date(a.scheduled_time) - new Date(b.scheduled_time),
  );
}

function liveForecastLayout(title, yTitle, message, xRangeOverride = undefined) {
  const xRange = xRangeOverride;
  return {
    title: { text: title, x: 0, font: { size: 15 } },
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(255,250,242,0.7)",
    margin: { l: 60, r: 24, t: 86, b: 72 },
    xaxis: {
      title: "Scheduled time",
      gridcolor: "rgba(101,115,134,0.22)",
      zeroline: false,
      range: xRange,
      rangeslider: { visible: true, thickness: 0.08 },
    },
    yaxis: {
      title: yTitle,
      gridcolor: "rgba(101,115,134,0.22)",
      zeroline: true,
      zerolinecolor: "rgba(23,33,43,0.55)",
    },
    annotations: [
      {
        x: 0,
        y: 1.14,
        xref: "paper",
        yref: "paper",
        text: "Blue is MBTA's current live window. Red is the latest local forecast. The shaded band is held-out RMSE uncertainty, not added noise.",
        showarrow: false,
        align: "left",
        font: { size: 11, color: "#657386" },
      },
    ],
    legend: { orientation: "h", y: -0.22 },
    font: { family: "Aptos, Segoe UI, sans-serif", color: "#17212b" },
  };
}

function paddedRange(values, minSpan) {
  const finite = values.filter((value) => Number.isFinite(value));
  if (!finite.length) {
    return [-1, 1];
  }
  let low = Math.min(...finite);
  let high = Math.max(...finite);
  const center = (low + high) / 2;
  const span = Math.max(high - low, minSpan);
  low = center - span / 2;
  high = center + span / 2;
  const padding = span * 0.18;
  return [low - padding, high + padding];
}

function horizonChartLayout(title, localRange, contextRange) {
  return {
    title: { text: title, x: 0, font: { size: 15 } },
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(255,250,242,0.7)",
    margin: { l: 66, r: 24, t: 68, b: 58 },
    xaxis: {
      domain: [0, 1],
      anchor: "y",
      showticklabels: false,
      gridcolor: "rgba(101,115,134,0.2)",
      zeroline: false,
    },
    yaxis: {
      domain: [0.48, 1],
      title: "Local V4 zoom (min)",
      range: localRange,
      gridcolor: "rgba(101,115,134,0.22)",
      zeroline: true,
      zerolinecolor: "rgba(23,33,43,0.5)",
    },
    xaxis2: {
      domain: [0, 1],
      anchor: "y2",
      gridcolor: "rgba(101,115,134,0.2)",
      zeroline: false,
    },
    yaxis2: {
      domain: [0, 0.28],
      title: "Baseline context (min)",
      range: contextRange,
      gridcolor: "rgba(101,115,134,0.22)",
      zeroline: true,
      zerolinecolor: "rgba(23,33,43,0.5)",
    },
    annotations: [
      {
        x: 0,
        y: 1.05,
        xref: "paper",
        yref: "paper",
        text: "Local forecast follows the selected scheduled time. Shading shows held-out RMSE uncertainty; the line itself is unchanged.",
        showarrow: false,
        align: "left",
        font: { size: 11, color: "#657386" },
      },
      {
        x: 0,
        y: 0.33,
        xref: "paper",
        yref: "paper",
        text: "Historical baseline shown separately so it does not flatten the V4 line.",
        showarrow: false,
        align: "left",
        font: { size: 11, color: "#657386" },
      },
    ],
    legend: { orientation: "h", y: -0.18 },
    font: { family: "Aptos, Segoe UI, sans-serif", color: "#17212b" },
  };
}

async function handleLiveCompare(event) {
  event.preventDefault();
  const form = event.currentTarget;
  const payload = formPayload(form);
  payload.direction_id = payload.direction_id || null;
  payload.prediction_limit = Number(payload.prediction_limit || 8);
  payload.vehicle_limit = 100;

  document.getElementById("live-mode").textContent = "Loading";
  document.getElementById("live-message").textContent = "Calling MBTA V3 API and local model...";
  try {
    const result = await fetchJson("/api/live-compare", {
      method: "POST",
      body: JSON.stringify(payload),
    });
    if (result.fallback_used) {
      syncRouteSelection(result.route_id, result.stop_id);
    }
    document.getElementById("live-mode").textContent = result.mode;
    document.getElementById("live-message").textContent =
      result.mean_abs_gap_minutes == null
        ? `${fallbackNotice(result)}${result.message}`
        : `${fallbackNotice(result)}${result.message} Mean absolute gap: ${result.mean_abs_gap_minutes.toFixed(2)} minutes.`;
    renderLiveChart(result.rows || []);
  } catch (error) {
    document.getElementById("live-mode").textContent = "Error";
    document.getElementById("live-message").textContent = readableError(error);
    renderLiveChart([]);
  }
}

function renderLiveChart(rows) {
  const sorted = rows
    .filter((row) => row.scheduled_time)
    .sort((a, b) => new Date(a.scheduled_time) - new Date(b.scheduled_time));
  const xValues = sorted.map((row) => plotlyLocalDateTime(row.scheduled_time));
  const localValues = sorted.map((row) => localRealtimeForecast(row));
  Plotly.newPlot(
    "live-chart",
    [
      {
        type: "scatter",
        mode: "lines+markers",
        name: "MBTA official",
        x: xValues,
        y: sorted.map((row) => row.official_delay_minutes),
        line: { color: palette.blue, width: 3 },
        marker: { size: 9 },
      },
      ...uncertaintyBandTraces(xValues, localValues, {
        fillcolor: "rgba(188,71,73,0.12)",
      }),
      {
        type: "scatter",
        mode: "lines",
        name: "Historical baseline",
        x: xValues,
        y: sorted.map((row) => row.historical_baseline_delay_minutes),
        line: { color: palette.amber, width: 2.5, dash: "dash" },
      },
      {
        type: "scatter",
        mode: "lines+markers",
        name: "Local forecast",
        x: xValues,
        y: localValues,
        line: { color: palette.red, width: 3, dash: "dot" },
        marker: { size: 8, symbol: "diamond" },
      },
    ],
    chartLayout("Latest live estimates by scheduled time", "Delay estimate (minutes)", {
      rangeSlider: true,
    }),
    { responsive: true, displayModeBar: false },
  );
}

function localRealtimeForecast(row) {
  const official = Number(row.official_delay_minutes);
  const model = Number(row.model_predicted_delay_minutes);
  const baseline = Number(row.historical_baseline_delay_minutes);
  if (Number.isFinite(official) && Number.isFinite(model) && Number.isFinite(baseline)) {
    return official + (model - baseline);
  }
  return firstFinite(row.official_informed_delay_minutes, row.model_predicted_delay_minutes);
}

function formPayload(form) {
  return Object.fromEntries(new FormData(form).entries());
}

function chartLayout(title, yTitleOrXTitle, options = {}) {
  const horizontal = options.horizontal ?? title.toLowerCase().includes("mae");
  return {
    title: { text: title, x: 0, font: { size: 15 } },
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(255,250,242,0.7)",
    margin: { l: horizontal ? 160 : 54, r: 22, t: 52, b: 52 },
    xaxis: {
      title: horizontal ? yTitleOrXTitle : "",
      gridcolor: "rgba(101,115,134,0.22)",
      zeroline: false,
      rangeslider: options.rangeSlider && !horizontal ? { visible: true, thickness: 0.08 } : undefined,
    },
    yaxis: {
      title: horizontal ? "" : yTitleOrXTitle,
      gridcolor: "rgba(101,115,134,0.22)",
      zeroline: false,
    },
    legend: { orientation: "h", y: -0.18 },
    font: { family: "Aptos, Segoe UI, sans-serif", color: "#17212b" },
  };
}

function pretty(value) {
  return String(value || "").replaceAll("_", " ");
}

function currentModelLabel() {
  const health = state.summary?.model?.health || {};
  if (health.model === "V4Tree") {
    const kind = String(health.model_kind || "tree").replaceAll("_", "-");
    const profile = health.feature_profile ? ` / ${health.feature_profile}` : "";
    return `Latest V4 ${kind}${profile}`;
  }
  if (health.model) {
    return `${health.model}${health.experiment ? ` / ${health.experiment}` : ""}`;
  }
  return "Latest local model";
}

function formatMinutes(value) {
  const number = Number(value);
  return Number.isFinite(number) ? `${number.toFixed(2)} min` : "n/a";
}

function formatTime(value) {
  if (!value) {
    return "n/a";
  }
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return String(value);
  }
  return date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

function readableError(error) {
  try {
    const parsed = JSON.parse(error.message);
    return parsed.detail || error.message;
  } catch {
    return error.message;
  }
}

async function init() {
  setDefaultTime();
  document.getElementById("predict-form").addEventListener("submit", handlePredict);
  document.getElementById("live-form").addEventListener("submit", handleLiveCompare);
  const [summary, visualPayload, metricsPayload, notesPayload, optionsPayload] = await Promise.all([
    fetchJson("/api/project-summary"),
    fetchJson("/api/visualizations"),
    fetchJson("/api/model-metrics"),
    fetchJson("/api/data-model-notes"),
    fetchJson("/api/options"),
  ]);
  renderSummary(summary);
  renderVisualizations(visualPayload);
  renderModelMetrics(metricsPayload);
  renderNotes(notesPayload);
  renderSelectionOptions(optionsPayload);
  refreshStaticFigures();
  renderPredictionHorizon({ rows: [] });
  renderLiveChart([]);
}

init().catch((error) => {
  document.body.insertAdjacentHTML(
    "afterbegin",
    `<div style="padding:14px 18px;background:#bc4749;color:white;font-weight:800">Dashboard failed to initialize: ${readableError(error)}</div>`,
  );
});
