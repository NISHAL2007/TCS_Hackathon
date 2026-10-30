// app.js — ExamGuard AI Light-Theme Frontend Engine
let state = {
  telemetry: [],
  thresholds: {},
  selectedIndex: 8,
  currentSampleData: null,
  overviewData: null,
  complaintsData: null,
  evaluationData: null,
  activeView: 'view-overview',
  charts: {},
  whatIfDebounceTimer: null,
  // Fixed canonical category order so bar positions never jump
  canonicalCategories: null
};

// Initialize Application
document.addEventListener('DOMContentLoaded', async () => {
  setupNavigation();
  await loadInitialData();
  await updateActiveSnapshot(state.selectedIndex);
});

// Setup Sidebar Navigation
function setupNavigation() {
  const navItems = document.querySelectorAll('.nav-item');
  navItems.forEach(item => {
    item.addEventListener('click', () => {
      navItems.forEach(n => n.classList.remove('active'));
      item.classList.add('active');
      const targetView = item.getAttribute('data-view');
      switchView(targetView);
    });
  });
}

function switchView(viewId) {
  state.activeView = viewId;
  document.querySelectorAll('.view-section').forEach(v => v.classList.remove('active-view'));
  const target = document.getElementById(viewId);
  if (target) target.classList.add('active-view');

  // Trigger relevant view renderers
  if (viewId === 'view-timeline') {
    renderDeepTimelineChart();
    renderLeadLagChart();
  }
  if (viewId === 'view-rca') renderRcaView();
  if (viewId === 'view-shap') renderShapView();
  if (viewId === 'view-whatif') renderWhatIfView();
  if (viewId === 'view-complaints') renderComplaintsView();
  if (viewId === 'view-reports') renderReportsView();
  if (viewId === 'view-evaluation') renderEvaluationView();
}

function quickJumpToRCA() {
  const rcaNav = document.querySelector('[data-view="view-rca"]');
  if (rcaNav) rcaNav.click();
}

// Fetch Initial Data
async function loadInitialData() {
  try {
    const res = await fetch('/api/telemetry');
    const data = await res.json();
    state.telemetry = data.telemetry;
    state.thresholds = data.thresholds;

    // Populate Snapshot Picker
    const picker = document.getElementById('samplePicker');
    picker.innerHTML = '';
    state.telemetry.forEach((row, i) => {
      const opt = document.createElement('option');
      opt.value = i;
      opt.text = `[${String(i).padStart(2, '0')}] ${row.timestamp.slice(-8)} — ${row.incident_type}`;
      if (i === state.selectedIndex) opt.selected = true;
      picker.appendChild(opt);
    });

    picker.addEventListener('change', (e) => {
      updateActiveSnapshot(parseInt(e.target.value));
    });

    // Load Overview Meta
    const ovRes = await fetch(`/api/overview?index=${state.selectedIndex}`);
    state.overviewData = await ovRes.json();
    renderOverviewMilestones();
    renderLeadLagChart();
  } catch (err) {
    console.error("Failed to load telemetry:", err);
  }
}

// Update Active Snapshot
async function updateActiveSnapshot(index) {
  state.selectedIndex = index;
  try {
    const res = await fetch(`/api/sample/${index}`);
    state.currentSampleData = await res.json();

    const ovRes = await fetch(`/api/overview?index=${index}`);
    state.overviewData = await ovRes.json();

    updateTopbarAndKPIs();
    renderOverviewMilestones();
    renderOverviewChart();

    // Re-render active view
    switchView(state.activeView);
  } catch (err) {
    console.error("Failed to fetch sample details:", err);
  }
}

// Update Topbar & KPI Strip
function updateTopbarAndKPIs() {
  const d = state.currentSampleData;
  if (!d) return;

  const s = d.sample;
  const sev = d.diagnostics.severity_assessment;

  // Topbar
  const topSeverity = document.getElementById('topbarSeverity');
  topSeverity.textContent = `${sev.severity_level.toUpperCase()} SEVERITY`;
  topSeverity.className = `badge badge-${sev.severity_level.toLowerCase()}`;

  document.getElementById('topbarTitle').textContent = `${s.incident_type} Focus`;
  document.getElementById('topbarTimestamp').textContent = s.timestamp;

  // Pulse Dot
  const liveDot = document.getElementById('liveDot');
  const statusText = document.getElementById('systemStatusText');
  if (sev.severity_level === 'Informational' || sev.severity_level === 'Low') {
    liveDot.className = 'live-dot';
    statusText.textContent = 'PORTAL NORMAL';
  } else {
    liveDot.className = 'live-dot active-incident';
    statusText.textContent = 'INCIDENT ACTIVE';
  }

  // KPIs
  document.getElementById('kpiResponse').textContent = `${s.response_time_ms.toLocaleString()} ms`;
  const rtDelta = Math.round(((s.response_time_ms - state.thresholds.response_time_ms) / state.thresholds.response_time_ms) * 100);
  const kpiRtDelta = document.getElementById('kpiResponseDelta');
  kpiRtDelta.textContent = rtDelta > 0 ? `+${rtDelta}% above baseline` : `Within baseline (≤ ${state.thresholds.response_time_ms}ms)`;
  kpiRtDelta.style.color = rtDelta > 0 ? 'var(--rose-crit)' : 'var(--teal-healthy)';

  document.getElementById('kpiDb').textContent = `${s.db_query_time_ms.toLocaleString()} ms`;
  document.getElementById('kpiCpu').textContent = `${s.cpu_usage_pct.toFixed(1)}%`;
  document.getElementById('kpiErrors').textContent = s.error_count;
  document.getElementById('kpiUsers').textContent = s.active_users.toLocaleString();

  // Overview Diagnostic Card
  const topHyp = d.diagnostics.hypotheses[0];
  document.getElementById('ovTopScore').textContent = `${topHyp.evidence_score_pct}% Match`;
  document.getElementById('ovTopHypothesis').textContent = topHyp.hypothesis;

  const pillsContainer = document.getElementById('ovEvidencePills');
  pillsContainer.innerHTML = '';
  topHyp.supporting_evidence.slice(0, 2).forEach(ev => {
    const p = document.createElement('div');
    p.className = 'cell-pill cell-supports';
    p.style.display = 'block';
    p.style.marginBottom = '4px';
    p.textContent = `✓ ${ev}`;
    pillsContainer.appendChild(p);
  });

  const nextChk = d.diagnostics.next_best_diagnostic_check;
  document.getElementById('ovNextCheckTitle').textContent = nextChk.check_title;
  document.getElementById('ovNextCheckTool').textContent = `Tool: ${nextChk.recommended_tool}`;
}

// 1. Overview Chart
function renderOverviewChart() {
  const ctx = document.getElementById('overviewChart');
  if (!ctx) return;

  const labels = state.telemetry.map(t => t.timestamp.slice(11, 16));
  const rtData = state.telemetry.map(t => t.response_time_ms);
  const dbData = state.telemetry.map(t => t.db_query_time_ms);

  const selectedIdx = state.selectedIndex;

  // Point radii and colors to distinctly highlight the selected milestone
  const rtPointRadii = rtData.map((_, i) => (i === selectedIdx ? 7 : 3));
  const rtPointHoverRadii = rtData.map((_, i) => (i === selectedIdx ? 9 : 5));
  const rtPointBg = rtData.map((_, i) => (i === selectedIdx ? '#FFFFFF' : '#2563EB'));
  const rtPointBorderWidth = rtData.map((_, i) => (i === selectedIdx ? 3 : 1));
  const rtPointBorderColor = rtData.map((_, i) => (i === selectedIdx ? '#1D4ED8' : '#2563EB'));

  const dbPointRadii = dbData.map((_, i) => (i === selectedIdx ? 7 : 2));
  const dbPointHoverRadii = dbData.map((_, i) => (i === selectedIdx ? 9 : 4));
  const dbPointBg = dbData.map((_, i) => (i === selectedIdx ? '#FFFFFF' : '#EC4899'));
  const dbPointBorderWidth = dbData.map((_, i) => (i === selectedIdx ? 3 : 1));
  const dbPointBorderColor = dbData.map((_, i) => (i === selectedIdx ? '#BE185D' : '#EC4899'));

  // Custom plugin to draw an indicator line and label at the selected timestamp
  const selectedFocusPlugin = {
    id: 'selectedFocusPlugin',
    afterDatasetsDraw(chart) {
      if (selectedIdx < 0 || selectedIdx >= labels.length) return;
      const meta = chart.getDatasetMeta(0);
      if (!meta || !meta.data[selectedIdx]) return;

      const element = meta.data[selectedIdx];
      const x = element.x;
      const topY = chart.chartArea.top;
      const bottomY = chart.chartArea.bottom;
      const chartCtx = chart.ctx;

      chartCtx.save();

      // Draw vertical guide line
      chartCtx.beginPath();
      chartCtx.moveTo(x, topY);
      chartCtx.lineTo(x, bottomY);
      chartCtx.lineWidth = 1.5;
      chartCtx.setLineDash([4, 4]);
      chartCtx.strokeStyle = 'rgba(37, 99, 235, 0.7)';
      chartCtx.stroke();

      // Draw pill badge at top of guide line
      const text = `Focus: ${labels[selectedIdx]}`;
      chartCtx.font = 'bold 10px Inter, sans-serif';
      const textWidth = chartCtx.measureText(text).width;
      const badgeW = textWidth + 14;
      const badgeH = 18;
      const badgeX = Math.max(chart.chartArea.left + 2, Math.min(x - badgeW / 2, chart.chartArea.right - badgeW - 2));
      const badgeY = topY + 4;

      chartCtx.setLineDash([]);
      chartCtx.fillStyle = '#2563EB';
      chartCtx.beginPath();
      chartCtx.roundRect(badgeX, badgeY, badgeW, badgeH, 4);
      chartCtx.fill();

      chartCtx.fillStyle = '#FFFFFF';
      chartCtx.textAlign = 'center';
      chartCtx.textBaseline = 'middle';
      chartCtx.fillText(text, badgeX + badgeW / 2, badgeY + badgeH / 2);

      chartCtx.restore();
    }
  };

  if (state.charts.overview) state.charts.overview.destroy();

  state.charts.overview = new Chart(ctx, {
    type: 'line',
    data: {
      labels: labels,
      datasets: [
        {
          label: 'Response Time (ms)',
          data: rtData,
          borderColor: '#2563EB',
          backgroundColor: 'rgba(37, 99, 235, 0.08)',
          fill: true,
          tension: 0.25,
          borderWidth: 2.5,
          pointRadius: rtPointRadii,
          pointHoverRadius: rtPointHoverRadii,
          pointBackgroundColor: rtPointBg,
          pointBorderColor: rtPointBorderColor,
          pointBorderWidth: rtPointBorderWidth
        },
        {
          label: 'DB Query Latency (ms)',
          data: dbData,
          borderColor: '#EC4899',
          borderDash: [4, 4],
          tension: 0.2,
          borderWidth: 2,
          pointRadius: dbPointRadii,
          pointHoverRadius: dbPointHoverRadii,
          pointBackgroundColor: dbPointBg,
          pointBorderColor: dbPointBorderColor,
          pointBorderWidth: dbPointBorderWidth
        }
      ]
    },
    plugins: [selectedFocusPlugin],
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { position: 'top', labels: { boxWidth: 12, font: { family: 'Inter', size: 11 } } },
        tooltip: {
          backgroundColor: '#0F172A',
          titleFont: { family: 'JetBrains Mono', size: 12 },
          bodyFont: { family: 'Inter', size: 12 }
        }
      },
      scales: {
        x: { grid: { color: '#F1F5F9' }, ticks: { font: { size: 10 } } },
        y: { grid: { color: '#F1F5F9' }, ticks: { font: { size: 10 } } }
      }
    }
  });
}

// Overview Milestones Cards
function renderOverviewMilestones() {
  const container = document.getElementById('milestoneCards');
  if (!container || !state.telemetry) return;

  container.innerHTML = '';
  const notable = [0, 8, 14, 19, 24, 30];

  notable.forEach(idx => {
    if (idx < state.telemetry.length) {
      const row = state.telemetry[idx];
      const card = document.createElement('div');
      card.style.background = idx === state.selectedIndex ? 'var(--primary-light)' : 'var(--bg-subtle)';
      card.style.border = idx === state.selectedIndex ? '1px solid var(--primary)' : '1px solid var(--border-light)';
      card.style.borderRadius = 'var(--radius-sm)';
      card.style.padding = '10px';
      card.style.cursor = 'pointer';
      card.style.textAlign = 'center';

      card.innerHTML = `
        <div class="font-mono" style="font-size: 0.72rem; color: var(--primary); font-weight:600;">${row.timestamp.slice(11, 16)}</div>
        <div style="font-size: 0.82rem; font-weight: 700; color: var(--text-main); margin: 3px 0;">${row.incident_type.split('/')[0]}</div>
        <div style="font-size: 0.72rem; color: ${row.response_time_ms > 1000 ? 'var(--rose-crit)' : 'var(--teal-healthy)'};">${row.response_time_ms}ms | ${row.error_count} err</div>
      `;

      card.onclick = () => {
        const picker = document.getElementById('samplePicker');
        picker.value = idx;
        updateActiveSnapshot(idx);
      };
      container.appendChild(card);
    }
  });
}

// 2. Incident Timeline Deep Dive
function renderDeepTimelineChart() {
  const ctx = document.getElementById('deepTimelineChart');
  if (!ctx || !state.telemetry) return;

  const labels = state.telemetry.map(t => t.timestamp.slice(11, 16));
  const rtData = state.telemetry.map(t => t.response_time_ms);
  const dbData = state.telemetry.map(t => t.db_query_time_ms);
  const cpuData = state.telemetry.map(t => t.cpu_usage_pct);
  const errData = state.telemetry.map(t => t.error_count);

  const selectedIdx = state.selectedIndex;

  const deepFocusPlugin = {
    id: 'deepFocusPlugin',
    afterDatasetsDraw(chart) {
      if (selectedIdx < 0 || selectedIdx >= labels.length) return;
      const meta = chart.getDatasetMeta(0);
      if (!meta || !meta.data[selectedIdx]) return;

      const element = meta.data[selectedIdx];
      const x = element.x;
      const topY = chart.chartArea.top;
      const bottomY = chart.chartArea.bottom;
      const chartCtx = chart.ctx;

      chartCtx.save();
      chartCtx.beginPath();
      chartCtx.moveTo(x, topY);
      chartCtx.lineTo(x, bottomY);
      chartCtx.lineWidth = 1.5;
      chartCtx.setLineDash([4, 4]);
      chartCtx.strokeStyle = 'rgba(37, 99, 235, 0.7)';
      chartCtx.stroke();

      const text = `Focus: ${labels[selectedIdx]}`;
      chartCtx.font = 'bold 10px Inter, sans-serif';
      const textWidth = chartCtx.measureText(text).width;
      const badgeW = textWidth + 14;
      const badgeH = 18;
      const badgeX = Math.max(chart.chartArea.left + 2, Math.min(x - badgeW / 2, chart.chartArea.right - badgeW - 2));
      const badgeY = topY + 4;

      chartCtx.setLineDash([]);
      chartCtx.fillStyle = '#2563EB';
      chartCtx.beginPath();
      chartCtx.roundRect(badgeX, badgeY, badgeW, badgeH, 4);
      chartCtx.fill();

      chartCtx.fillStyle = '#FFFFFF';
      chartCtx.textAlign = 'center';
      chartCtx.textBaseline = 'middle';
      chartCtx.fillText(text, badgeX + badgeW / 2, badgeY + badgeH / 2);

      chartCtx.restore();
    }
  };

  if (state.charts.deepTimeline) state.charts.deepTimeline.destroy();

  state.charts.deepTimeline = new Chart(ctx, {
    type: 'line',
    data: {
      labels: labels,
      datasets: [
        {
          label: 'Response Time (ms)',
          data: rtData,
          borderColor: '#2563EB',
          borderWidth: 2.5,
          tension: 0.2,
          pointRadius: rtData.map((_, i) => (i === selectedIdx ? 7 : 3)),
          pointBackgroundColor: rtData.map((_, i) => (i === selectedIdx ? '#FFFFFF' : '#2563EB')),
          pointBorderColor: '#2563EB',
          pointBorderWidth: rtData.map((_, i) => (i === selectedIdx ? 3 : 1))
        },
        {
          label: 'DB Query Latency (ms)',
          data: dbData,
          borderColor: '#EC4899',
          borderWidth: 2,
          tension: 0.2,
          pointRadius: dbData.map((_, i) => (i === selectedIdx ? 7 : 2)),
          pointBackgroundColor: dbData.map((_, i) => (i === selectedIdx ? '#FFFFFF' : '#EC4899')),
          pointBorderColor: '#EC4899',
          pointBorderWidth: dbData.map((_, i) => (i === selectedIdx ? 3 : 1))
        },
        {
          label: 'CPU Usage (%)',
          data: cpuData,
          borderColor: '#D97706',
          borderWidth: 2,
          tension: 0.2,
          pointRadius: cpuData.map((_, i) => (i === selectedIdx ? 7 : 2)),
          pointBackgroundColor: cpuData.map((_, i) => (i === selectedIdx ? '#FFFFFF' : '#D97706')),
          pointBorderColor: '#D97706',
          pointBorderWidth: cpuData.map((_, i) => (i === selectedIdx ? 3 : 1))
        },
        {
          label: 'Errors (Count)',
          data: errData,
          borderColor: '#E11D48',
          borderWidth: 2,
          tension: 0.2,
          pointRadius: errData.map((_, i) => (i === selectedIdx ? 7 : 2)),
          pointBackgroundColor: errData.map((_, i) => (i === selectedIdx ? '#FFFFFF' : '#E11D48')),
          pointBorderColor: '#E11D48',
          pointBorderWidth: errData.map((_, i) => (i === selectedIdx ? 3 : 1))
        }
      ]
    },
    plugins: [deepFocusPlugin],
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { position: 'top', labels: { boxWidth: 12, font: { family: 'Inter', size: 11 } } }
      },
      scales: {
        x: { grid: { color: '#F1F5F9' } },
        y: { grid: { color: '#F1F5F9' } }
      }
    }
  });

  // Patient Zero box
  const pz = state.overviewData ? state.overviewData.first_anomaly : null;
  const pzBox = document.getElementById('patientZeroBox');
  if (pzBox && pz) {
    const isNominal = pz.first_breached_metric === 'All Metrics Nominal';
    const tagColor = isNominal ? 'var(--teal-healthy)' : 'var(--rose-crit)';
    const headerTitle = pz.window_label ? pz.window_label : 'Incident Patient Zero';
    pzBox.innerHTML = `
      <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
        <span style="font-size: 0.82rem; font-weight: 700; color: var(--text-muted); text-transform: uppercase;">${headerTitle}</span>
        <span class="badge" style="background: ${isNominal ? 'var(--teal-light)' : 'var(--rose-light)'}; color: ${tagColor}; font-size:0.72rem;">${isNominal ? 'NOMINAL' : 'INITIAL TRIGGER'}</span>
      </div>
      <div style="font-size: 1.15rem; font-weight: 800; color: ${tagColor}; margin-bottom: 6px;">${pz.first_breached_metric} ${pz.first_breached_value > 0 ? '(' + pz.first_breached_value + ')' : ''}</div>
      <div style="font-size: 0.88rem; color: var(--text-main); line-height: 1.5;">${pz.narrative}</div>
      <div style="margin-top: 14px; font-size: 0.80rem; color: var(--text-muted); background: var(--bg-subtle); padding: 8px 12px; border-radius: var(--radius-sm);">
        <b>Initial User Ticket:</b> <i>"${pz.complaint_at_first_anomaly}"</i>
      </div>
    `;
  }
}

function renderLeadLagChart() {
  const ctx = document.getElementById('leadLagChart');
  if (!ctx || !state.overviewData) return;

  const ll = state.overviewData.lead_lag;
  const labels = ll.map(item => item.metric);
  const data = ll.map(item => item.lead1_correlation);

  if (state.charts.leadLag) state.charts.leadLag.destroy();

  state.charts.leadLag = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: labels,
      datasets: [{
        label: 'Lead-1 Cross Correlation against Response Time',
        data: data,
        backgroundColor: '#4F46E5',
        borderRadius: 4
      }]
    },
    options: {
      indexAxis: 'y',
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: { min: -1, max: 1, grid: { color: '#F1F5F9' } },
        y: { grid: { display: false } }
      }
    }
  });
}

// 3. Root-Cause Analysis View
function renderRcaView() {
  const d = state.currentSampleData;
  if (!d) return;

  // Ranked Hypotheses Chart
  const ctx = document.getElementById('hypothesesChart');
  if (ctx) {
    const hyps = d.diagnostics.hypotheses;
    const labels = hyps.map(h => h.hypothesis);
    const scores = hyps.map(h => h.evidence_score_pct);

    if (state.charts.hypotheses) state.charts.hypotheses.destroy();

    state.charts.hypotheses = new Chart(ctx, {
      type: 'bar',
      data: {
        labels: labels,
        datasets: [{
          label: 'Evidence Score (%)',
          data: scores,
          backgroundColor: scores.map(s => s >= 70 ? '#0D9488' : (s >= 40 ? '#2563EB' : '#94A3B8')),
          borderRadius: 4
        }]
      },
      options: {
        indexAxis: 'y',
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: {
          x: { min: 0, max: 100, grid: { color: '#F1F5F9' } },
          y: { grid: { display: false } }
        }
      }
    });
  }

  // Evidence Matrix Table
  const tbody = document.getElementById('matrixTableBody');
  if (tbody && d.evidence_matrix) {
    tbody.innerHTML = '';
    d.evidence_matrix.forEach(row => {
      const tr = document.createElement('tr');
      let cellsHtml = `<td style="font-weight: 700; color: var(--text-main);">${row.hypothesis}</td>`;
      const signals = ["Response Time", "CPU", "Memory", "DB Latency", "Network", "Errors", "Complaints"];

      signals.forEach(sig => {
        const cell = row.signals[sig] || { status: 'NEUTRAL', reason: '' };
        const cls = cell.status === 'SUPPORTS' ? 'cell-supports' : (cell.status === 'CONTRADICTS' ? 'cell-contradicts' : 'cell-neutral');
        const icon = cell.status === 'SUPPORTS' ? '✓' : (cell.status === 'CONTRADICTS' ? '✕' : '—');
        cellsHtml += `<td><span class="cell-pill ${cls}" title="${cell.reason}">${icon} ${cell.status}</span></td>`;
      });

      tr.innerHTML = cellsHtml;
      tbody.appendChild(tr);
    });
  }
}

// 4. SHAP View
function renderShapView() {
  const d = state.currentSampleData;
  if (!d) return;

  // Local Waterfall
  const ctxLocal = document.getElementById('shapLocalChart');
  if (ctxLocal && d.shap) {
    const contribs = [...d.shap.contributions].reverse();
    const labels = contribs.map(c => c.feature);
    const values = contribs.map(c => c.attribution);

    if (state.charts.shapLocal) state.charts.shapLocal.destroy();

    state.charts.shapLocal = new Chart(ctxLocal, {
      type: 'bar',
      data: {
        labels: labels,
        datasets: [{
          label: 'SHAP Contribution',
          data: values,
          backgroundColor: values.map(v => v >= 0 ? '#2563EB' : '#94A3B8'),
          borderRadius: 4
        }]
      },
      options: {
        indexAxis: 'y',
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: {
          x: { grid: { color: '#F1F5F9' } },
          y: { grid: { display: false } }
        }
      }
    });

    document.getElementById('shapLocalSubtitle').textContent = `Target: ${d.shap.predicted_class} (Method: ${d.shap.method})`;
  }

  // Global Importance
  const ctxGlobal = document.getElementById('shapGlobalChart');
  if (ctxGlobal && state.overviewData && state.overviewData.global_importance) {
    const ranking = [...state.overviewData.global_importance.ranking].reverse();
    const labels = ranking.map(r => r.feature);
    const values = ranking.map(r => r.importance);

    if (state.charts.shapGlobal) state.charts.shapGlobal.destroy();

    state.charts.shapGlobal = new Chart(ctxGlobal, {
      type: 'bar',
      data: {
        labels: labels,
        datasets: [{
          label: 'Gini Importance',
          data: values,
          backgroundColor: '#4F46E5',
          borderRadius: 4
        }]
      },
      options: {
        indexAxis: 'y',
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: {
          x: { grid: { color: '#F1F5F9' } },
          y: { grid: { display: false } }
        }
      }
    });
  }

  // Narrative
  const narrBox = document.getElementById('shapNarrativeBox');
  if (narrBox && d.shap) {
    narrBox.innerHTML = d.shap.plain_language_summary.replace(/\n/g, '<br/>');
  }
}

// 5. What-If Simulator View
function renderWhatIfView() {
  const d = state.currentSampleData;
  if (!d) return;

  const s = d.sample;

  // Destroy old chart when switching snapshots so categories refresh
  if (state.charts.whatIfCompare) {
    state.charts.whatIfCompare.destroy();
    state.charts.whatIfCompare = null;
  }
  state.canonicalCategories = null; // Reset so new snapshot rebuilds category order

  document.getElementById('sliderRt').value = s.response_time_ms;
  document.getElementById('sliderDb').value = s.db_query_time_ms;
  document.getElementById('sliderCpu').value = s.cpu_usage_pct;
  document.getElementById('sliderMem').value = s.memory_usage_pct;
  document.getElementById('sliderErr').value = s.error_count;
  document.getElementById('sliderNet').value = s.network_latency_ms;

  document.getElementById('valSliderRt').textContent = `${s.response_time_ms} ms`;
  document.getElementById('valSliderDb').textContent = `${s.db_query_time_ms} ms`;
  document.getElementById('valSliderCpu').textContent = `${s.cpu_usage_pct.toFixed(1)}%`;
  document.getElementById('valSliderMem').textContent = `${s.memory_usage_pct.toFixed(1)}%`;
  document.getElementById('valSliderErr').textContent = s.error_count;
  document.getElementById('valSliderNet').textContent = `${s.network_latency_ms} ms`;

  // Wire all sliders to debounced updater (remove any previous listeners first)
  ['sliderRt','sliderDb','sliderCpu','sliderMem','sliderErr','sliderNet'].forEach(id => {
    const el = document.getElementById(id);
    el.oninput = updateWhatIfDebounced;
  });

  updateWhatIf();
}

function resetWhatIfSliders() {
  renderWhatIfView();
}

// Debounced wrapper — waits 280ms after slider stops before calling API
function updateWhatIfDebounced() {
  clearTimeout(state.whatIfDebounceTimer);
  state.whatIfDebounceTimer = setTimeout(() => updateWhatIf(), 280);
}

async function updateWhatIf() {
  const rt = parseFloat(document.getElementById('sliderRt').value);
  const db = parseFloat(document.getElementById('sliderDb').value);
  const cpu = parseFloat(document.getElementById('sliderCpu').value);
  const mem = parseFloat(document.getElementById('sliderMem').value);
  const err = parseInt(document.getElementById('sliderErr').value);
  const net = parseFloat(document.getElementById('sliderNet').value);

  document.getElementById('valSliderRt').textContent = `${rt} ms`;
  document.getElementById('valSliderDb').textContent = `${db} ms`;
  document.getElementById('valSliderCpu').textContent = `${cpu.toFixed(1)}%`;
  document.getElementById('valSliderMem').textContent = `${mem.toFixed(1)}%`;
  document.getElementById('valSliderErr').textContent = err;
  document.getElementById('valSliderNet').textContent = `${net} ms`;

  try {
    const res = await fetch('/api/simulate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        sample_index: state.selectedIndex,
        overrides: {
          response_time_ms: rt,
          db_query_time_ms: db,
          cpu_usage_pct: cpu,
          memory_usage_pct: mem,
          error_count: err,
          network_latency_ms: net
        }
      })
    });
    const sim = await res.json();

    document.getElementById('whatifOrigClass').textContent = sim.original_prediction.incident_type;
    document.getElementById('whatifOrigConf').textContent = `${(sim.original_prediction.confidence * 100).toFixed(1)}% Confidence`;

    document.getElementById('whatifSimClass').textContent = sim.simulated_prediction.incident_type;
    document.getElementById('whatifSimConf').textContent = `${(sim.simulated_prediction.confidence * 100).toFixed(1)}% Confidence`;

    const badge = document.getElementById('whatifShiftBadge');
    if (sim.prediction_shifted) {
      badge.textContent = "Prediction Shifted!";
      badge.className = "badge badge-critical";
    } else {
      badge.textContent = "Prediction Maintained";
      badge.className = "badge badge-normal";
    }

    // Build probability maps
    const origProbs = Object.fromEntries(sim.original_prediction.sorted_probabilities);
    const simProbs  = Object.fromEntries(sim.simulated_prediction.sorted_probabilities);

    // Fix canonical category order once (alphabetical) so bars never jump
    if (!state.canonicalCategories) {
      state.canonicalCategories = Object.keys(origProbs).sort();
    }
    const cats = state.canonicalCategories;

    const origData = cats.map(c => origProbs[c] || 0);
    const simData  = cats.map(c => simProbs[c]  || 0);

    // Comparison Chart — create once, update in-place for smooth animation
    const ctx = document.getElementById('whatIfCompareChart');
    if (ctx) {
      if (!state.charts.whatIfCompare) {
        // First render: create the chart with animation config
        state.charts.whatIfCompare = new Chart(ctx, {
          type: 'bar',
          data: {
            labels: cats,
            datasets: [
              {
                label: 'Observed State',
                data: origData,
                backgroundColor: cats.map(() => 'rgba(148,163,184,0.75)'),
                borderColor: cats.map(() => '#94A3B8'),
                borderWidth: 1.5,
                borderRadius: 6,
                borderSkipped: false
              },
              {
                label: 'Simulated State',
                data: simData,
                backgroundColor: cats.map(c => simProbs[c] > 0.4 ? 'rgba(13,148,136,0.85)' : 'rgba(13,148,136,0.40)'),
                borderColor: cats.map(c => simProbs[c] > 0.4 ? '#0D9488' : 'rgba(13,148,136,0.6)'),
                borderWidth: 1.5,
                borderRadius: 6,
                borderSkipped: false
              }
            ]
          },
          options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: {
              duration: 500,
              easing: 'easeInOutCubic'
            },
            transitions: {
              active: { animation: { duration: 400 } }
            },
            plugins: {
              legend: { position: 'top', labels: { font: { family: 'Inter', size: 11 }, boxWidth: 12 } },
              tooltip: {
                callbacks: {
                  label: (ctx) => ` ${ctx.dataset.label}: ${(ctx.raw * 100).toFixed(1)}%`
                }
              }
            },
            scales: {
              x: {
                ticks: { font: { size: 10, family: 'Inter' }, maxRotation: 15 },
                grid: { color: '#F1F5F9' }
              },
              y: {
                min: 0,
                max: 1,
                grid: { color: '#F1F5F9' },
                ticks: {
                  font: { size: 10 },
                  callback: v => `${(v * 100).toFixed(0)}%`
                }
              }
            }
          }
        });
      } else {
        // Subsequent renders: update data IN-PLACE → triggers smooth animation
        const ch = state.charts.whatIfCompare;
        ch.data.datasets[0].data = origData;
        ch.data.datasets[1].data = simData;

        // Recolor simulated bars based on new probability strength
        ch.data.datasets[1].backgroundColor = cats.map(c =>
          (simProbs[c] || 0) > 0.4 ? 'rgba(13,148,136,0.85)' : 'rgba(13,148,136,0.40)'
        );
        ch.data.datasets[1].borderColor = cats.map(c =>
          (simProbs[c] || 0) > 0.4 ? '#0D9488' : 'rgba(13,148,136,0.6)'
        );

        ch.update('active'); // 'active' mode plays the smooth transition animation
      }
    }

    // Coupling warnings
    const alertBox = document.getElementById('whatifCouplingAlerts');
    alertBox.innerHTML = '';
    if (sim.coupling_warnings && sim.coupling_warnings.length > 0) {
      sim.coupling_warnings.forEach(w => {
        const d = document.createElement('div');
        d.className = 'badge badge-high';
        d.style.display = 'block';
        d.style.marginBottom = '6px';
        d.textContent = `⚠️ Coupling Warning: ${w}`;
        alertBox.appendChild(d);
      });
    }
  } catch (err) {
    console.error("Simulation error:", err);
  }
}

// 6. Complaint Analytics View
async function renderComplaintsView() {
  try {
    const res = await fetch('/api/complaints');
    state.complaintsData = await res.json();
    const data = state.complaintsData;

    // Donut Chart
    const ctx = document.getElementById('complaintPieChart');
    if (ctx && data.stats) {
      const dist = data.stats.category_distribution;
      const labels = Object.keys(dist);
      const values = Object.values(dist);

      if (state.charts.complaintPie) state.charts.complaintPie.destroy();

      state.charts.complaintPie = new Chart(ctx, {
        type: 'doughnut',
        data: {
          labels: labels,
          datasets: [{
            data: values,
            backgroundColor: ['#2563EB', '#EC4899', '#D97706', '#0D9488', '#4F46E5', '#94A3B8']
          }]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: { legend: { position: 'right' } }
        }
      });
    }

    // List of tickets
    const list = document.getElementById('complaintsList');
    if (list && data.tickets) {
      list.innerHTML = '';
      data.tickets.slice(0, 15).forEach(t => {
        const div = document.createElement('div');
        div.style.padding = '10px';
        div.style.borderBottom = '1px solid var(--border-light)';
        div.innerHTML = `
          <div style="display:flex; justify-content:space-between; font-size:0.75rem; color:var(--text-muted);">
            <span class="font-mono">${t.timestamp.slice(11, 16)}</span>
            <span class="badge badge-ai" style="font-size:0.70rem;">${t.category}</span>
          </div>
          <div style="font-size:0.88rem; color:var(--text-main); margin-top:4px; font-weight:500;">"${t.complaint_text}"</div>
        `;
        list.appendChild(div);
      });
    }
  } catch (err) {
    console.error("Failed to load complaints:", err);
  }
}

// 7. Incident Reports View
function renderReportsView() {
  const d = state.currentSampleData;
  if (!d) return;

  const s = d.sample;
  const diag = d.diagnostics;
  const sev = diag.severity_assessment;
  const topHyp = diag.hypotheses[0];

  const html = `
    <div style="display:flex; justify-content:space-between; align-items:center; border-bottom:1px solid var(--border-light); padding-bottom:12px; margin-bottom:16px;">
      <div>
        <h2 style="font-size:1.3rem; font-weight:800; color:var(--text-main);">INCIDENT POST-MORTEM AUDIT REPORT</h2>
        <div style="font-size:0.80rem; color:var(--text-muted);">Incident ID: <code>INC-EXAM-${s.timestamp.slice(-8).replace(/:/g, '')}</code> | Window: ${s.timestamp}</div>
      </div>
      <span class="badge badge-${sev.severity_level.toLowerCase()}" style="font-size:0.85rem; padding:6px 14px;">${sev.severity_level} SEVERITY</span>
    </div>

    <div style="display:grid; grid-template-columns:repeat(4, 1fr); gap:12px; margin-bottom:20px;">
      <div style="background:var(--bg-surface); padding:12px; border-radius:6px; border:1px solid var(--border-light);">
        <div style="font-size:0.72rem; color:var(--text-muted); font-weight:600;">RESPONSE TIME</div>
        <div style="font-size:1.2rem; font-weight:700;">${s.response_time_ms} ms</div>
      </div>
      <div style="background:var(--bg-surface); padding:12px; border-radius:6px; border:1px solid var(--border-light);">
        <div style="font-size:0.72rem; color:var(--text-muted); font-weight:600;">DB LATENCY</div>
        <div style="font-size:1.2rem; font-weight:700;">${s.db_query_time_ms} ms</div>
      </div>
      <div style="background:var(--bg-surface); padding:12px; border-radius:6px; border:1px solid var(--border-light);">
        <div style="font-size:0.72rem; color:var(--text-muted); font-weight:600;">CPU UTILIZATION</div>
        <div style="font-size:1.2rem; font-weight:700;">${s.cpu_usage_pct.toFixed(1)}%</div>
      </div>
      <div style="background:var(--bg-surface); padding:12px; border-radius:6px; border:1px solid var(--border-light);">
        <div style="font-size:0.72rem; color:var(--text-muted); font-weight:600;">HTTP ERRORS</div>
        <div style="font-size:1.2rem; font-weight:700;">${s.error_count}</div>
      </div>
    </div>

    <h4 style="font-size:1rem; font-weight:700; margin-bottom:8px;">1. Diagnostic Root-Cause Finding</h4>
    <p><b>Leading Hypothesis:</b> <code>${topHyp.hypothesis}</code> (Evidence Match: <b>${topHyp.evidence_score_pct}%</b>)</p>
    <p><b>Supporting Telemetry:</b> ${topHyp.supporting_evidence.join('; ')}</p>

    <h4 style="font-size:1rem; font-weight:700; margin-top:16px; margin-bottom:8px;">2. Recommended Corrective Actions</h4>
    <p>• <b>Immediate Remediation:</b> ${topHyp.recommended_immediate_action}</p>
    <p>• <b>Next Best Diagnostic Check:</b> ${diag.next_best_diagnostic_check.check_title} using tool <code>${diag.next_best_diagnostic_check.recommended_tool}</code></p>
  `;

  document.getElementById('reportMarkdownPreview').innerHTML = html;
}

function downloadOfficialPdf() {
  window.open(`/api/report/pdf/${state.selectedIndex}`, '_blank');
}

// 8. Model Evaluation View
async function renderEvaluationView() {
  try {
    const res = await fetch('/api/evaluation');
    state.evaluationData = await res.json();
    const d = state.evaluationData;

    // Algorithm Comparison Bar
    const ctx = document.getElementById('evalCompareChart');
    if (ctx && d.comparison) {
      const models = Object.keys(d.comparison);
      const holdout = models.map(m => d.comparison[m].test_accuracy);
      const cv = models.map(m => d.comparison[m].cv_mean_accuracy);

      if (state.charts.evalCompare) state.charts.evalCompare.destroy();

      state.charts.evalCompare = new Chart(ctx, {
        type: 'bar',
        data: {
          labels: models,
          datasets: [
            { label: 'Holdout Test', data: holdout, backgroundColor: '#2563EB', borderRadius: 4 },
            { label: '3-Fold CV Mean', data: cv, backgroundColor: '#4F46E5', borderRadius: 4 }
          ]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          scales: { y: { min: 0, max: 1 } }
        }
      });
    }

    // Confusion Matrix Table
    const matrixBox = document.getElementById('evalMatrixBox');
    if (matrixBox && d.comparison && d.comparison['Random Forest']) {
      const rf = d.comparison['Random Forest'];
      let html = '<table class="matrix-table"><thead><tr><th>Actual / Predicted</th>';
      rf.test_classes.forEach(c => { html += `<th>${c.split('/')[0]}</th>`; });
      html += '</tr></thead><tbody>';

      rf.confusion_matrix.forEach((row, i) => {
        html += `<tr><td style="font-weight:700;">${rf.test_classes[i].split('/')[0]}</td>`;
        row.forEach((val, j) => {
          const isDiag = i === j && val > 0;
          html += `<td style="text-align:center; ${isDiag ? 'background:var(--teal-light); font-weight:700; color:var(--teal-healthy);' : ''}">${val}</td>`;
        });
        html += '</tr>';
      });
      html += '</tbody></table>';
      matrixBox.innerHTML = html;
    }

    document.getElementById('evalLimitationsText').textContent = d.limitations_disclosure;
  } catch (err) {
    console.error("Failed to load evaluation data:", err);
  }
}
